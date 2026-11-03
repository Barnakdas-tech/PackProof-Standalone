"""The artifact model: an evidence object, and the validators that keep it one.

`PackageArtifact` is the first object in this project that is not a description somebody else
wrote, so the interesting failures are the ones where a field could *disagree with the others* —
a name that is not the digest it reports, a `VERIFIED` that checked nothing, a `UNAVAILABLE` that
lists claims anyway. Each of those is a report that would read as evidence while describing
different bytes than the digest does, so the model refuses it rather than rendering it.

The second thing these tests pin is the property the download path depends on: there is no field
a URL could be stored in. That is checked against the model's own schema (`model_fields`,
`model_dump`, `model_json_schema`), not against a redaction routine, because redaction has to be
remembered at every future call site and an absent field does not.

`strict=True` is part of the design rather than a default, so it is tested: a `str` where a
`Path` belongs, a `list` where a tuple of claims belongs, and the bare word `"VERIFIED"` where
the enum member belongs are all refusals. A model that quietly coerced them would let a caller
build an artifact out of a dictionary read off disk.
"""

import hashlib
import json
from pathlib import Path

import pydantic
import pytest

from packproof.registry.npm import parse_npm_reference
from packproof.schemas.artifacts import (
    CHECKABLE_CLAIMS,
    CHECKED_CLAIM_ORDER,
    ArtifactVerification,
    PackageArtifact,
)
from packproof.schemas.models import PackageRef

PAYLOAD = b"a tarball, hypothetically speaking"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
REF: PackageRef = parse_npm_reference("left-pad@1.3.0")


def _artifact(**overrides: object) -> PackageArtifact:
    """One construction site for the whole file, so an override says exactly what broke."""
    fields: dict[str, object] = {
        "reference": REF,
        "directory": Path("/workspace"),
        "file_name": f"{DIGEST}.tgz",
        "sha256": DIGEST,
        "size_bytes": len(PAYLOAD),
        "verification": ArtifactVerification.VERIFIED,
        "checked_claims": ("sri-sha512",),
    }
    fields.update(overrides)
    return PackageArtifact(**fields)


def _rejection(**overrides: object) -> str:
    """The message a refused construction produces, so tests can assert on what it *says*."""
    with pytest.raises(pydantic.ValidationError) as caught:
        _artifact(**overrides)
    return str(caught.value)


def _error_type(**overrides: object) -> str:
    """The pydantic error code for one refused field, so strict-mode refusals are asserted by
    their mechanism and not by whichever words a message happens to contain."""
    with pytest.raises(pydantic.ValidationError) as caught:
        _artifact(**overrides)
    return str(caught.value.errors()[0]["type"])


# --------------------------------------------------------------------------- the two outcomes

VERIFIED_WITH_CLAIMS = (ArtifactVerification.VERIFIED, ("sri-sha1",))
UNAVAILABLE_WITHOUT = (ArtifactVerification.UNAVAILABLE, ())


def test_the_vocabulary_has_exactly_two_outcomes_and_no_mismatch_member() -> None:
    """A mismatch raises, so a third enum value would be a state nothing can ever produce —
    and a `MISMATCH` value is precisely what a caller could forget to compare against."""
    assert [member.value for member in ArtifactVerification] == ["VERIFIED", "UNAVAILABLE"]
    assert not hasattr(ArtifactVerification, "MISMATCH")


def test_there_is_no_field_that_means_trusted() -> None:
    """`is_verified: bool` would put "checked and held" and "nobody said" one `False` apart."""
    assert "is_verified" not in PackageArtifact.model_fields
    assert "trusted" not in PackageArtifact.model_fields
    assert "verified" not in [
        name for name in PackageArtifact.model_fields if name != "verification"
    ]


def test_a_string_where_a_path_belongs_is_refused_rather_than_coerced() -> None:
    """`strict=True` is part of the design: a model that quietly turned `"/workspace"` into a
    `Path` would also accept a dictionary read off disk."""
    assert _error_type(directory="/workspace") == "is_instance_of"


def test_a_list_where_a_tuple_of_claims_belongs_is_refused_rather_than_coerced() -> None:
    assert _error_type(checked_claims=["sri-sha1"]) == "tuple_type"


def test_a_number_for_size_where_an_int_belongs_is_refused() -> None:
    assert _error_type(size_bytes=str(len(PAYLOAD))) == "int_type"


def test_the_enum_member_is_required_not_its_spelling() -> None:
    assert _error_type(verification="UNAVAILABLE") == "is_instance_of"


@pytest.mark.parametrize(
    ("verification", "claims"),
    [VERIFIED_WITH_CLAIMS, UNAVAILABLE_WITHOUT],
)
def test_the_two_legal_pairings_construct(
    verification: ArtifactVerification, claims: tuple
) -> None:
    artifact = _artifact(verification=verification, checked_claims=claims)
    assert artifact.verification is verification
    assert artifact.checked_claims == claims


def test_verified_with_nothing_checked_is_refused() -> None:
    message = _rejection(verification=ArtifactVerification.VERIFIED, checked_claims=())
    assert "cannot be VERIFIED with no claim checked" in message


def test_unavailable_while_reporting_claims_is_refused() -> None:
    message = _rejection(
        verification=ArtifactVerification.UNAVAILABLE, checked_claims=("sri-sha1",)
    )
    assert "cannot be UNAVAILABLE while reporting 1 checked claim" in message


# --------------------------------------------------------------------------- name and digest


def test_a_name_that_is_not_the_digest_it_reports_is_refused() -> None:
    other = hashlib.sha256(b"other bytes entirely").hexdigest()
    message = _rejection(file_name=f"{other}.tgz")
    assert "is not named for the digest it reports" in message


@pytest.mark.parametrize(
    ("file_name", "reason"),
    [
        ("not-content-addressed.tgz", "a name nobody published"),
        (f"{DIGEST.upper()}.tgz", "uppercase hex is not the spelling this process writes"),
        (f"{DIGEST}.tar.gz", "the extension is npm's, and only npm's"),
        (f"{DIGEST[:63]}.tgz", "one hex character short of a digest"),
        (f"{DIGEST}x.tgz", "one hex character too many"),
        ("", "an empty name"),
        (f"{DIGEST}.tgz/", "a name carrying a separator is not a name"),
        ("../../escape.tgz", "a name that climbs out of the workspace is not a name either"),
        (f"{DIGEST}.tgz\ntgz", "a separator is a separator whether or not anyone can see it"),
    ],
)
def test_only_the_content_addressed_spelling_is_a_file_name(file_name: str, reason: str) -> None:
    assert "must be 64 lowercase hex characters followed by '.tgz'" in _rejection(
        file_name=file_name
    ), reason


@pytest.mark.parametrize(
    "sha256",
    [
        DIGEST.upper(),
        DIGEST[:-1],
        DIGEST + "0",
        "z" * 64,
        "",
    ],
)
def test_the_recorded_digest_is_exactly_one_sha256_in_lowercase_hex(sha256: str) -> None:
    assert "must be 64 lowercase hex characters" in _rejection(sha256=sha256)


def test_the_path_is_derived_so_it_cannot_disagree_with_the_fields() -> None:
    artifact = _artifact()
    assert artifact.path == artifact.directory / artifact.file_name
    assert "path" not in PackageArtifact.model_fields


# --------------------------------------------------------------------------- the claim vocabulary


def test_every_recordable_claim_is_in_the_stored_order() -> None:
    """A claim the validator would accept but the order does not mention would be sorted out of
    every artifact, so the two vocabularies are asserted to cover each other."""
    assert set(CHECKED_CLAIM_ORDER) == set(CHECKABLE_CLAIMS)
    assert len(CHECKED_CLAIM_ORDER) == len(CHECKABLE_CLAIMS)


def test_the_stored_order_is_ascending_in_strength() -> None:
    """`npm_download` reads this tuple backwards to name the strongest failed claim. If the
    order ever stops being ascending, that reading silently starts naming the weakest."""
    assert CHECKED_CLAIM_ORDER[0] == "shasum-sha1"
    assert CHECKED_CLAIM_ORDER[-1] == "sri-sha512"
    assert CHECKED_CLAIM_ORDER.index("sri-sha1") < CHECKED_CLAIM_ORDER.index("sri-sha256")


def test_an_unknown_claim_is_refused_and_the_message_names_the_vocabulary() -> None:
    message = _rejection(checked_claims=("md5",))
    assert "is not a claim PackProof knows how to check" in message
    assert "sri-sha512" in message


def test_the_sha1_spelling_from_sri_is_not_the_sha1_spelling_from_shasum() -> None:
    """Same algorithm over the same bytes, arrived at from two different registry fields. A
    report that merged them would hide which one the registry actually supplied."""
    assert _artifact(checked_claims=("sri-sha1",)).checked_claims == ("sri-sha1",)
    assert _artifact(checked_claims=("shasum-sha1",)).checked_claims == ("shasum-sha1",)


def test_claims_are_stored_in_the_fixed_order_whatever_order_they_arrive_in() -> None:
    """Two identical downloads must produce two identical-looking artifacts, or a report stops
    being diffable."""
    forward = _artifact(checked_claims=("shasum-sha1", "sri-sha256", "sri-sha512"))
    assert forward.checked_claims == ("shasum-sha1", "sri-sha256", "sri-sha512")
    assert "must be in the fixed order" in _rejection(
        checked_claims=("sri-sha512", "sri-sha256", "shasum-sha1")
    )


def test_a_claim_recorded_twice_is_refused() -> None:
    assert "recorded once, not once per spelling of itself" in _rejection(
        checked_claims=("sri-sha1", "sri-sha1")
    )


# --------------------------------------------------------------------------- where the artifact lives


def test_a_relative_directory_is_refused() -> None:
    """A relative location means "wherever the process happens to be standing", and the next
    stage could not re-open the file from it."""
    assert "must be absolute" in _rejection(directory=Path("workspace"))


# --------------------------------------------------------------------------- no place for a URL


def test_the_model_has_no_field_a_download_url_could_be_stored_in() -> None:
    """Not "the URL is redacted" — the URL has nowhere to live. A redaction rule has to be
    remembered at every future call site; an absent field does not."""
    assert sorted(PackageArtifact.model_fields) == [
        "checked_claims",
        "directory",
        "file_name",
        "reference",
        "sha256",
        "size_bytes",
        "verification",
    ]


def test_no_rendering_of_an_artifact_can_contain_a_url() -> None:
    """Every channel a report might use, checked over one instance rather than asserted about
    the class."""
    artifact = _artifact()
    renderings = (
        repr(artifact),
        str(artifact),
        json.dumps(artifact.model_dump(mode="json"), default=str),
        artifact.model_dump_json(),
        json.dumps(artifact.model_json_schema(), default=str),
    )
    for rendering in renderings:
        assert "http" not in rendering
        assert "tgz?" not in rendering
        assert "token" not in rendering.lower()


def test_the_artifact_is_frozen_so_a_report_cannot_be_edited_afterwards() -> None:
    artifact = _artifact()
    with pytest.raises(pydantic.ValidationError):
        artifact.size_bytes = 0  # type: ignore[misc]


def test_an_extra_field_is_refused_rather_than_silently_dropped() -> None:
    """`extra="forbid"` is what makes a construction from an unreviewed dictionary fail loudly —
    including a dictionary that tried to carry `tarball_url`."""
    assert "Extra inputs are not permitted" in _rejection(tarball_url="https://x.invalid/a.tgz")
