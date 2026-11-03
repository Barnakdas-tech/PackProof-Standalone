r"""The metadata models are the boundary a registry response crosses before anything reads it.

Everything here is checked through the models themselves rather than through a client,
because the rules are the models' own: what is absent, what is malformed, what is too long to
display, and what is a shape PackProof refuses to guess about. `test_npm_fetcher.py` covers the
transport; these cover the judgement.

Two distinctions do most of the work, and both come from how npm behaves. Absent is not the
same as wrong: a 2014-era release without `dist.integrity` is a real publish with a missing
field, and it has to come back as `None` rather than fail. A field that *is* present in a shape
that cannot be read is a refusal, because reporting it as absent would say "the registry did not
say" when the registry said something PackProof could not understand.

The second distinction is that nothing here is repaired. The bounds and the character rules
refuse; no validator trims, lowercases, expands a shorthand or supplies a default.

`str(ValidationError)` is asserted on as the user-visible text only where the input is itself
ASCII. It is not ASCII in general: pydantic renders a rejected input with a repr that prints a
printable non-ASCII character verbatim, so a refusal about `'\u00e9'` carries the accent into
the terminal. The structured reason is what a caller surfaces instead — `errors()[i]["msg"]`
is this module's own text, escaped through `ascii()` — and `test_npm_fetcher.py` checks that
the client reads only that field.
"""

import base64
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from packproof.schemas.models import (
    MAX_METADATA_TEXT_LENGTH,
    PackageDistribution,
    PackageRef,
    PackageRepository,
    PackageVersionMetadata,
)

VALID_TARBALL = "https://registry.npmjs.org/express/-/express-4.18.2.tgz"
VALID_SHASUM = "3fabe08296e930c796c19e3c516979386ba9fd59"

#: A bidi override and a zero-width joiner: characters whose whole job is to change how the
#: text around them is displayed or joined, rather than to say anything.
_FORMAT_CHARACTERS = ("\u202e", "\u200d")

#: A newline, an escape and a NUL — the controls that split a report line or start an ANSI
#: sequence.
_CONTROL_CHARACTERS = ("\n", "\x1b", "\x00")


def sri(algorithm: str, byte_length: int) -> str:
    """A syntactically complete SRI hash: base64 of `byte_length` zero bytes, prefixed."""
    return f"{algorithm}-{base64.b64encode(bytes(byte_length)).decode('ascii')}"


# --------------------------------------------------------------------------- untrusted text


def test_a_description_within_the_limit_is_kept_exactly_as_it_arrived() -> None:
    text = "Fast, unopinionated, minimalist web framework"

    assert _metadata(description=text).description == text


def test_a_description_one_character_over_the_limit_is_refused_for_its_length() -> None:
    with pytest.raises(ValidationError, match="characters, above"):
        _metadata(description="x" * (MAX_METADATA_TEXT_LENGTH + 1))


def test_a_pathological_description_is_described_by_size_rather_than_quoted_in_full() -> None:
    """A two-megabyte field is refused for being two megabytes. Repeating it back is not the
    diagnosis, and a refusal that is itself two megabytes cannot be read in a terminal."""
    hostile = "a" * 2_000_000

    with pytest.raises(ValidationError) as excinfo:
        _metadata(description=hostile)

    message = str(excinfo.value)
    assert "2000000 characters" in message
    assert len(message) < 1_000
    assert message.isascii()


def test_a_pathological_repository_field_is_refused_the_same_way() -> None:
    with pytest.raises(ValidationError) as excinfo:
        PackageRepository(directory="b" * 2_000_000)

    assert str(excinfo.value).isascii()
    assert "b" * 1_000 not in str(excinfo.value)


@pytest.mark.parametrize("character", _CONTROL_CHARACTERS + _FORMAT_CHARACTERS)
def test_a_character_that_changes_how_a_line_displays_is_refused(character: str) -> None:
    """`Cc` controls move the terminal; `Cf` format characters reorder or hide what a reviewer
    reads. Neither belongs in a line of a report, and neither is data anyone intends."""
    with pytest.raises(ValidationError, match="would change how a report line is displayed"):
        _metadata(description=f"express{character}is a framework")


def test_a_lone_surrogate_is_refused_because_it_cannot_be_written_to_a_file() -> None:
    # The surrogate is a code literal, not prose: pytest rewrites this module's source, and a
    # real lone surrogate in a docstring becomes a constant the interpreter cannot encode,
    # which fails collection instead of failing an assertion.
    with pytest.raises(ValidationError, match="Unicode Cs character"):
        _metadata(description="express\ud800is a framework")


def test_ordinary_non_ascii_text_is_kept_and_is_not_a_refusal() -> None:
    """A Japanese description is the registry's honest answer. Refusing it would refuse the
    package, and transliterating it would be PackProof editing evidence."""
    text = "高速で、意見にとらわれない、ミニマリストなウェブフレームワーク"

    metadata = _metadata(description=text)

    assert metadata.description == text
    assert metadata.model_dump()["description"] == text


def test_the_reason_a_validator_gives_for_non_ascii_input_is_still_ascii() -> None:
    """The value may be legitimate; the reason PackProof gives about it may not be
    undisplayable.

    Every refusal message in this module quotes what it needs through `ascii()`, so the
    `msg` a caller surfaces is ASCII whatever the field contained. `str(ValidationError)` is
    *not*, and the difference is the reason for the rule: pydantic appends its own rendering of
    the input value, and that rendering prints a printable non-ASCII character verbatim. Only
    the structured reason crosses a boundary into a terminal or a report.
    """
    with pytest.raises(ValidationError) as excinfo:
        _metadata(description="x" * (MAX_METADATA_TEXT_LENGTH + 1) + "\u00e9")

    assert not str(excinfo.value).isascii()
    for item in excinfo.value.errors():
        assert "".join(str(part) for part in item["loc"]).isascii()
        assert item["msg"].isascii()
    assert "2050 characters" in excinfo.value.errors()[0]["msg"]


# --------------------------------------------------------------------------- repository


def test_a_repository_keeps_its_url_type_and_directory_apart() -> None:
    repository = PackageRepository(
        url="git+https://github.com/expressjs/express.git", kind="git", directory="packages/x"
    )

    assert repository.kind == "git"
    assert repository.directory == "packages/x"


def test_a_repository_needs_only_a_url() -> None:
    """`{url, type}` and `{url}` are both real publishes, and neither is a gap PackProof
    should invent a type for."""
    repository = PackageRepository(url="https://github.com/expressjs/express")

    assert repository.kind is None
    assert repository.directory is None


def test_a_repository_shorthand_is_stored_without_a_host_being_invented() -> None:
    """npm documents `repository: "user/repo"` as a string. Expanding that into
    `https://github.com/user/repo` would make PackProof the author of the provenance claim it
    is supposed to be checking, so a shorthand is kept exactly as it arrived."""
    assert PackageRepository(url="expressjs/express").url == "expressjs/express"


def test_a_claimed_repository_without_a_url_is_incomplete_rather_than_absent() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        PackageRepository.model_validate({"type": "git"})


def test_a_repository_directory_is_text_and_is_not_treated_as_a_place_to_write() -> None:
    """`directory` is whatever the publisher wrote in `package.json`. It is stored as a claim;
    nothing in Stage 2.2 resolves it against the filesystem."""
    assert PackageRepository(url="https://example.invalid/x", directory="../../etc").directory == (
        "../../etc"
    )


# --------------------------------------------------------------------------- distribution


@pytest.mark.parametrize(
    ("algorithm", "byte_length"), [("sha1", 20), ("sha256", 32), ("sha384", 48), ("sha512", 64)]
)
def test_every_sri_algorithm_npm_may_name_is_accepted_at_its_own_length(
    algorithm: str, byte_length: int
) -> None:
    assert _distribution(integrity=sri(algorithm, byte_length)).integrity == sri(
        algorithm, byte_length
    )


def test_integrity_absent_is_not_integrity_invalid() -> None:
    """`vue` 0.8.6 was published before npm wrote SRI hashes: in 1,288 measured version
    entries it is the one without `integrity`. Its artifact is described by `shasum` alone,
    and the model has to be able to say that truthfully instead of failing the lookup."""
    distribution = _distribution()

    assert distribution.integrity is None
    assert distribution.shasum is None


@pytest.mark.parametrize(
    ("integrity", "reason"),
    [
        ("md5-9ed3dbeb79e8e6d48995b8e9eb74b948", "not 'algorithm-base64'"),
        ("sha512", "not 'algorithm-base64'"),
        ("sha512-not base64!", "does not decode"),
        ("sha1-" + "A" * 20, "decodes to 15 bytes, and sha1 is 20"),
        ("sha256-" + "A" * 44, "decodes to 33 bytes, and sha256 is 32"),
        ("sha384-" + "A" * 20, "decodes to 15 bytes, and sha384 is 48"),
        (f"{sri('sha512', 64)} {sri('sha256', 32)}", "does not decode"),
    ],
)
def test_a_hash_that_cannot_be_compared_against_real_bytes_is_refused(
    integrity: str, reason: str
) -> None:
    with pytest.raises(ValidationError, match=reason.replace("(", r"\(")):
        _distribution(integrity=integrity)


def test_base64_that_decodes_but_is_not_the_canonical_spelling_is_refused() -> None:
    """`AB==` decodes to the same two bytes as `AE==` and is not what an encoder would ever
    write. A digest compared as text has to be the text an encoder produces, or a genuine
    mismatch and a spelling difference become indistinguishable."""
    with pytest.raises(ValidationError, match="canonical base64"):
        _distribution(integrity="sha1-AB==")


def test_a_hash_of_the_wrong_length_for_its_own_algorithm_is_refused() -> None:
    """Canonical base64 that decodes to 63 bytes is not a SHA-512 digest, whatever the
    algorithm prefix claims, and a later stage comparing digests needs the length to be
    settled before the bytes mean anything."""
    with pytest.raises(ValidationError, match="decodes to 63 bytes, and sha512 is 64"):
        _distribution(integrity=sri("sha512", 63))


@pytest.mark.parametrize(
    "shasum",
    [
        "3FABE08296E930C796C19E3C516979386BA9FD59",  # uppercase
        "3fabe08296e930c796c19e3c516979386ba9fd5",  # one digit short
        VALID_SHASUM + "9",  # one digit long
        "zfabe08296e930c796c19e3c516979386ba9fd59",  # not hex
        "",  # present and empty is not the same as absent
        "3fabe08296e930c796c19e3c516979 386ba9fd59",  # a space inside the hex
    ],
)
def test_a_shasum_that_is_not_npm_spelling_is_refused(shasum: str) -> None:
    with pytest.raises(ValidationError, match="40 lowercase hex"):
        _distribution(shasum=shasum)


def test_an_uppercase_shasum_is_refused_because_computed_digests_are_lowercase() -> None:
    """A later stage compares this string against a digest it computes, and computed digests
    are lowercase. Accepting the uppercase spelling would report an integrity failure over a
    spelling, which is worse than refusing it: the report would be wrong about the artifact."""
    with pytest.raises(ValidationError, match="lowercase"):
        _distribution(shasum="A" * 40)


@pytest.mark.parametrize(
    "tarball_url",
    [
        "http://registry.npmjs.org/e/-/e.tgz",
        "ftp://registry.npmjs.org/e/-/e.tgz",
        "file:///etc/passwd",
        "registry.npmjs.org/e/-/e.tgz",
        "/e/-/e.tgz",
        "https://registry.npmjs.org/e/-/e.tgz#fragment",
        "https://user:pass@registry.npmjs.org/e/-/e.tgz",
        "https://:@registry.npmjs.org/e/-/e.tgz",
        "https:///e/-/e.tgz",
        "https://registry.npmjs.org/e /-/e.tgz",
        "https://registry.npmjs.org/\u00e9/-/e.tgz",
        "https://registry.npmjs.org/e/-/\u00e9.tgz",
        "https://registry.npmjs.org/e/-/e.tgz\n",
    ],
)
def test_a_download_target_that_is_not_one_clean_https_url_is_refused(tarball_url: str) -> None:
    with pytest.raises(ValidationError, match="dist.tarball"):
        _distribution(tarball_url=tarball_url)


def test_an_https_url_on_a_host_nobody_chose_is_recorded_without_being_trusted() -> None:
    """`.invalid` is reserved, so nothing here can reach it even by accident, and the model
    does not decide what a foreign host means: this field is data about an artifact, not
    permission to fetch it. Stage 2.2 requests nothing."""
    url = "https://other.example.invalid/x/-/x-1.0.0.tgz"

    assert _distribution(tarball_url=url).tarball_url == url


def test_a_tarball_path_that_looks_like_traversal_is_still_only_a_url_at_this_stage() -> None:
    """Nothing writes to disk from this string in Stage 2.2. The extraction boundary owns that
    decision, and pretending the check happened here would hide the fact that it has not."""
    url = "https://registry.npmjs.org/../../etc/passwd"

    assert "../.." in _distribution(tarball_url=url).tarball_url


def test_a_tarball_url_over_the_text_limit_is_refused_as_a_size_before_its_shape() -> None:
    with pytest.raises(ValidationError, match="characters, above"):
        _distribution(tarball_url="https://registry.npmjs.org/" + "a" * 3_000_000)


def test_a_dist_without_a_tarball_is_incomplete_not_absent() -> None:
    """`tarball` is the one field this stage cannot do without, so a document that omits it
    fails instead of yielding metadata that describes no artifact."""
    with pytest.raises(ValidationError, match="Field required"):
        PackageDistribution.model_validate({"shasum": VALID_SHASUM})


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (PackageDistribution, {"tarball_url": b"\xffnot text"}),
        (PackageDistribution, {"tarball_url": VALID_TARBALL, "integrity": b"sha512-\xff"}),
        (PackageDistribution, {"tarball_url": VALID_TARBALL, "shasum": b"\x00" * 40}),
        (PackageRepository, {"url": VALID_TARBALL, "kind": b"git"}),
        (PackageRepository, {"url": VALID_TARBALL, "directory": b"packages/x"}),
    ],
)
def test_bytes_are_refused_rather_than_decoded_into_text(
    model: type[PackageDistribution | PackageRepository], payload: dict[str, object]
) -> None:
    """Strict mode: a value that is not text does not become text by having an encoding chosen
    for it, because that encoding is a decision PackProof would be making about evidence. Every
    field in this model family is checked the same way, and the byte strings below would all
    decode cleanly in lax mode — which is the point of refusing them."""
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        model.model_validate(payload)


def test_a_number_in_a_text_field_is_refused_rather_than_stringified() -> None:
    with pytest.raises(ValidationError, match="valid string"):
        PackageDistribution.model_validate({"tarball_url": VALID_TARBALL, "shasum": 123})


# --------------------------------------------------------------------------- the version record


def test_the_record_carries_the_reference_it_answers_rather_than_a_copy_of_its_parts() -> None:
    metadata = _metadata()

    assert metadata.reference == PackageRef(name="express", version="4.18.2")
    assert metadata.reference.fullname == "express"


def test_a_reference_supplied_as_a_dictionary_is_still_rebuilt_by_the_reference_rules() -> None:
    """What `strict=True` guards, measured rather than assumed.

    Strict mode refuses a non-text *scalar*: `bytes`, a number and a boolean cannot become a
    name. It does not refuse a mapping in place of a nested model — `model_validate()` feeds
    that dict through `PackageRef` and builds one. So the boundary still holds for a nested
    field, just one layer down: nothing is trusted for arriving under the key `reference`, and
    every name and version inside it is re-checked against the same allowlist the CLI crossed.
    """
    metadata = PackageVersionMetadata.model_validate(
        {
            "reference": {"name": "express", "version": "4.18.2"},
            "distribution": {"tarball_url": VALID_TARBALL},
        }
    )

    assert isinstance(metadata.reference, PackageRef)
    assert isinstance(metadata.distribution, PackageDistribution)
    assert metadata.reference.fullname == "express"


@pytest.mark.parametrize(
    ("reference", "reason"),
    [
        ({"name": "Express", "version": "4.18.2"}, "lowercase"),
        ({"name": "../express", "version": "4.18.2"}, "may contain only"),
        ({"name": "express", "version": "^4.18.2"}, "exact MAJOR.MINOR.PATCH"),
        ({"name": "express", "version": 4.18}, "valid string"),
        ({"name": "express", "version": "4.18.2", "readme": "everything"}, "Extra inputs"),
    ],
)
def test_a_dictionary_claiming_to_be_a_reference_is_worth_no_more_than_its_fields(
    reference: dict[str, object], reason: str
) -> None:
    """The four ways a nested mapping could be wrong, each refused for the reason the field
    itself would be refused for. A model that re-validates its nested dicts cannot be walked
    past by wrapping an unvetted value in the right key name."""
    with pytest.raises(ValidationError, match=reason):
        PackageVersionMetadata.model_validate(
            {"reference": reference, "distribution": {"tarball_url": VALID_TARBALL}}
        )


def test_a_field_nobody_asked_for_is_refused_rather_than_carried_along() -> None:
    """`extra="forbid"` is what makes the client's projection load-bearing. A key that arrives
    by accident — a `readme`, a `maintainers` list, the registry's own `_rev` — is a decision
    about what belongs in a report, and nobody made it."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PackageVersionMetadata.model_validate(
            {
                "reference": PackageRef(name="express", version="4.18.2"),
                "distribution": PackageDistribution(tarball_url=VALID_TARBALL),
                "readme": "# everything about me",
            }
        )


def test_optional_fields_a_response_omits_stay_omitted_in_a_report() -> None:
    metadata = _metadata(description=None, repository=None, published_at=None)

    dumped = metadata.model_dump()
    assert dumped["description"] is None
    assert dumped["repository"] is None
    assert dumped["published_at"] is None


def test_a_timestamp_with_a_zone_is_kept_as_an_instant() -> None:
    published = _metadata(published_at="2022-10-08T20:14:32.495Z").published_at

    assert published == datetime(2022, 10, 8, 20, 14, 32, 495_000, tzinfo=UTC)
    assert published is not None and published.tzinfo is not None


def test_a_timestamp_with_an_offset_is_normalized_to_the_same_instant_it_names() -> None:
    """Registry times are written in the publisher's zone. Comparing them needs an instant,
    not a local reading that means a different minute depending on where it is read."""
    assert _metadata(published_at="2022-10-08T22:14:32.495+02:00").published_at == datetime(
        2022, 10, 8, 20, 14, 32, 495_000, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("published_at", "reason"),
    [
        ("2022-10-08T20:14:32", "wall clock"),
        ("yesterday", "not an ISO 8601 timestamp"),
        ("2022-10-08T20:14:32Zextra", "not an ISO 8601 timestamp"),
        (1_700_000_000, "must be text or absent"),
        (True, "must be text or absent"),
        ([], "must be text or absent"),
        ({"$date": "2022-10-08"}, "must be text or absent"),
    ],
)
def test_a_publication_time_that_is_not_an_instant_is_refused(
    published_at: object, reason: str
) -> None:
    with pytest.raises(ValidationError, match=reason):
        _metadata(published_at=published_at)


def test_a_timestamp_longer_than_any_real_timestamp_is_refused_without_being_parsed() -> None:
    """`fromisoformat` scans its input, so the length is checked first: a field of a
    megabyte costs a comparison instead of a parse."""
    with pytest.raises(ValidationError, match="is not a timestamp"):
        _metadata(published_at="9" * 5_000)


def test_a_validated_record_cannot_be_rewritten_afterwards() -> None:
    """A finding cites these values. A model a later stage can quietly edit is not evidence."""
    metadata = _metadata()

    with pytest.raises(ValidationError):
        metadata.description = "rewritten"  # type: ignore[misc]


def test_two_records_with_the_same_fields_are_equal_and_hash_to_one_entry() -> None:
    """Findings are deduped by package and version, so equality has to mean the fields. The
    frozen config gives the models a `__hash__` built from them, which is what lets a set of
    records be a set of packages."""
    first, second, other = _metadata(), _metadata(), _metadata(description="another")

    assert first == second
    assert len({first, second}) == 1
    assert first != other
    assert len({first, other}) == 2


def test_a_dump_nests_its_models_and_a_json_dump_round_trips_to_the_same_record() -> None:
    """A record is a tree, so `model_dump()` gives back a tree of plain dicts — and reading
    one back rebuilds the typed pieces rather than trusting them.

    `mode="json"` is the shape a report writer needs, and it is checked here rather than later:
    a `datetime` becomes text and comes back as the same instant, and the whole tree serializes
    through `json.dumps` without a custom encoder. The typed model is still the API; a dump is
    a rendering of it, and any name in that rendering passes `PackageRef`'s validators again on
    the way back in.
    """
    metadata = _metadata(published_at="2022-10-08T20:14:32.495Z")
    dumped = metadata.model_dump()

    assert isinstance(dumped["reference"], dict)
    assert isinstance(dumped["distribution"], dict)
    assert PackageVersionMetadata.model_validate(dumped) == metadata

    as_json = metadata.model_dump(mode="json")
    assert isinstance(as_json["published_at"], str)
    assert PackageVersionMetadata.model_validate(as_json) == metadata
    assert json.loads(json.dumps(as_json)) == as_json


# --------------------------------------------------------------------------- helpers


def _distribution(**overrides: object) -> PackageDistribution:
    fields: dict[str, object] = {"tarball_url": VALID_TARBALL}
    fields.update(overrides)
    return PackageDistribution.model_validate(fields)


def _metadata(**overrides: object) -> PackageVersionMetadata:
    fields: dict[str, object] = {
        "reference": PackageRef(name="express", version="4.18.2"),
        "distribution": PackageDistribution(tarball_url=VALID_TARBALL),
    }
    fields.update(overrides)
    return PackageVersionMetadata.model_validate(fields)
