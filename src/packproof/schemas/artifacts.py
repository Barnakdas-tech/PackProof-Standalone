r"""What a completed download is: one artifact, on disk, with a stated integrity outcome.

`docs/architecture.md` makes `schemas/` the boundary external values cross to become PackProof
objects, and the artifact is the first object on the far side of that boundary that is not a
description someone else wrote. `PackageVersionMetadata` records what the registry claimed;
`PackageArtifact` records what PackProof now has, and the two are deliberately not the same type
with the same fields. A model that could be built from the metadata alone would be a model that
claims to describe bytes without ever having read them.

The verification field carries that separation, and it has exactly two values. `VERIFIED` means
the metadata supplied at least one integrity claim, every claim it supplied was checked against
the bytes this process wrote, and all of them held. `UNAVAILABLE` means the metadata supplied no
claim to check — the pre-SRI release whose `dist.integrity` is absent, and whose `dist.shasum` is
too. A mismatch is neither: it raises, and no `PackageArtifact` exists for bytes that failed a
claim. There is no `MISMATCH` value and no `is_verified: bool`, because a value a caller can
compare against is a value a caller can forget to compare, and the two states that must never be
confused — "checked and held" and "nobody said" — are exactly the pair that a boolean collapses.

Making the meaning live in the type is what stops the absent case from being reported as the
verified one, and the enforcement is a model validator rather than a convention: `VERIFIED` is
unconstructable with an empty `checked_claims`, and `UNAVAILABLE` is unconstructable with a
non-empty one. A caller that tried to build a verified artifact without saying what it checked
would be refused by the same mechanism that refuses a malformed package name in `PackageRef`,
which is the point of putting the rule on the field instead of in the code that happens to
produce one.

Nothing in here can hold a download URL. `PackageDistribution.tarball_url` is where the registry's
address lives, and it stays there: an npm tarball URL can carry a signed query string, and a
signed query string is a capability — whoever holds it can fetch the artifact without being the one
who was granted it. So `PackageArtifact` names the artifact by what it *is* (`sha256`, `size_bytes`)
and by where it landed (`directory`, `file_name`), and an artifact's `repr()`, its `str()`,
`model_dump()` and `model_json_schema()` contain no field a URL could be stored in. That is a
stronger property than redacting one, because redaction has to be remembered at every future call
site.

`file_name` is content-addressed and validated as such: `<64 lowercase hex>.tgz`, and the hex must
equal `sha256`. The two fields disagreeing would mean a report naming bytes that are not the ones
the digest describes, so the model refuses the disagreement instead of printing it. `directory` is
required to be absolute for the same reason a workspace path is validated in `config/settings.py`:
a relative location means "wherever the process happens to be standing", and an artifact whose
location depends on the working directory cannot be re-opened by the next stage.

`size_bytes` and `sha256` describe the artifact as it exists on disk, not as the registry
described it in `dist.unpackedSize` or in any other field. npm's own size claims are about the
*unpacked* tree and are written by whoever published it; these two numbers are counted by this
process over the bytes it received, which is the only kind of number an integrity report can be
built on.
"""

import re
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packproof.schemas.models import PackageRef

#: The shape of a content-addressed artifact name: this process's SHA-256, as lowercase hex,
#: with npm's own extension. Enforced on the field so a name that came from anywhere else —
#: including a spelling derived from registry text — cannot be attached to an artifact.
_CONTENT_ADDRESSED_NAME = re.compile(r"[0-9a-f]{64}\.tgz\Z")

#: The digest half on its own, so the field that records it checks the same shape the name is
#: built from rather than a second, slightly different description of it.
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")

#: The closed vocabulary an artifact may say it checked. Anything else is a bug in the code that
#: built the artifact, not a claim a reviewer can act on, so the model refuses it rather than
#: rendering it. `shasum-sha1` is spelled apart from `sri-sha1` on purpose: the same algorithm
#: over the same bytes, arrived at from two different registry fields, and a report that merged
#: them would hide which one the registry actually supplied.
CHECKABLE_CLAIMS = frozenset({"sri-sha1", "sri-sha256", "sri-sha384", "sri-sha512", "shasum-sha1"})

#: The order `checked_claims` is always stored in. A set would render in whichever order the
#: hasher happened to fill, and two identical downloads would then produce two different-looking
#: artifacts in a report that is supposed to be diffable. It is also ascending in strength, which
#: is why the code that *prefers* the strongest claim reads this tuple backwards rather than
#: keeping a second ranking of the same names that could disagree with the first.
CHECKED_CLAIM_ORDER = ("shasum-sha1", "sri-sha1", "sri-sha256", "sri-sha384", "sri-sha512")


class ArtifactVerification(StrEnum):
    """The only two integrity outcomes a downloaded artifact can carry.

    Kept apart from a boolean because the absent case is not a failure and must not read like one.
    A package published before npm wrote SRI strings has no `dist.integrity`; its bytes are
    exactly as real as a verified package's, and what is missing is the registry's claim about
    them. `False` would put that on the same footing as a mismatch, which is the mistake every
    later stage in this project exists to avoid.
    """

    VERIFIED = "VERIFIED"
    UNAVAILABLE = "UNAVAILABLE"


class PackageArtifact(BaseModel):
    """One package tarball on disk, named for its own contents, with its integrity outcome stated.

    Produced only by a download that completed, streamed to a file PackProof created, hashed while
    it was written and published by link. Every field describes the bytes that are now in the
    workspace; none of them is copied from a registry response, because a registry response
    describes what the publisher intended and this model is the evidence about what arrived.

    `checked_claims` is empty exactly when `verification` is `UNAVAILABLE`, and a `ValueError` is
    the enforcement rather than a comment. There is no field that means "trusted".
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    reference: PackageRef = Field(
        description="The validated reference this artifact was downloaded for, so the "
        "bytes and the request they answer travel together."
    )
    directory: Path = Field(
        description="The workspace directory the artifact was published into, absolute."
    )
    file_name: str = Field(description="`<sha256>.tgz`: the artifact's own content-addressed name.")
    sha256: str = Field(
        description="SHA-256 of the bytes this process wrote, computed while streaming them."
    )
    size_bytes: int = Field(
        ge=0,
        description="Length of those bytes, counted by this process, not claimed by the registry.",
    )
    verification: ArtifactVerification = Field(
        description="VERIFIED if the metadata supplied claims and all of them held; "
        "UNAVAILABLE if it supplied none."
    )
    checked_claims: tuple[str, ...] = Field(
        description="Which integrity claims were available to check and did hold, from a "
        "closed vocabulary, in a fixed order."
    )

    @field_validator("sha256")
    @classmethod
    def _digest_is_lowercase_hex_of_one_sha256(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(
                f"a recorded sha256 must be 64 lowercase hex characters, and "
                f"{ascii(value[:16])}... is not"
            )
        return value

    @field_validator("file_name")
    @classmethod
    def _file_name_is_content_addressed(cls, value: str) -> str:
        if _CONTENT_ADDRESSED_NAME.fullmatch(value) is None:
            raise ValueError(
                f"an artifact file name must be 64 lowercase hex characters followed by "
                f"'.tgz', and {ascii(value[:24])}... is not one PackProof published"
            )
        return value

    @field_validator("directory")
    @classmethod
    def _directory_is_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(
                "an artifact directory must be absolute; a relative one names a location that "
                "depends on which directory the process was standing in"
            )
        return value

    @field_validator("checked_claims")
    @classmethod
    def _claims_are_the_closed_vocabulary(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [claim for claim in value if claim not in CHECKABLE_CLAIMS]
        if unknown:
            raise ValueError(
                f"{[ascii(claim) for claim in unknown]} is not a claim PackProof knows how to "
                f"check; the vocabulary is {sorted(CHECKABLE_CLAIMS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError("a checked claim is recorded once, not once per spelling of itself")
        ordered = tuple(claim for claim in CHECKED_CLAIM_ORDER if claim in value)
        if ordered != value:
            raise ValueError(
                f"checked_claims must be in the fixed order {ordered}, not "
                f"{[ascii(claim) for claim in value]}"
            )
        return value

    @model_validator(mode="after")
    def _name_and_digest_and_verification_agree(self) -> Self:
        """The three cross-field facts that make this model an evidence object.

        The name must be the digest it reports (otherwise a report points at bytes that are not the
        ones it describes), and the verification must match the claims actually checked (otherwise
        `VERIFIED` becomes a label a caller can attach to an unchecked file).
        """
        if self.file_name != f"{self.sha256}.tgz":
            raise ValueError(
                f"{self.file_name!r} is not named for the digest it reports "
                f"({self.sha256[:12]!r}...)"
            )
        if self.verification is ArtifactVerification.VERIFIED and not self.checked_claims:
            raise ValueError(
                "an artifact cannot be VERIFIED with no claim checked: verification is a claim "
                "about a comparison, and there was none"
            )
        if self.verification is ArtifactVerification.UNAVAILABLE and self.checked_claims:
            raise ValueError(
                f"an artifact cannot be UNAVAILABLE while reporting {len(self.checked_claims)} "
                "checked claim(s); absent means the metadata supplied nothing to check"
            )
        return self

    @property
    def path(self) -> Path:
        """Where the artifact is. Derived, so it cannot disagree with the fields above."""
        return self.directory / self.file_name
