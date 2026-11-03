r"""Shared models: the boundary external input crosses to become a PackProof object.

`docs/architecture.md` makes `schemas/` the place where registry JSON, `package.json`
fields and command-line references turn into typed models, and names that conversion the
validation boundary for everything the engine reads. Two families live here: `PackageRef`,
which is what a request was *for*, and the metadata models — `PackageRepository`,
`PackageDistribution`, `PackageVersionMetadata` — which are what a registry answered. Both
enforce their rules on the fields themselves rather than only inside the code that happens
to produce one.

Enforcing the grammar on the field is what keeps a second entry route safe. The metadata
client reads a name and a version out of a registry response and constructs `PackageRef`
directly, and that response describes a package anyone may publish under any spelling. A
permissive model would hand an attacker-controlled string to whatever builds a request path
or a workspace directory from it, and the vetting the CLI performed would simply not have
happened on that path.

Optionality in the metadata models is a claim about the registry, not a convenience, and it
is measured rather than assumed. Across 1,288 version entries sampled from six published
packages, `dist.tarball` and `dist.shasum` appear in every one, `dist.integrity` in 1,287
(`vue` 0.8.6 predates it) and a per-version `repository` in 1,249. So a field that is
sometimes absent is typed `| None` and is reported as absent, while a field that is present
and malformed fails validation. Nothing in this module supplies a value the registry did
not send.

Names and versions are admitted by allowlist, not by a denylist of known-bad spellings,
because both are used downstream as path segments. `..`, `/`, `\`, `%`, NUL, whitespace
and every other punctuation mark never reach a consumer that would have to escape them
correctly in order to stay safe. A version may carry build metadata after a `+`, which is
legal semver; `+` is path-safe but is a reserved character in a URL query, so the registry
client that builds a request must percent-encode it rather than assume a bare `+` survives
form encoding.

Digits are matched as `[0-9]` rather than `\d` throughout. In `str` patterns Python's
`\d` also matches Unicode decimal digits, so `٣` or fullwidth `１` would satisfy a
`\d`-based version grammar and be sent to the registry as if it were the digit the person
typing meant. Same-looking is not the same, and an integrity tool that quietly asks about
a different string than the one a reviewer read has lost the comparison it exists to make.

Every text field is validated in strict mode, so a value must already be text. In lax
mode pydantic decodes `bytes` and `bytearray` into a `str`, which would let a non-text
value cross the boundary by having PackProof pick a decoding for it — a field lifted out
of a binary parser, or a payload read as bytes, would silently become a name. Strict mode
refuses them, and every field validator still runs on what is left. A `str` subclass is
still normalised to exact `str`, so a subclass carrying a hostile `__eq__` cannot make two
different packages compare equal or collapse into one deduped finding.

Where strictness stops is worth naming rather than finding out later: it is about scalars.
A mapping supplied for a nested model is not refused — pydantic feeds it to that model and
runs its validators over its keys. So `reference` arriving as a dict is still checked name
by name, character by character, and the boundary holds one layer down; but nothing in here
is trusted because of the key it arrived under, and the client constructs the nested models
itself rather than handing over raw mappings.

Numeric version components are bounded to `Number.MAX_SAFE_INTEGER`, mirroring node-semver:
its constructor throws `TypeError('Invalid major version')` — and the same for minor and
patch — when a component exceeds it, so `99999999999999999999.0.0` satisfies the semver
grammar yet names an artifact npm's own tooling cannot handle. The bound covers the three
core components and nothing else, because that is exactly what upstream bounds; prerelease
and build identifiers are not magnitude-checked there and are not magnitude-checked here.

Borrowing a restriction from node-semver is not the same as accepting everything it accepts,
and the two sets are not equal. What is implemented here is a *subset* of upstream's strict
parsing: the same `0|[1-9]\d*` core components, the same prerelease identifiers and the same
`[0-9A-Za-z-]+` build identifiers, minus the optional `v` prefix that its `FULLPLAIN`
carries, minus the `trim()` its constructor performs, and minus the normalization — which
rewrites `v1.2.3` to `1.2.3` and leaves `+build` out of `SemVer.version` altogether. Its
loose mode is wider again (`NUMERICIDENTIFIERLOOSE` is `\d+`, so `1.2.03` parses there), and
no loose mode is implemented. Every one of those refusals is the same decision seen from the
other side: a reference that had to be rewritten before it could be analyzed is a reference
that does not name the artifact a person typed, and stripping a prefix is a resolution step,
not a parse.

Every message raised here is ASCII-only, and every value it quotes goes through `ascii()`
rather than `repr()`. A refusal is printed to a terminal whose encoding PackProof does not
choose: under `LC_ALL=C` a message containing one Arabic-Indic digit — or one em dash that
the source wrote deliberately — raises `UnicodeEncodeError` while being displayed, so the
error about bad input becomes a worse error with no input named. `ascii()` escapes both the
smuggled character and the message's own punctuation into `\uXXXX` form, which reads the
same to a reviewer and encodes in any locale.

That covers what this module writes, not what pydantic wraps around it: the outer rendering
of a `ValidationError` appends its own repr of the rejected input and prints a printable
non-ASCII character verbatim, so `str(error)` is only as displayable as the value inside it.
The escaping has to be re-applied wherever a refusal leaves this module, which is what the
registry client does — it surfaces the structured reason from `errors()`, never the rendered
exception.
"""

import base64
import re
import unicodedata
from datetime import datetime
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

#: npm's own ceiling for a package name, scope included.
MAX_NAME_LENGTH = 214

#: node-semver, the implementation the npm registry runs, refuses to parse a version
#: string longer than this. Bounding the field to the same number means a version of any
#: size an attacker or a mistyped shell quote supplies costs a fixed amount to validate,
#: and never reaches a log line or a report file at full length.
MAX_VERSION_LENGTH = 256

#: JavaScript's `Number.MAX_SAFE_INTEGER`, and the ceiling node-semver applies to a
#: version's major, minor and patch before it throws `TypeError`. The value is exact and
#: is compared against parsed integers, so a 40-digit component is refused for the same
#: reason the registry's own tooling refuses it: it cannot round-trip through a number.
MAX_SEMVER_COMPONENT = 9_007_199_254_740_991

#: Names npm refuses because they collide with the layout of an installed tree.
RESERVED_NAMES = frozenset({"node_modules", "favicon.ico"})

#: Ceiling on a piece of untrusted display text taken out of a registry response.
#: Measured across 1,288 real version entries in six published packages, a `description`
#: peaked at 124 characters and a repository URL at 62, so this is headroom rather than a
#: limit anyone normal reaches. It exists because the text is whoever-published-it's text
#: and it ends up in a report: an unbounded field lets a registry response size a PackProof
#: output line, and `readme` — the field that genuinely runs to megabytes — is therefore
#: not modeled at all.
MAX_METADATA_TEXT_LENGTH = 2_048

#: The hash algorithms SRI names, and the byte length each one's digest must decode to.
#: npm emits `sha512-…` for every version of a modern package; the others are listed
#: because SRI permits them and a digest of the wrong length is a malformed value, not a
#: different algorithm.
_SRI_DIGEST_LENGTHS = {"sha1": 20, "sha256": 32, "sha384": 48, "sha512": 64}

#: npm's own spelling of a SHA-1 digest: 40 lowercase hex digits, as `dist.shasum` has
#: carried it in every one of the 1,288 measured entries.
_SHA1_HEX = re.compile(r"[0-9a-f]{40}")

# Unicode categories that have no business in a report line: Cc control (NUL, ESC, a
# newline that splits a log), Cf format (zero-width joiners and bidi overrides — the
# characters that let visible text mean something else, the "Trojan source" class), and
# Cs lone surrogates, which cannot be encoded to UTF-8 and so cannot be written to any
# file. Other non-ASCII is kept verbatim: a Japanese description is the registry's honest
# answer, and refusing it would refuse the package.
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})

# The three name-shaped things npm defines are not one grammar, so they are not one
# pattern. npm's documented rule is that "the names of scoped packages can begin with a
# dot or an underscore. This is not permitted without a scope." — so the character *set*
# is shared (`_NAME_CHARACTERS`) and the leading character is decided per kind:
#
#   unscoped package name  must begin with a letter or digit  (_UNSCOPED_NAME_START)
#   scoped package name    may begin with '.', '_' or an alphanumeric
#   scope namespace        must begin with a letter or digit
#
# The namespace keeps the stricter rule deliberately. npm's validator applies its
# leading-character check to the scope as well, and a namespace is the directory level
# above the package in every cache layout the analyzer will build, so a hidden namespace
# would buy nothing and cost one predictable path segment.
_NAME_CHARACTERS = re.compile(r"[a-z0-9._-]+")
_UNSCOPED_NAME_START = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SCOPE_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9._-]*")

# A scoped name may start with a dot, which makes the two names that mean "here" and "up"
# reachable by a legal npm spelling. They are refused on their own terms rather than by a
# charset that would also have to refuse every legitimate `.hidden` package. `...` is kept:
# it carries no filesystem meaning on any supported platform, and refusing it would be a
# rule of ours rather than npm's.
_NON_NAMES = frozenset({".", ".."})

# Exact `MAJOR.MINOR.PATCH`, with an optional prerelease tail and an optional build-metadata
# tail, following semver.org and node-semver — the parser the npm registry itself runs,
# whose grammar is `( '+' build )?` over dot-separated `[0-9A-Za-z-]+` identifiers.
#
# Leading zeroes in a *numeric* identifier are impossible by construction, and that matches
# node-semver's default parsing rather than exceeding it: upstream builds both its strict
# core and its prerelease identifiers from `NUMERICIDENTIFIER = 0|[1-9]\d*`, so `1.2.03` and
# `1.2.3-01` are invalid there as here, while `BUILDIDENTIFIER = [0-9A-Za-z-]+` allows
# `1.2.3+001` in both. The asymmetry between prerelease and build is semver's, not a gap
# here. The one place upstream is wider is loose mode, where `NUMERICIDENTIFIERLOOSE = \d+`
# admits the leading zero; no loose mode is implemented, deliberately.
#
# Both patterns below are used with `fullmatch`, never with `^…$` anchors: `$` also matches
# before a trailing newline, so a version typed as `1.2.3\n` would validate and carry a line
# break into whatever path segment is later built from it.
_SEMVER_CORE = r"0|[1-9][0-9]*"
_SEMVER_IDENTIFIER = r"0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*"
_SEMVER_BUILD_IDENTIFIER = r"[0-9A-Za-z-]+"
_EXACT_SEMVER = re.compile(
    rf"(?:{_SEMVER_CORE})\.(?:{_SEMVER_CORE})\.(?:{_SEMVER_CORE})"
    rf"(?:-(?:{_SEMVER_IDENTIFIER})(?:\.(?:{_SEMVER_IDENTIFIER}))*)?"
    rf"(?:\+{_SEMVER_BUILD_IDENTIFIER}(?:\.{_SEMVER_BUILD_IDENTIFIER})*)?"
)


class PackageRef(BaseModel):
    """A package pinned to one exact published version.

    `scope` is held without its `@`: the sigil is npm's spelling of "this name is
    namespaced", not part of the identity, and a field that carries it invites a
    comparison against a value that does not. `fullname` is the single place that puts
    the `@` back, which is what a registry request path or a report needs.

    `version` accepts a prerelease and accepts build metadata, because both are legal
    semver and both name a published artifact. A prerelease (`1.0.0-rc.1`) and a build
    (`1.0.0+build.5`) are different strings over different bytes, so this model treats
    them as different references.

    That last sentence is where a hazard lives, and naming it is cheaper than meeting it:
    semver *precedence* ignores build metadata, so `1.0.0+a` and `1.0.0+b` are the same
    version to node-semver and to any sort or dedupe built on semver comparison, while
    they are two distinct artifacts to PackProof. Anything that groups, sorts or keys
    findings by version must use this string, not a semver comparison, or two tarballs
    collapse into one and the report describes neither. `PackageRef.__eq__` and
    `__hash__` are inherited from the field values, so they already keep them apart.

    `+` is path-safe but is reserved in a URL query, where form encoding turns it into a
    space; the registry client that builds a request owns that encoding. Refusing the
    character here would have been mistaking an encoding problem for invalid input.

    Every field is validated strictly and every value is validated after coercion, so
    reaching this model with something that is not text is a refusal rather than a
    decoding decision: `bytes`, `bytearray`, numbers and booleans never become a name.

    A numeric major, minor or patch beyond `MAX_SEMVER_COMPONENT` is refused because
    node-semver — what npm's own tooling parses with — throws on it; prerelease and build
    identifiers carry no such bound here, matching upstream. That ceiling is one restriction
    taken from node-semver, not a claim that the two accepted sets are the same. This model
    takes an exact, unprefixed `MAJOR.MINOR.PATCH` whose numeric identifiers have no leading
    zeroes and whose text is neither trimmed nor rewritten. node-semver's default parsing is
    wider in three places: its strict `FULL` pattern accepts a leading `v`, its constructor
    calls `trim()` before matching, and its `format()` normalizes the result — `v1.2.3` comes
    back as `1.2.3`, and build metadata is dropped from the normalized string entirely. Its
    loose mode is wider still, accepting leading zeroes (`1.2.03`) that default parsing
    rejects here and there alike. The `v` prefix in particular is *rejected, not normalized*:
    a reference PackProof rewrote to understand is a reference that no longer names what the
    reviewer read, and this model is the evidence an integrity report is written against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(
        description=(
            "The package name without its scope: 'node' for '@types/node'. Lowercase, "
            "URL-safe, never one of npm's reserved names, and — when the reference is "
            "unscoped — never beginning with '.' or '_'."
        )
    )
    version: str = Field(
        description=(
            "One exact semver release: prerelease and build metadata included, with every "
            "core component within node-semver's numeric ceiling."
        )
    )
    scope: str | None = Field(
        default=None,
        description="The namespace without its '@', as in 'types' for '@types/node'.",
    )

    @field_validator("name")
    @classmethod
    def _name_is_a_valid_package_name(cls, value: str) -> str:
        if not value:
            raise ValueError("the package name is empty")
        if len(value) > MAX_NAME_LENGTH:
            # A length comparison first, so a pathological name is refused without being
            # scanned by the charset pattern at all. This cannot refuse anything the
            # fullname rule would have allowed: a scope only ever *adds* characters, so a
            # name longer than the whole-name ceiling can never produce a legal fullname.
            raise ValueError(
                f"a package name may be at most {MAX_NAME_LENGTH} characters before its "
                f"scope, and this one is {len(value)} (it begins {ascii(value[:40])})"
            )
        if value in _NON_NAMES:
            raise ValueError(
                f"{ascii(value)} names a path position, not a package; it is refused "
                "whatever scope it is written under"
            )
        if value.lower() in RESERVED_NAMES:
            raise ValueError(f"{ascii(value)} is a name npm reserves for an installed tree")
        if value != value.lower():
            raise ValueError("npm package names are lowercase")
        if _NAME_CHARACTERS.fullmatch(value) is None:
            raise ValueError("names may contain only lowercase letters, digits, '.', '_' and '-'")
        return value

    @field_validator("scope")
    @classmethod
    def _scope_is_a_valid_namespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value:
            raise ValueError("an '@' with an empty namespace is not a scoped package")
        if len(value) > MAX_NAME_LENGTH:
            # Same reasoning as the name field: `@` and `/` are counted by the fullname
            # ceiling, so a namespace longer than that ceiling can never be part of a
            # legal reference, and refusing it here saves the pattern a long scan.
            raise ValueError(
                f"a namespace may be at most {MAX_NAME_LENGTH} characters, and this one is "
                f"{len(value)} (it begins {ascii(value[:40])})"
            )
        if value != value.lower():
            raise ValueError("npm scopes are lowercase")
        if _SCOPE_NAMESPACE.fullmatch(value) is None:
            raise ValueError(
                "a namespace must begin with a lowercase letter or digit and may then "
                "contain only lowercase letters, digits, '.', '_' and '-'"
            )
        return value

    @field_validator("version")
    @classmethod
    def _version_is_one_exact_release(cls, value: str) -> str:
        if len(value) > MAX_VERSION_LENGTH:
            raise ValueError(
                f"a version may be at most {MAX_VERSION_LENGTH} characters, and "
                f"{ascii(value[:32])}... is {len(value)}"
            )
        if _EXACT_SEMVER.fullmatch(value) is None:
            raise ValueError(f"{ascii(value)} is not an exact MAJOR.MINOR.PATCH version")
        # Grammar first, magnitude second. `fullmatch` has already guaranteed that what
        # precedes the first '-' or '+' is exactly three dot-separated ASCII-digit
        # components, so `int()` below cannot fail and the length bound above already
        # capped the work. '+' is split before '-' because a prerelease identifier may not
        # contain a '+' while a build identifier may contain a '-', so the first '+' is
        # always the start of the build tail.
        core = value.split("+", 1)[0].split("-", 1)[0]
        if any(int(part) > MAX_SEMVER_COMPONENT for part in core.split(".")):
            raise ValueError(
                f"{ascii(value)} has a numeric component above {MAX_SEMVER_COMPONENT}, "
                "which is larger than node-semver parses"
            )
        return value

    @model_validator(mode="after")
    def _name_and_scope_form_one_npm_identity(self) -> "PackageRef":
        """The rules that are only decidable once both halves are in hand.

        The leading-character rule is scope-dependent, so it cannot live in the `name`
        validator: that runs before `scope` has been validated and a name like `.hidden` is
        legal in one reference and invalid in another. `.` and `..` are refused there, in
        the field, because they are refused everywhere.
        """
        if self.scope is None and _UNSCOPED_NAME_START.fullmatch(self.name) is None:
            raise ValueError(
                "an unscoped package name may not begin with '.' or '_'; npm allows that "
                "leading character only inside a scope"
            )
        if len(self.fullname) > MAX_NAME_LENGTH:
            raise ValueError(
                f"a package name may be at most {MAX_NAME_LENGTH} characters, and this one "
                f"is {len(self.fullname)} (it begins {ascii(self.fullname[:40])})"
            )
        return self

    @property
    def fullname(self) -> str:
        """The name the registry knows this package by, scope spelled as npm spells it."""
        return f"@{self.scope}/{self.name}" if self.scope is not None else self.name


def _metadata_text(value: str, *, label: str) -> str:
    """Return registry text unchanged, or refuse it for the two ways it cannot be shown.

    Nothing is trimmed, normalized or censored here: the value a reviewer reads in a report
    is the byte sequence the registry sent, which is the point of keeping it. What is
    refused is text that cannot survive being read — longer than any real field by orders
    of magnitude, or carrying a character whose job is to change what the *line* does or to
    make visible text mean something other than what it looks like.

    `label` names the field in the message rather than the offending value, because the
    value is the untrusted part of this transaction; a fragment of it is quoted only when
    the message has to say how long or what began the field.
    """
    if len(value) > MAX_METADATA_TEXT_LENGTH:
        raise ValueError(
            f"{label} is {len(value)} characters, above the {MAX_METADATA_TEXT_LENGTH} "
            f"PackProof accepts from a registry response (it begins {ascii(value[:40])})"
        )
    for character in value:
        category = unicodedata.category(character)
        if category in _UNSAFE_TEXT_CATEGORIES:
            raise ValueError(
                f"{label} contains {ascii(character)}, a Unicode {category} character that "
                "would change how a report line is displayed rather than what it says"
            )
    return value


def _download_target(value: str, *, label: str) -> str:
    """Validate a URL PackProof records but does not contact, and return it unchanged.

    HTTPS, a host, no credentials, no fragment, and no whitespace or non-ASCII anywhere.
    Each rule refuses a spelling that would mean something different to whatever downloads
    this later: `user:pass@` is a credential nobody configured, a fragment is a locator an
    HTTP client silently drops, and a space or an unencoded non-ASCII character is a URL
    that has to be guessed at before it can be encoded — and a download target whose bytes
    are guessed at is not the artifact this metadata describes.

    The path is deliberately not examined. A traversal in a download URL is Stage 2.3's
    problem, and it is a problem about the *local* filename it chooses, not about text in
    this field; bounding it here would suggest the check happens on the way out.
    """
    _metadata_text(value, label=label)
    if not value.isascii():
        raise ValueError(f"{label} must be ASCII to name a download target without guessing")
    if any(character.isspace() for character in value):
        raise ValueError(f"{label} contains whitespace, so it is not one URL")

    try:
        parts = urlsplit(value)
    except ValueError:
        raise ValueError(f"{label} could not be parsed as a URL at all") from None

    if parts.scheme != "https":
        raise ValueError(
            f"{label} uses {ascii(parts.scheme)}; PackProof records only HTTPS download targets"
        )
    if not parts.netloc or parts.hostname is None:
        raise ValueError(f"{label} has no host to download from")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"{label} embeds credentials that no PackProof configuration put there")
    if parts.fragment:
        raise ValueError(f"{label} carries a fragment, which an HTTP client drops before fetching")
    return value


class PackageRepository(BaseModel):
    """Where a published package says its source lives — as the registry said it.

    Recorded, never contacted. Nothing in Stage 2.2 resolves a repository, and the URL is
    not trusted because it arrived in a registry response: a package's metadata is written
    by whoever published it, so this field is evidence about a *claim*, which is exactly
    what a later package-to-source comparison has to check rather than assume.

    `url` keeps the registry's own spelling. npm documents `repository` as either an object
    or a shorthand string, and the string may be `"user/repo"`, `"github:user/repo"` or a
    bare URL. The client stores a shorthand exactly as it was written instead of expanding
    it into a `https://github.com/…` address: inventing the host would be PackProof writing
    metadata the registry never sent.

    `kind` is npm's `repository.type`. It is renamed rather than shadowing the builtin, and
    the two measured spellings in the sampled packages are `git` and absent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    url: str = Field(
        description="The source location in the registry's own words, bounded and reportable."
    )
    kind: str | None = Field(default=None, description="npm's `repository.type`, usually 'git'.")
    directory: str | None = Field(
        default=None,
        description="npm's `repository.directory`, naming a package inside a larger repository.",
    )

    @field_validator("url", "kind", "directory")
    @classmethod
    def _is_reportable_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _metadata_text(value, label=f"repository.{info.field_name}")


class PackageDistribution(BaseModel):
    """What the registry says one published artifact is: where it is and how to check it.

    The three fields are the provenance seed for every later stage — Stage 2.3 downloads
    `tarball_url`, Stage 2.4 recomputes the digests and compares them against these two.
    Neither digest proves the artifact is safe; together they make "the bytes I read are the
    bytes this metadata describes" a question with a decidable answer, which is the most
    integrity metadata is ever able to do.

    `integrity` and `shasum` are optional because the registry genuinely omits them, not
    because a missing value is inconvenient: in the 1,288 measured version entries
    `integrity` was absent once, in a 2014-era release that predates SRI in npm. A field
    that is absent stays absent, and a report says it was absent.

    `shasum` is required to be lowercase because that is what npm writes, and PackProof
    compares this string against a digest it computes itself. A registry that said `ABC…`
    would produce a mismatch against `abc…` and report an integrity failure over a case
    difference; refusing the uppercase spelling makes that impossible instead of handling
    it, which is the same rule `PackageRef` applies to a `v` prefix — reject, do not
    normalize.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tarball_url: str = Field(
        description="HTTPS location the registry offers the packed artifact from, recorded not fetched."
    )
    integrity: str | None = Field(
        default=None,
        description="One SRI hash as npm spells it, e.g. 'sha512-<base64>'.",
    )
    shasum: str | None = Field(
        default=None,
        description="npm's legacy 40-hex lowercase SHA-1 of the packed artifact.",
    )

    @field_validator("tarball_url")
    @classmethod
    def _is_a_https_download_target(cls, value: str) -> str:
        return _download_target(value, label="dist.tarball")

    @field_validator("integrity")
    @classmethod
    def _is_a_complete_sri_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _metadata_text(value, label="dist.integrity")
        algorithm, separator, digest = value.partition("-")
        if not separator or algorithm not in _SRI_DIGEST_LENGTHS:
            raise ValueError(
                f"dist.integrity {ascii(value[:40])} is not 'algorithm-base64' over one of "
                f"{', '.join(sorted(_SRI_DIGEST_LENGTHS))}"
            )
        expected = _SRI_DIGEST_LENGTHS[algorithm]
        try:
            decoded = base64.b64decode(digest, validate=True)
        except ValueError:
            raise ValueError(
                f"dist.integrity {ascii(value[:40])} has a base64 part that does not decode"
            ) from None
        # Re-encode before comparing length: `validate=True` accepts strings that decode but
        # are not the canonical spelling of those bytes, and a digest that does not survive
        # its own round trip cannot be compared byte-for-byte against one computed from real
        # bytes — the comparison a later stage performs is on the text, not on the decoding.
        canonical = base64.b64encode(decoded).decode("ascii")
        if canonical != digest:
            raise ValueError(
                f"dist.integrity {ascii(value[:40])} decodes, but {ascii(canonical)} is the "
                "canonical base64 for those bytes; the spelling it arrived in is not"
            )
        if len(decoded) != expected:
            raise ValueError(
                f"dist.integrity {ascii(value[:40])} declares a {algorithm} hash but decodes "
                f"to {len(decoded)} bytes, and {algorithm} is {expected}"
            )
        return value

    @field_validator("shasum")
    @classmethod
    def _is_a_lowercase_sha1(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _metadata_text(value, label="dist.shasum")
        if _SHA1_HEX.fullmatch(value) is None:
            raise ValueError(
                f"dist.shasum {ascii(value[:48])} is not 40 lowercase hex digits as npm writes it"
            )
        return value


class PackageVersionMetadata(BaseModel):
    """The registry's validated answer about one `PackageRef`.

    `reference` is carried rather than a name string repeated, because the two halves of
    this transaction are "what PackProof asked" and "what the registry said", and a model
    that stores only the answer cannot express that they disagreed. The client that builds
    this checks the agreement before constructing it, so an instance of this class always
    describes the package the caller named.

    Only the fields a report needs are modeled. `readme` is the largest field in a real
    packument by orders of magnitude and is deliberately absent, along with `maintainers`,
    `users`, `_rev` and the rest of a registry document — carrying them would put
    whoever-published-it prose into PackProof output without anything deciding to read it.

    `description` and `repository` are optional because they are optional in the registry:
    39 of the 1,288 measured version entries carried no per-version `repository`, and an
    absent one is reported as absent rather than filled in from the top of the document,
    which is a different claim about a different publish.

    `published_at` is the per-version publish time the registry keeps in `time`. It must be
    an instant, not a wall clock: a naive timestamp would be compared against other
    timestamps and against clock readings in whichever zone the process happens to run in,
    and an ordering of releases derived from that is not the ordering the registry meant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    reference: PackageRef = Field(
        description="The validated request this response answers, so name and version travel together."
    )
    distribution: PackageDistribution = Field(
        description="The artifact the registry says this version is. A missing `dist` is a failure, not a gap."
    )
    description: str | None = Field(
        default=None,
        description="The package's own one-line description, verbatim and bounded.",
    )
    repository: PackageRepository | None = Field(
        default=None,
        description="Where the publisher claims the source is, recorded without being resolved.",
    )
    published_at: datetime | None = Field(
        default=None,
        description="When the registry says this version was published, as a timezone-aware instant.",
    )

    @field_validator("description")
    @classmethod
    def _description_is_reportable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _metadata_text(value, label="description")

    @field_validator("published_at", mode="before")
    @classmethod
    def _publication_is_an_instant(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            if len(value) > 64:
                raise ValueError(
                    f"a published timestamp of {len(value)} characters is not a timestamp"
                )
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(
                    f"published_at {ascii(value[:40])} is not an ISO 8601 timestamp"
                ) from None
        elif isinstance(value, datetime):
            parsed = value
        else:
            raise ValueError(
                f"published_at must be text or absent, not {ascii(type(value).__name__)}"
            )
        if parsed.tzinfo is None:
            raise ValueError(
                f"published_at {ascii(str(parsed)[:40])} names a wall clock, not an instant"
            )
        return parsed
