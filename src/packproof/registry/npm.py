"""npm-specific parsing of a package reference.

Nothing in this module knows about HTTP, the filesystem or a registry response. Its whole
job is to turn the string a person types — `express@4.21.2`, `@types/node@22.10.2` — into
a validated `PackageRef`, which is why it sits beside the fetcher that will use it in
Stage 2.2 instead of inside the registry-agnostic engine: the `@scope/name` shape and the
dist-tag vocabulary belong to npm, and PyPI will bring its own parser that produces the
same model.

Splitting on `@` is the part that needs care, because a scoped name begins with the same
character that separates a name from its version. Splitting `@types/node@22.10.2` on the
first `@` yields an empty name and `types/node@22.10.2` — wrong, and still carrying a
slash, which is precisely the string a later URL builder would have to re-escape to stay
safe. So the leading `@` is consumed as a scope marker first, the scope runs to the first
`/`, and only what is left is split on `@`.

Grammar lives on `PackageRef`, not here. This module decides *structure* — where the scope
ends and where the version begins — and reports whatever the model refuses, with a hint
naming the npm idiom that produced the rejection. Restating the semver and name patterns
here would give the two layers different ideas of what a valid reference is, which is how
a parser ends up accepting things the engine later turns into tracebacks.

Every rejection is the same exception type. A command line that enumerates the ways a
string can be wrong will eventually miss one and print a traceback over an argument that
came from outside, so `parse_npm_reference` converts even `pydantic.ValidationError` into
`InvalidPackageReferenceError`, and quotes the reference back — "what did it actually
read" is the first question a mistyped argument raises, and answers like `''` or a string
with a space in it are only visible when the input is shown.

Every rejection is also ASCII. A message is written to a terminal whose encoding belongs to
the operator, not to PackProof, and a refusal that cannot be encoded dies as a
`UnicodeEncodeError` before it ever says what was wrong — an error about bad input replaced
by an error about the error. So nothing is interpolated with `repr` (which escapes control
characters but passes printable non-ASCII through untouched, and so would echo an
Arabic-Indic digit verbatim) but with `ascii`, and no message literal contains a character
above `\\x7f`.

Which inputs are quoted back is a wider set than the reference alone. A class name reaches
the "must be text" message, and a field name reaches `_explain` — and a field name is not
necessarily one PackProof declared, because pydantic reports the offending key for an
unexpected field, and `package.json` and registry JSON are free to call a key anything. Both
are rendered through `_escape`, so the guarantee belongs to the message-building layer
rather than to a list of names the parser happens to recognise today.
"""

from pydantic import ValidationError

from packproof.schemas.models import MAX_VERSION_LENGTH, PackageRef


class InvalidPackageReferenceError(ValueError):
    """A package reference was supplied but could not be honored.

    Separate from `ConfigurationError`, which answers "what did the operator set"; this
    answers "what did someone hand the CLI", and the two call for different fixes. It
    subclasses `ValueError` so a caller that only cares that the input was bad can catch
    the same clause it already uses for bad values generally.
    """


#: Spellings that make a version a *set* of releases rather than one release.
_RANGE_MARKERS = ("^", "~", ">", "<", "=", "*", "||")

#: How much of an refused input a message may repeat.
_MAX_ECHOED_INPUT = 120


def _show(value: object) -> str:
    """`ascii` of untrusted input, shortened.

    Three separate hazards, handled by one expression. `ascii` escapes control characters,
    so an ANSI sequence smuggled inside a package reference reaches the terminal as the six
    characters `\\x1b` instead of as a live instruction that can rewrite the line the user is
    reading. It also escapes printable non-ASCII, which `repr` does not: `repr('١')` gives
    back the Arabic-Indic digit itself, and a message carrying it cannot be encoded for an
    ASCII terminal. The length bound then keeps a two-megabyte argument — a pasted file, a
    shell quoting mistake, a deliberate flood — out of the message, because the same text is
    printed to a terminal, captured in a CI log and copied into a report.
    """
    rendered = ascii(value)
    if len(rendered) <= _MAX_ECHOED_INPUT:
        return rendered
    return f"{rendered[:_MAX_ECHOED_INPUT]}... ({len(rendered)} characters)"


def _escape(value: object) -> str:
    """`value` as text guaranteed ASCII, without the quotes `ascii()` would add.

    `_show` quotes, because a reader comparing a refusal to what they typed should see the
    quotation marks: `'express@1.0.0 '` exposes the trailing space that `'express@1.0.0'`
    hid. A class name or a field name is different — it is inserted *into* a sentence, and
    quotes there read as part of the name.

    `backslashreplace` does the escaping and does it for everything that cannot be encoded,
    including the lone surrogates that `str.encode("ascii")` would otherwise raise on, so a
    value that never entered this module's own validation still cannot reach a terminal as
    an instruction or as an unprintable byte. ASCII words pass through untouched, which is
    why ordinary messages keep their shape.
    """
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def parse_npm_reference(text: str) -> PackageRef:
    """Parse an exact npm package reference into a validated `PackageRef`.

    Accepted shapes are `NAME@VERSION` and `@SCOPE/NAME@VERSION` with an exact
    `MAJOR.MINOR.PATCH`, prerelease included. Missing versions, ranges, dist-tags, URLs,
    traversal attempts and malformed scoped names are refused rather than repaired: a
    reference that has to be guessed at is not evidence about any particular artifact.

    Args:
        text: The reference exactly as it was typed. Never taken from a package's own
            metadata — a published `package.json` describing itself is untrusted input.

    Returns:
        The parsed reference, immutable and already validated field by field.

    Raises:
        InvalidPackageReferenceError: For every unusable *value*. pydantic's
            `ValidationError` and the `AttributeError` a non-string would raise on
            `.strip()` are both converted, so callers have one failure to handle.

            The guarantee is about values, not about objects that imitate them: an argument
            whose own `str` methods raise propagates unchanged. Catching that would mean
            catching `BaseException` inside a parser, which is how a programming error in
            PackProof becomes a message about the user's package reference. No external
            channel can supply such an object — a command-line argument, a Typer parameter
            and a JSON string are all exact `str` — so the boundary being described here is
            the one this function can actually be asked about.
    """
    if not isinstance(text, str):
        raise InvalidPackageReferenceError(
            f"a package reference must be text, not {_escape(type(text).__name__)} ({_show(text)})"
        )
    if not text:
        raise InvalidPackageReferenceError(
            "a package reference is empty; expected NAME@VERSION, as in express@4.21.2"
        )
    if text != text.strip():
        raise InvalidPackageReferenceError(
            f"package reference ({_show(text)}) has leading or trailing whitespace; "
            "npm names and versions contain none"
        )
    if ":" in text:
        raise InvalidPackageReferenceError(
            f"package reference ({_show(text)}) contains ':', and no npm package reference "
            "does; a URL, git endpoint or 'npm:' alias names something npm resolves at "
            "request time, while PackProof needs the exact NAME@VERSION it resolves to"
        )

    scope, remainder = _split_scope(text)
    name, version = _split_version(remainder, text)

    try:
        return PackageRef(name=name, version=version, scope=scope)
    except ValidationError as exc:
        raise InvalidPackageReferenceError(_explain(exc, text, version=version)) from exc


def _split_scope(text: str) -> tuple[str | None, str]:
    """Consume a leading `@scope/` and return the namespace without its sigil.

    The scope is delimited by `/` rather than by `@`, which is what makes `@babel/core`
    and `@babel/core@7.26.0` parse the same way up to the version. A reference that opens
    with `@` and never reaches a `/` is refused here, not handed on as a name that happens
    to start with one: the model would reject it too, but by then the message has lost the
    shape the person was aiming at.
    """
    if not text.startswith("@"):
        return None, text

    scope, separator, remainder = text[1:].partition("/")
    if not separator:
        raise InvalidPackageReferenceError(
            f"package reference ({_show(text)}) begins with '@' but names no namespace; a "
            "scoped package is '@NAMESPACE/NAME@VERSION', as in @types/node@22.10.2"
        )
    if not remainder:
        raise InvalidPackageReferenceError(
            f"package reference ({_show(text)}) has nothing after the scope's '/'"
        )
    return scope, remainder


def _split_version(remainder: str, text: str) -> tuple[str, str]:
    """Separate the name from the version on the first remaining `@`.

    `text` is the whole original reference and is used only to quote it back: an error
    that names `types/node` when the person wrote `@types/node` sends them looking at the
    wrong string.

    A second `@` is rejected on its own rather than left to the version grammar. npm
    aliases (`lodash@npm:express@4.21.2`) and accidental repeat-pastes both arrive this
    way, and "only a scoped name's leading '@' is expected" is a fact about npm that the
    semver pattern cannot state.
    """
    name, separator, version = remainder.partition("@")
    if not separator:
        raise InvalidPackageReferenceError(
            f"package reference ({_show(text)}) is missing a version; PackProof analyzes one "
            "exact revision, so 'express' is refused where 'express@4.21.2' is accepted"
        )
    if "@" in version:
        raise InvalidPackageReferenceError(
            f"package reference ({_show(text)}) contains more than one '@'; only a scoped "
            "name's leading '@' is expected, and aliases like 'x@npm:y@1.0.0' are not resolved"
        )
    return name, version


def _version_hint(version: str) -> str:
    """Name the npm idiom a rejected version looks like, when it looks like one.

    This is advice, never a decision: `PackageRef` has already refused the value, and
    re-deciding it here would put a second grammar in the codebase. The value of the hint
    is that a range and a dist-tag and a leading-zero version all fail the same pattern
    while calling for three different corrections.

    The text is ASCII for the reason `parse_npm_reference` states: a hint nobody can print
    is not advice.
    """
    if not version:
        return "nothing follows the '@'"
    if len(version) > MAX_VERSION_LENGTH:
        # Too long to be any npm idiom, and the refusal that already fired names the
        # length. Guessing "dist-tag" from the absence of a digit here would add a
        # confident wrong answer to a message that is already correct.
        return ""
    if any(marker in version for marker in _RANGE_MARKERS):
        return "a version range describes many releases, and there is no single artifact to verify"
    if not version.isascii():
        return (
            "versions are ASCII; a look-alike Unicode digit is not the digit it resembles "
            "and asks the registry about something other than what was read"
        )
    # Lowercase only — node-semver's strict `v?` never matches `V`, and `"" in "vV"` is True.
    if version.startswith("v"):
        return (
            "a leading 'v' is git-tag spelling; node-semver accepts it and normalizes it "
            "away, and PackProof refuses the prefix rather than rewriting what was typed"
        )
    if not any(digit in "0123456789" for digit in version):
        return (
            "an npm dist-tag names whatever the registry resolves at request time, so a "
            "scan of 'latest' would not reproduce tomorrow"
        )
    if "+" in version:
        return (
            "build metadata after '+' is dot-separated identifiers of letters, digits "
            "and '-', with no empty identifier between them"
        )
    if len(version.split(".")) < 3:
        return "a partial version like '1.2' or '1.x' is a range; give an exact MAJOR.MINOR.PATCH"
    return ""


def _explain(error: ValidationError, text: str, *, version: str) -> str:
    """Turn pydantic's structured errors into one message that names the reference.

    `loc` says which field was refused, which matters because a scoped reference has two
    name-like parts and a person should not have to guess which one the complaint is about.

    Neither the location nor the reason is treated as trusted text. `parse_npm_reference`
    passes only the three fields it declared, so nothing else can appear here *today*, but
    `extra="forbid"` makes the location whatever key an unexpected payload supplied — and a
    published `package.json` may name a key `é` — so a later caller that hands this function
    a model error from registry data would otherwise print that name straight to a terminal.
    Escaping both halves is what keeps `_explain` total rather than correct-by-argument.
    """
    reasons = []
    for item in error.errors():
        field = _escape(item["loc"][0]) if item["loc"] else "package reference"
        reason = item["msg"]
        if field == "version":
            hint = _version_hint(version)
            if hint:
                reason = f"{reason} - {hint}"
        reasons.append(f"{field}: {_escape(reason)}")
    return f"package reference ({_show(text)}) was refused: {'; '.join(reasons)}"
