"""npm reference parsing: the grammar accepted, everything refused, and each refusal's reason.

Three ideas carry the file. A reference is *one exact artifact*, so anything that describes
a set of releases — a range, a partial version, a dist-tag npm resolves at request time —
is refused even though npm itself would accept it. The refusal is the security behaviour: a
name that reaches a request path or a workspace directory unvetted is where traversal
starts, so the interesting assertion is not that garbage fails but that it fails as one
known error type, naming what was typed. And that naming is ASCII, because the message is
written to a terminal whose encoding belongs to whoever ran the tool.

A fourth theme joins it in the model-boundary sections: the accepted set is an exact,
unprefixed *subset* of what node-semver parses — its numeric ceiling borrowed, its `v`-prefix
normalization refused. Fields are text and only text, a numeric version component is bounded
where node-semver bounds it, and a pathological name is refused on its length before anything
scans it.
"""

import codecs
import io

import pytest
from pydantic import ValidationError

from packproof.registry import InvalidPackageReferenceError, parse_npm_reference
from packproof.registry.npm import _escape, _explain, _version_hint
from packproof.schemas import PackageRef
from packproof.schemas.models import (
    MAX_NAME_LENGTH,
    MAX_SEMVER_COMPONENT,
    MAX_VERSION_LENGTH,
    RESERVED_NAMES,
)

# The four spellings Milestone 2 promises to support, in one place so a later change that
# quietly stops accepting one of them fails here instead of in front of a user.
BRIEF_EXAMPLES = (
    "express@4.21.2",
    "lodash@4.17.21",
    "@types/node@22.10.2",
    "@babel/core@7.26.0",
)

#: A class whose *name* is not ASCII. Python identifiers may be non-ASCII, and the parser
#: quotes `type(text).__name__` when handed something that is not text, so this is the
#: spelling of a message PackProof builds from a value it never validated.
NON_ASCII_NAMED_CLASS = type("\u00e9vil", (object,), {})

#: The opening clause of the hint that belongs to a lowercase `v` and to nothing else: the
#: one prefix node-semver's strict grammar accepts. Named once so the tests below can say
#: both that it fires and that it does not fire, without restating the whole sentence.
_V_PREFIX_HINT = "a leading 'v' is git-tag spelling"

# ----------------------------------------------------------------------------- accepted


@pytest.mark.parametrize("text", BRIEF_EXAMPLES)
def test_the_references_the_milestone_promises_parse(text: str) -> None:
    assert isinstance(parse_npm_reference(text), PackageRef)


@pytest.mark.parametrize(
    ("text", "name", "version"),
    [
        ("express@4.21.2", "express", "4.21.2"),
        ("lodash@4.17.21", "lodash", "4.17.21"),
        ("left-pad@1.3.0", "left-pad", "1.3.0"),
        ("my.pkg_2@1.2.3", "my.pkg_2", "1.2.3"),
        ("7zip@1.0.0", "7zip", "1.0.0"),
        ("a@1.0.0", "a", "1.0.0"),
        ("underscore_name@0.0.1", "underscore_name", "0.0.1"),
    ],
)
def test_unscoped_references_carry_no_scope(text: str, name: str, version: str) -> None:
    ref = parse_npm_reference(text)

    assert ref.name == name
    assert ref.version == version
    assert ref.scope is None
    assert ref.fullname == name


@pytest.mark.parametrize(
    ("text", "scope", "name", "version"),
    [
        ("@types/node@22.10.2", "types", "node", "22.10.2"),
        ("@babel/core@7.26.0", "babel", "core", "7.26.0"),
        ("@angular/core@19.0.0", "angular", "core", "19.0.0"),
        ("@shopify/polaris@12.0.0", "shopify", "polaris", "12.0.0"),
        ("@foo.bar/baz-qux@1.0.0-rc.1", "foo.bar", "baz-qux", "1.0.0-rc.1"),
    ],
)
def test_scoped_references_keep_the_namespace_separate_from_the_name(
    text: str, scope: str, name: str, version: str
) -> None:
    """The leading '@' is a scope marker, not the name/version separator.

    A naive first-'@' split would report the name as '' and the version as
    `types/node@22.10.2` — a value still carrying a slash, which is exactly the string a
    later URL builder would have to re-escape to stay safe.
    """
    ref = parse_npm_reference(text)

    assert ref.scope == scope
    assert ref.name == name
    assert ref.version == version
    assert ref.fullname == f"@{scope}/{name}"


@pytest.mark.parametrize(
    "text",
    [
        "@scope/.hidden@1.0.0",
        "@scope/_private@1.0.0",
        "@types/.sub@2.0.0",
        "@babel/_helper-core@7.26.0-rc.1",
    ],
)
def test_a_scoped_name_may_begin_with_a_dot_or_an_underscore(text: str) -> None:
    """npm's documented rule, quoted in full because the two halves differ only here:

    "The names of scoped packages can begin with a dot or an underscore. This is not
    permitted without a scope." Refusing `@scope/.hidden` would reject a reference npm
    publishes, so the leading character is decided by whether a scope is present rather
    than by the shape of the name.
    """
    ref = parse_npm_reference(text)

    assert ref.scope is not None
    assert ref.name[0] in (".", "_")


@pytest.mark.parametrize("text", [".hidden@1.0.0", "_private@1.0.0", ".cache@1.0.0", "_x_y@0.0.1"])
def test_the_same_leading_character_is_refused_without_a_scope(text: str) -> None:
    """The scoped allowance moved one rule in one position; unscoped validation did not."""
    with pytest.raises(InvalidPackageReferenceError, match="unscoped"):
        parse_npm_reference(text)


@pytest.mark.parametrize(
    "text",
    [
        ".@1.0.0",
        "..@1.0.0",
        "@scope/.@1.0.0",
        "@scope/..@1.0.0",
        "@scope/../x@1.0.0",
        "@./x@1.0.0",
        "@../x@1.0.0",
    ],
)
def test_here_and_up_are_refused_in_every_position(text: str) -> None:
    """The allowance is for a leading dot, not for the two spellings a path join honours.

    This is the whole reason the scoped rule is safe to widen: `.` and `..` are refused by
    name, in the field validator, whatever scope surrounds them — so a leading dot can only
    ever reach a consumer as the first character of a longer segment.
    """
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(text)


def test_a_name_that_is_a_path_position_is_named_as_one() -> None:
    with pytest.raises(InvalidPackageReferenceError, match="path position"):
        parse_npm_reference("@scope/..@1.0.0")


def test_an_underscore_is_allowed_inside_a_namespace() -> None:
    """`_` after the first character of a namespace is legal, and Finding 4 exists because
    the message used to leave it out while the grammar admitted it."""
    assert parse_npm_reference("@a_b/x@1.0.0").scope == "a_b"


def test_the_namespace_message_lists_every_character_the_grammar_admits() -> None:
    """A message that under-states its own pattern sends people deleting good characters."""
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference("@a!b/x@1.0.0")

    message = str(excinfo.value)
    for character in ("'.'", "'_'", "'-'"):
        assert character in message, f"{character} missing from {message}"


@pytest.mark.parametrize(
    "version", ["0.0.0", "1.0.0-rc.1", "2.0.0-alpha.0", "1.2.3-beta.11", "1.0.0-0", "1.0.0-0a"]
)
def test_prerelease_versions_are_exact_and_accepted(version: str) -> None:
    """A prerelease is a published artifact with its own bytes, so it is pinnable."""
    assert parse_npm_reference(f"pkg@{version}").version == version


def test_a_prerelease_may_reach_uppercase_where_a_name_may_not() -> None:
    """Deliberate asymmetry, not an oversight.

    npm requires lowercase names because two names differing only in case would collide on
    a case-insensitive registry. Semver allows ASCII letters in prerelease identifiers, and
    `1.0.0-RC.1` and `1.0.0-rc.1` are distinct prereleases rather than aliases.
    """
    assert parse_npm_reference("pkg@1.0.0-RC.1").version == "1.0.0-RC.1"


# node-semver, the parser the npm registry runs, accepts `MAJOR.MINOR.PATCH` with an
# optional `-prerelease` and an optional `+build` tail, so a build identifier is part of the
# grammar of an exact version rather than a forgivable mistake.
@pytest.mark.parametrize(
    "version",
    [
        "1.2.3+build.5",
        "1.0.0+20130313144700",
        "4.17.21+sha.a1b2c3d",
        "1.0.0+001",
        "1.0.0+exp.sha.5114f85.1",
        "1.0.0+BUILD.MIXED0-9",
        "1.0.0-rc.1+build.5",
        "2.0.0-alpha.1+0.6.5.test-1",
    ],
)
def test_build_metadata_is_part_of_an_exact_version(version: str) -> None:
    """`1.0.0+001` is legal while `1.0.0-01` is not, and that asymmetry is semver's.

    Build identifiers are `[0-9A-Za-z-]+` with no leading-zero restriction; numeric
    *prerelease* identifiers keep it. The earlier refusal of `+` confused "this character
    needs careful encoding in a URL" with "this version does not exist".
    """
    assert parse_npm_reference(f"pkg@{version}").version == version


@pytest.mark.parametrize(
    "version",
    [
        "1.2.3+",
        "1.2.3+build..5",
        "1.2.3++.build",
        "1.2.3+und_score",
        "1.2.3+bu?ild",
        "1.2.3+build.5+x",
        "1.2.3+build%205",
        "1.2.3+١",
    ],
)
def test_malformed_build_metadata_is_still_not_a_version(version: str) -> None:
    """Accepting the tail is not accepting anything after the `+`: an empty identifier, a
    second `+` and a character outside `[0-9A-Za-z-]` are all still refused, because a
    version is used as a path segment and a URL fragment."""
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(f"pkg@{version}")


def test_a_malformed_build_tail_says_what_a_build_tail_looks_like() -> None:
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference("pkg@1.2.3+build..5")

    assert "dot-separated identifiers" in str(excinfo.value)


def test_two_build_metadatas_are_one_version_to_semver_and_two_here() -> None:
    """The hazard that accepting `+` opens, pinned by a test rather than by a comment.

    Semver precedence ignores build metadata, so node-semver calls `1.0.0+a` and `1.0.0+b`
    the same version. They are two published artifacts with two byte streams, and an
    integrity report that collapsed them would describe neither. `PackageRef` compares the
    string it was given, so grouping, sorting and the scan-id derivation downstream must key
    on that string and never on a semver comparison.
    """
    first = parse_npm_reference("pkg@1.0.0+build.1")
    second = parse_npm_reference("pkg@1.0.0+build.2")
    again = parse_npm_reference("pkg@1.0.0+build.1")

    assert first != second
    assert len({first, second}) == 2
    assert first == again
    assert len({first, again}) == 1
    assert first.version == "1.0.0+build.1"


def test_a_version_with_build_metadata_is_not_equal_to_the_bare_version() -> None:
    """`1.0.0` and `1.0.0+build.1` are also two references here, whatever precedence says."""
    assert parse_npm_reference("pkg@1.0.0") != parse_npm_reference("pkg@1.0.0+build.1")


def test_a_numeric_component_above_node_semvers_ceiling_is_refused() -> None:
    """A restriction taken from node-semver, not a claim that the two accepted sets are equal.

    node-semver's `SemVer` constructor throws `TypeError('Invalid major version')`, and the
    equivalent for minor and patch, once a component exceeds JavaScript's
    `Number.MAX_SAFE_INTEGER`. `99999999999999999999.0.0` is well-formed semver that the
    registry's own tooling cannot handle, so it names no artifact anyone could ask for again.
    Upstream stays wider in other directions — a leading `v`, surrounding whitespace, leading
    zeroes under `loose` — and each of those is refused on its own merits elsewhere in this
    file, so this test pins the ceiling only.
    """
    assert MAX_SEMVER_COMPONENT == 2**53 - 1

    at_ceiling = parse_npm_reference(f"pkg@{MAX_SEMVER_COMPONENT}.0.0")
    assert at_ceiling.version == f"{MAX_SEMVER_COMPONENT}.0.0"

    for version in (
        f"{MAX_SEMVER_COMPONENT + 1}.0.0",
        f"1.{MAX_SEMVER_COMPONENT + 1}.0",
        f"1.0.{MAX_SEMVER_COMPONENT + 1}",
        "99999999999999999999.0.0",
    ):
        with pytest.raises(InvalidPackageReferenceError, match="numeric component") as excinfo:
            parse_npm_reference(f"pkg@{version}")
        assert str(excinfo.value).isascii()


def test_every_core_component_may_sit_at_the_ceiling_together() -> None:
    """The bound is per component, so the widest legal version is three of them."""
    version = f"{MAX_SEMVER_COMPONENT}.{MAX_SEMVER_COMPONENT}.{MAX_SEMVER_COMPONENT}"
    assert parse_npm_reference(f"pkg@{version}").version == version


@pytest.mark.parametrize(
    "version",
    [
        f"1.0.0-{MAX_SEMVER_COMPONENT}",
        f"1.0.0-{MAX_SEMVER_COMPONENT + 1}",
        f"1.0.0-{MAX_SEMVER_COMPONENT * 10}",
        f"1.0.0+{MAX_SEMVER_COMPONENT + 1}",
        f"1.0.0-rc.{MAX_SEMVER_COMPONENT + 1}+{MAX_SEMVER_COMPONENT + 1}",
    ],
)
def test_prerelease_and_build_identifiers_are_not_magnitude_bounded(version: str) -> None:
    """Matching upstream means matching its limits and its absence of limits.

    node-semver magnitude-checks major, minor and patch and nothing else; a prerelease
    identifier is turned into a number only when both sides of a comparison are numeric, and
    no constructor check bounds it. Inventing a ceiling here would refuse versions the
    registry publishes, which is the opposite of the compatibility the last two tests asked
    for — so the asymmetry is upstream's, recorded rather than quietly copied.
    """
    assert parse_npm_reference(f"pkg@{version}").version == version


def test_ordinary_versions_and_build_metadata_are_untouched_by_the_ceiling() -> None:
    """The check runs after the grammar, so it can only narrow what was already accepted."""
    for version in ("0.0.0", "1.2.3", "4.21.2", "1.0.0-rc.1", "1.0.0+build.5", "1.2.3+001"):
        assert parse_npm_reference(f"pkg@{version}").version == version


def test_position_decides_which_half_is_the_version() -> None:
    """`v1.2.3` is a legal npm name, and the separator — not the shape — makes it one.

    The left half is the name whatever it looks like, so no heuristic about "which side
    resembles a version" enters the parser.
    """
    assert parse_npm_reference("v1.2.3@1.0.0") == PackageRef(name="v1.2.3", version="1.0.0")
    assert parse_npm_reference("1.2.3@4.5.6") == PackageRef(name="1.2.3", version="4.5.6")


@pytest.mark.parametrize(
    "text",
    [
        "express@4.21.2",
        "@types/node@22.10.2",
        "@babel/core@7.26.0-rc.1",
        "my.pkg_2@1.2.3",
        "v1.2.3@1.0.0",
        "@scope/.hidden@1.0.0",
        "@scope/_private@1.0.0",
        "pkg@1.0.0-rc.1+build.5",
    ],
)
def test_no_accepted_field_can_read_as_a_path(text: str) -> None:
    """The invariant every downstream consumer is written against.

    An unscoped name can never *be* `..` because it must begin with a letter or digit, and a
    scoped name can never *be* `.` or `..` because those are refused by name — so a
    parent-directory component cannot appear in any accepted value regardless of how it is
    later joined to a directory. Build metadata is the one character an accepted version may
    carry that a URL has to encode, and it is not a path character.
    """
    ref = parse_npm_reference(text)
    values = [ref.name, ref.version]
    if ref.scope is not None:
        values.append(ref.scope)

    for value in values:
        assert value not in ("", ".", "..")
        assert "/" not in value
        assert "\\" not in value
        assert "\x00" not in value
        assert value == value.strip()


def test_the_length_bound_counts_the_scope_too() -> None:
    """npm's 214 applies to the whole name, `@scope/` included."""
    assert len(parse_npm_reference(f"{'a' * MAX_NAME_LENGTH}@1.0.0").name) == MAX_NAME_LENGTH

    with pytest.raises(InvalidPackageReferenceError, match="at most"):
        parse_npm_reference(f"{'a' * (MAX_NAME_LENGTH + 1)}@1.0.0")
    with pytest.raises(InvalidPackageReferenceError, match="at most"):
        # '@s/' plus a 213-character name is 216 characters of registry identity.
        parse_npm_reference(f"@s/{'a' * (MAX_NAME_LENGTH - 1)}@1.0.0")


def test_a_name_at_the_ceiling_is_accepted_on_both_routes() -> None:
    """Exactly 214, unscoped and scoped, so the early field check cannot be one too tight.

    A 211-character name inside `@s/` is 214 characters of registry identity: the early check
    on `name` compares against the *whole-name* ceiling, which no valid name can exceed on
    its own because a scope only ever adds characters. Anything the field check refuses is
    therefore something the fullname rule would have refused later and more expensively.
    """
    unscoped = parse_npm_reference(f"{'a' * MAX_NAME_LENGTH}@1.0.0")
    assert len(unscoped.fullname) == MAX_NAME_LENGTH

    scoped = parse_npm_reference(f"@s/{'a' * (MAX_NAME_LENGTH - 3)}@1.0.0")
    assert len(scoped.fullname) == MAX_NAME_LENGTH
    assert scoped.scope == "s"

    assert len(PackageRef(name="a" * MAX_NAME_LENGTH, version="1.0.0").fullname) == MAX_NAME_LENGTH
    assert (
        len(PackageRef(name="a" * (MAX_NAME_LENGTH - 3), version="1.0.0", scope="s").fullname)
        == MAX_NAME_LENGTH
    )


def test_one_character_past_the_ceiling_is_refused_on_both_routes() -> None:
    """The boundary stays where npm put it after the check moved earlier."""
    with pytest.raises(InvalidPackageReferenceError, match="at most"):
        parse_npm_reference(f"{'a' * (MAX_NAME_LENGTH + 1)}@1.0.0")
    with pytest.raises(InvalidPackageReferenceError, match="at most"):
        parse_npm_reference(f"@s/{'a' * (MAX_NAME_LENGTH - 2)}@1.0.0")

    with pytest.raises(ValidationError, match="at most"):
        PackageRef(name="a" * (MAX_NAME_LENGTH + 1), version="1.0.0")
    with pytest.raises(ValidationError, match="at most"):
        PackageRef(name="a" * (MAX_NAME_LENGTH - 2), version="1.0.0", scope="s")


@pytest.mark.parametrize("character", ["a", "\u00e9"])
def test_a_pathological_name_is_refused_on_length_before_anything_scans_it(
    character: str,
) -> None:
    """Two megabytes of name costs one comparison, not a pass of the charset pattern.

    `str.__len__` does not inspect the contents, so the size of the input no longer sets the
    cost of refusing it — and that holds for a non-ASCII name too, which is the case where a
    scan would have to look at every character.

    The assertion is on the sentence PackProof wrote, not on `str(ValidationError)`. pydantic
    truncates an `input_value` it echoes but keeps printable non-ASCII bare in it, so its
    aggregate rendering is bounded and still not encodable; `_explain` reads `errors()` for
    exactly that reason, and the parser route below is what an operator ever sees.
    """
    pathological = character * 2_000_000

    with pytest.raises(ValidationError) as excinfo:
        PackageRef(name=pathological, version="1.0.0")

    reason = excinfo.value.errors()[0]["msg"]
    assert "at most 214" in reason
    assert "2000000" in reason, "the refusal did not say how long the name was"
    assert "a" * 500 not in reason and "\u00e9" * 500 not in reason, "the refusal echoed it"
    reason.encode("ascii")
    assert reason.isascii()

    with pytest.raises(InvalidPackageReferenceError) as refusal:
        parse_npm_reference(f"{pathological}@1.0.0")

    message = str(refusal.value)
    message.encode("ascii")
    assert message.isascii()
    assert "a" * 500 not in message and "\u00e9" * 500 not in message
    assert len(message) < 500, f"the parser message grew with the input: {len(message)}"


def test_a_pathological_namespace_is_refused_on_length_too() -> None:
    """The namespace is the other field a caller can make arbitrarily long, and it is the
    one that becomes a directory level in every cache layout the analyzer will build."""
    with pytest.raises(ValidationError) as excinfo:
        PackageRef(name="x", version="1.0.0", scope="\u00e9" * 2_000_000)

    reason = excinfo.value.errors()[0]["msg"]
    assert "namespace may be at most 214 characters" in reason
    assert "2000000" in reason
    assert "\u00e9" * 500 not in reason
    reason.encode("ascii")
    assert reason.isascii()

    # Built outside the f-string: a backslash inside an f-string replacement field is
    # Python 3.12 syntax, and this project supports 3.11.
    huge_namespace = "\u00e9" * 2_000_000
    with pytest.raises(InvalidPackageReferenceError) as refusal:
        parse_npm_reference(f"@{huge_namespace}/x@1.0.0")

    message = str(refusal.value)
    message.encode("ascii")
    assert message.isascii()
    assert len(message) < 500


def test_the_model_level_fullname_invariant_still_runs() -> None:
    """The early field checks do not replace the complete-name rule.

    A 212-character name is under the name ceiling, and a one-character namespace is too,
    but `@s/` plus 212 is 215 — a number only the pair can produce. That refusal still has
    to come from somewhere, and the somewhere is the model validator.
    """
    with pytest.raises(ValidationError, match="214 characters"):
        PackageRef(name="a" * 212, version="1.0.0", scope="s")


@pytest.mark.parametrize("name", sorted(RESERVED_NAMES))
def test_npm_reserved_names_are_refused(name: str) -> None:
    with pytest.raises(InvalidPackageReferenceError, match="reserves"):
        parse_npm_reference(f"{name}@1.0.0")


def test_reserved_names_are_refused_before_the_lowercase_rule() -> None:
    """`Node_Modules` is the same collision, reached by a different spelling."""
    with pytest.raises(InvalidPackageReferenceError, match="reserves"):
        parse_npm_reference("Node_Modules@1.0.0")


# ----------------------------------------------------------------------------- refused


@pytest.mark.parametrize(
    "text",
    [
        "express",
        "@types/node",
        "",
        "   ",
        "express@",
        "@",
        "/",
        "a@1.0.0@2.0.0",
        "lodash@npm:express@4.21.2",
    ],
)
def test_a_reference_without_exactly_one_name_and_version_is_refused(text: str) -> None:
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(text)


@pytest.mark.parametrize("text", ["express", "@types/node", "lodash"])
def test_a_missing_version_is_named_as_missing(text: str) -> None:
    """The most common mistake deserves the clearest sentence rather than a grammar note."""
    with pytest.raises(InvalidPackageReferenceError, match="missing a version"):
        parse_npm_reference(text)


@pytest.mark.parametrize(
    "version", ["^4.21.2", "~4.21.0", ">=4.0.0", "<5.0.0", "=4.21.2", "*", "1.x", "1.2", "4"]
)
def test_version_ranges_are_refused_because_they_are_not_one_artifact(version: str) -> None:
    with pytest.raises(InvalidPackageReferenceError, match="range"):
        parse_npm_reference(f"express@{version}")


@pytest.mark.parametrize("tag", ["latest", "next", "beta", "alpha", "legacy"])
def test_dist_tags_are_refused_because_they_do_not_reproduce(tag: str) -> None:
    """`express@latest` means a different package every week, so a finding made today
    would describe an artifact nobody could ask about again."""
    with pytest.raises(InvalidPackageReferenceError, match="dist-tag"):
        parse_npm_reference(f"express@{tag}")


@pytest.mark.parametrize(
    "text",
    [
        "https://registry.npmjs.org/express",
        "http://localhost:4873/express",
        "git+ssh://git@github.com/expressjs/express.git",
        "git://github.com/expressjs/express",
        "github:expressjs/express",
        "file:./local-package",
        "npm:express@4.21.2",
    ],
)
def test_urls_and_other_protocols_are_refused(text: str) -> None:
    """A colon is not a character npm references contain, so no name, version or scope
    can carry one — which makes it a reliable early signal rather than a heuristic."""
    with pytest.raises(InvalidPackageReferenceError, match="contains ':'"):
        parse_npm_reference(text)


@pytest.mark.parametrize(
    "text",
    [
        "../../etc/passwd@1.0.0",
        "..@1.0.0",
        ".hidden@1.0.0",
        "_private@1.0.0",
        "express@4.21.2/../../../etc/passwd",
        "express@../bar",
        "@scope/../../x@1.0.0",
        "a/b@1.0.0",
        "express@%2e%2e%2fpasswd",
        "express@1.0.0%00",
        "foo\x00bar@1.0.0",
        "C:\\packages\\express@1.0.0",
    ],
)
def test_path_and_url_traversal_attempts_are_refused(text: str) -> None:
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(text)


@pytest.mark.parametrize(
    "text",
    [
        "@types",
        "@types/",
        "@/node@1.0.0",
        "@types/node/extra@1.0.0",
        "@@types/node@1.0.0",
        "@types/node@@1.0.0",
        "@Types/node@1.0.0",
        "@scope/../name@1.0.0",
        "types/node@22.10.2",
    ],
)
def test_malformed_scoped_references_are_refused(text: str) -> None:
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(text)


@pytest.mark.parametrize("text", ["@types", "@types/"])
def test_a_broken_scope_says_what_a_scope_should_look_like(text: str) -> None:
    """`@types` is a person who meant `@types/node@VERSION`; the message should still be
    about the shape they were reaching for, not only about what was rejected."""
    with pytest.raises(InvalidPackageReferenceError, match="@NAMESPACE/NAME@VERSION|'/'"):
        parse_npm_reference(text)


def test_an_empty_namespace_is_named_as_one() -> None:
    """`@/node` reads as a slash-misplaced typo, so the message says which half is missing."""
    with pytest.raises(InvalidPackageReferenceError, match="empty namespace"):
        parse_npm_reference("@/node@1.0.0")


@pytest.mark.parametrize(
    "text",
    [
        "Express@4.21.2",
        "EXPRESS@4.21.2",
        "foo bar@1.0.0",
        "foo\tbar@1.0.0",
        "foo!bar@1.0.0",
        "foo(1)@1.0.0",
        "foo*bar@1.0.0",
        "foo'bar@1.0.0",
        "foo~bar@1.0.0",
        "foo%41@1.0.0",
        "foo&bar@1.0.0",
        "foo;rm -rf@1.0.0",
        "foo;@1.0.0",
        "Éxpress@1.0.0",
        "ｆｏｏ@1.0.0",
        "fo٠@1.0.0",
        "١foo@1.0.0",
    ],
)
def test_characters_outside_the_npm_alphabet_are_refused(text: str) -> None:
    """An allowlist, so the list of things that must not appear is not a list anyone has
    to keep complete.

    The two Arabic-Indic rows are why the character classes spell `[0-9]`: `[a-z\\d]` would
    accept them, `PackageRef` would hand back a name that looks like the one a reviewer read
    but is not it, and every comparison built on it — request path, cache directory, report
    line — would silently be about a different package.
    """
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(text)


@pytest.mark.parametrize("version", ["1.0.0-١a", "1.0.0-١", "1.0.0-ab١"])
def test_a_prerelease_identifier_may_not_carry_a_unicode_digit(version: str) -> None:
    """`[0-9]*[a-zA-Z-]…` is the one place a prerelease can begin with a digit, so a `\\d`
    there would widen a version that otherwise looks exactly like `1.0.0-1a`."""
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(f"pkg@{version}")


@pytest.mark.parametrize(
    "version",
    [
        "01.2.3",
        "1.02.3",
        "1.2.3.4",
        "1.2.3-",
        "1.2.3-01",
        "v1.2.3",
        "1.2.3 ",
        "1.2.3\n",
        "١.٢.٣",
        "１.２.３",
        "1.2.٣",
    ],
)
def test_versions_that_are_not_one_exact_ascii_release_are_refused(version: str) -> None:
    """The Unicode rows are the reason the grammar spells digits `[0-9]`: Python's `\\d`
    matches them, and a version that looks like the one a reviewer read but is not it would
    silently change which artifact the whole scan is about."""
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(f"express@{version}")


@pytest.mark.parametrize("version", ["1.2.3+build..5", "v1.2.3", "1.2"])
def test_a_refused_version_also_names_the_idiom_it_mistook(version: str) -> None:
    """`PackageRef` owns the grammar and says 'not exact'; this layer's job is to add the
    npm-specific advice that turns it into something a person can act on. The separator is
    ASCII because the whole message has to be."""
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(f"express@{version}")

    assert "not an exact MAJOR.MINOR.PATCH version" in str(excinfo.value)
    assert " - " in str(excinfo.value)


def test_a_lowercase_v_prefix_is_hinted_as_the_spelling_upstream_normalizes() -> None:
    """The git-tag advice belongs to `v1.2.3` because node-semver really does accept it.

    Its strict `FULLPLAIN` carries `v?`, and `format()` rebuilds the string from parsed
    numbers, so `semver.valid('v1.2.3')` returns `'1.2.3'` in both strict and loose mode.
    PackProof refuses the prefix instead of rewriting the reference it was handed, and the
    hint has to name which of the two happened: one is a normalization a person may rely on,
    the other is a spelling they have to drop before the reference means anything.
    """
    assert _version_hint("v1.2.3").startswith(_V_PREFIX_HINT)

    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference("express@v1.2.3")

    message = str(excinfo.value)
    assert "not an exact MAJOR.MINOR.PATCH version" in message
    assert _V_PREFIX_HINT in message
    assert "normalizes it away" in message
    message.encode("ascii")


@pytest.mark.parametrize(
    ("version", "expected_hint"),
    [
        ("V1.2.3", ""),
        ("", "nothing follows the '@'"),
    ],
)
def test_the_lowercase_v_hint_does_not_fire_for_a_capital_or_an_absence(
    version: str, expected_hint: str
) -> None:
    """Two spellings the old membership test caught, for two unrelated reasons.

    `V1.2.3`: node-semver matches it neither way - strict is `v?`, loose is `[v=\\s]*` - so
    `semver.valid('V1.2.3')` is `null`, and git-tag advice would send the reader toward a
    prefix that upstream does not accept either. The refusal was always right; only the
    advice was wrong, and it now stays silent.

    `''`: in Python `'' in 'vV'` is `True`, so `version[:1] in "vV"` was a test an empty
    version passed. The branch two above already returns 'nothing follows the' advice, which
    makes the trap latent rather than live, but a slice that can be empty is still the wrong
    shape for a prefix check and `startswith` says what is meant.
    """
    assert _version_hint(version) == expected_hint

    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(f"express@{version}")

    message = str(excinfo.value)
    assert _V_PREFIX_HINT not in message
    assert "not an exact MAJOR.MINOR.PATCH version" in message
    message.encode("ascii")


@pytest.mark.parametrize("text", [None, 42, 3.5, True, b"express@4.21.2", ["express@4.21.2"], {}])
def test_input_that_is_not_text_is_refused_not_raised(text: object) -> None:
    """`AttributeError` from `.strip()` and `TypeError` from a comparison are the failure
    modes a traceback-avoiding CLI cannot have, whatever type a caller passes."""
    with pytest.raises(InvalidPackageReferenceError):
        parse_npm_reference(text)  # type: ignore[arg-type]


def test_the_guarantee_is_about_values_not_about_objects_that_imitate_them() -> None:
    """The boundary of "one failure to handle", stated where it can be checked.

    Every failure a *value* can produce arrives as `InvalidPackageReferenceError`. An object
    that impersonates a string and raises from its own methods does not, and deliberately so:
    converting that would mean catching `BaseException` inside a parser, which turns a bug in
    PackProof into a message blaming someone else's package reference.

    Normalising with `str(text)` was considered and rejected. The same method table that lets
    `strip()` raise lets it *lie*, and the second class below shows what that costs: a parser
    that normalized first would go on to validate `express@1.0.0` while the operator had
    typed something else, which is the exact substitution an integrity tool exists to prevent.
    The refusal quotes the object that arrived, so the lie is at least visible.
    """

    class RaisesOnStrip(str):
        def strip(self, *args: object) -> str:
            raise RuntimeError("not a problem with the value")

    with pytest.raises(RuntimeError):
        parse_npm_reference(RaisesOnStrip("express@1.0.0"))  # type: ignore[arg-type]

    class LiesAboutStrip(str):
        def strip(self, *args: object) -> str:
            return "express@1.0.0"

    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(LiesAboutStrip("evil@9.9.9"))  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "evil@9.9.9" in message, "the refusal quoted the normalized value, not the one given"
    assert "express@1.0.0" not in message


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "express",
        "Express@4.21.2",
        "express@^4.0.0",
        "express@latest",
        "../../etc/passwd@1.0.0",
        "@types",
        "@/node@1.0.0",
        "express@1.2.3+build..5",
        "@scope/..@1.0.0",
        ".hidden@1.0.0",
        "express@١.٢.٣",
        "git://github.com/expressjs/express",
        "node_modules@1.0.0",
        None,
        42,
    ],
)
def test_every_refusal_travels_as_one_exception_type(text: object) -> None:
    """Exactly one failure to handle.

    `pydantic.ValidationError` is caught and converted rather than allowed to pass through
    as a `ValueError` subclass, so a command line that catches this one type cannot
    accidentally print a traceback for the field-level rejections.
    """
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(text)  # type: ignore[arg-type]

    assert excinfo.type is InvalidPackageReferenceError


@pytest.mark.parametrize("text", ["Express@4.21.2", "express@latest", "express@1.0.0 "])
def test_a_refusal_quotes_the_reference_that_was_typed(text: str) -> None:
    """Half of diagnosing a mistyped argument is seeing what the parser actually received:
    a trailing space and an empty string are otherwise invisible."""
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(text)

    assert repr(text) in str(excinfo.value)


def test_an_absurdly_long_version_is_refused_without_being_echoed_back() -> None:
    """`MAX_VERSION_LENGTH` mirrors node-semver, the registry's own parser, so the cost of
    validating a version is bounded by a constant rather than by whatever arrived.

    The second half matters as much as the refusal: a diagnostic that prints the value it
    rejected would put two megabytes of filler into a terminal, a CI log and any report
    written from it, so the message quotes a prefix.
    """
    boundary = f"1.0.0-{'a' * (MAX_VERSION_LENGTH - 6)}"
    assert len(boundary) == MAX_VERSION_LENGTH
    assert parse_npm_reference(f"pkg@{boundary}").version == boundary

    with pytest.raises(InvalidPackageReferenceError, match="at most") as excinfo:
        parse_npm_reference(f"pkg@{'a' * 2_000_000}")

    message = str(excinfo.value)
    assert "a" * 500 not in message, "the refusal echoed the input"
    assert "characters)" in message, "the refusal did not say how much it left out"
    assert len(message) < 400
    assert "dist-tag" not in message, "a two-megabyte string is not an npm idiom"


@pytest.mark.parametrize(
    "text",
    [
        "express@1.0.0\x1b[2K",
        "exp\x1b[31bress@1.0.0",
        "express@\x071.0.0",
        "express@\x001.0.0",
        "express@1.0.0\r\nsecond line",
    ],
)
def test_a_refusal_cannot_rewrite_the_terminal_it_is_printed_in(text: str) -> None:
    """A reference is an attacker-supplied string and this message is printed to a terminal.

    Every interpolation in the parser goes through `ascii()` or `_escape()`, which turns a
    live escape sequence into the six characters backslash-x-1-b: readable to a reviewer,
    inert to the shell that displays it.
    """
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(text)

    message = str(excinfo.value)
    for control in ("\x1b", "\x00", "\x07", "\r", "\n"):
        assert control not in message, f"{control!r} reached the message unescaped"


# Every route that produces a refusal, so a message that grew a non-ASCII character in some
# branch nobody thought to print cannot pass a test that only checked another branch.
REFUSAL_MESSAGE_CASES = [
    None,
    42,
    3.5,
    True,
    b"express@4.21.2",
    ["express@4.21.2"],
    {},
    "",
    "   ",
    " express@1.0.0",
    "express@1.0.0 ",
    "express",
    "express@",
    "a@1.0.0@2.0.0",
    "https://registry.npmjs.org/express",
    "npm:express@4.21.2",
    "@types",
    "@types/",
    "@/node@1.0.0",
    "@scope/@x@1.0.0",
    "Express@1.0.0",
    "١foo@1.0.0",
    "foo bar@1.0.0",
    "Éxpress@1.0.0",
    "ｆｏｏ@1.0.0",
    "node_modules@1.0.0",
    "Node_Modules@1.0.0",
    "..@1.0.0",
    "@scope/..@1.0.0",
    ".hidden@1.0.0",
    "@a!b/x@1.0.0",
    "pkg@^1.0.0",
    "pkg@latest",
    "pkg@1.2.3+build..5",
    "pkg@v1.2.3",
    "pkg@١.٢.٣",
    "pkg@１.２.３",
    "pkg@1.2",
    "pkg@01.2.3",
    "pkg@1.0.0\x1b[2K",
    f"{'a' * 215}@1.0.0",
    f"pkg@{'a' * 2_000}",
    NON_ASCII_NAMED_CLASS(),
]


@pytest.mark.parametrize("text", REFUSAL_MESSAGE_CASES)
def test_every_refusal_message_encodes_as_ascii(text: object) -> None:
    """Not "contains no control characters" — the bytes actually go through an ASCII codec.

    An ASCII stdout is not hypothetical: `LC_ALL=C`, a bare CI executor and a mis-set
    terminal all produce one. A message that cannot be encoded dies as
    `UnicodeEncodeError` while being printed, which replaces an error about bad input with
    an error about the error and names no input at all.
    """
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(text)  # type: ignore[arg-type]

    message = str(excinfo.value)
    message.encode("ascii")
    assert message.isascii(), message


def test_a_refusal_survives_a_real_ascii_output_stream() -> None:
    """The same claim one level up: the writer a text stream uses, over every message.

    `codecs.getwriter("ascii")` is the encoder a `PYTHONIOENCODING=ascii` stdout writes
    through, so this exercises the failure mode rather than a proxy for it.
    """
    buffer = io.BytesIO()
    writer = codecs.getwriter("ascii")(buffer)

    for text in REFUSAL_MESSAGE_CASES:
        with pytest.raises(InvalidPackageReferenceError) as excinfo:
            parse_npm_reference(text)  # type: ignore[arg-type]
        writer.write(str(excinfo.value))
        writer.write("\n")

    written = buffer.getvalue()
    assert written.startswith(b"package reference") or written.startswith(b"a package")
    assert len(written.splitlines()) == len(REFUSAL_MESSAGE_CASES)


def test_a_unicode_digit_is_echoed_escaped_rather_than_bare() -> None:
    """`repr()` escapes control characters but prints printable non-ASCII as itself, so the
    em-dash fix alone would not have made these messages encodable."""
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference("express@١.٢.٣")

    message = str(excinfo.value)
    assert "\\u0661" in message, "the refused character is no longer visible to a reviewer"
    assert "١" not in message, "a bare non-ASCII character reached the message"


def test_the_models_error_is_ascii_where_pydantics_own_rendering_is_not() -> None:
    """Why `ValidationError` is converted rather than re-raised, now load-bearing.

    pydantic renders an error by appending its own `input_value='١foo'` line, and that repr
    keeps printable non-ASCII bare. A command line that printed the `ValidationError`
    directly would crash on an ASCII terminal even though every sentence PackProof writes is
    clean — so the conversion is not only one exception type, it is the only ASCII-safe
    rendering of the refusal.
    """
    with pytest.raises(ValidationError) as excinfo:
        PackageRef(name="١foo", version="1.0.0")
    with pytest.raises(UnicodeEncodeError):
        str(excinfo.value).encode("ascii")

    with pytest.raises(InvalidPackageReferenceError) as refusal:
        parse_npm_reference("١foo@1.0.0")
    assert str(refusal.value).isascii()


def test_a_non_ascii_class_name_cannot_reach_the_message_raw() -> None:
    """Finding A: the "must be text" branch quotes a name it never validated.

    Python identifiers may be non-ASCII, so `type(text).__name__` is input in the same sense
    the reference is. Reaching it needs an object of such a class — which a command line, a
    Typer parameter and a JSON payload all cannot produce — but the message is the message,
    and a guarantee with an exception in it is not a guarantee.
    """
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference(NON_ASCII_NAMED_CLASS())  # type: ignore[arg-type]

    message = str(excinfo.value)
    message.encode("ascii")
    assert message.isascii(), message
    assert "\u00e9vil" not in message, "the raw class name reached the terminal"
    assert "\\xe9vil" in message, "the class name became invisible to a reviewer"
    assert "must be text, not" in message, "the refusal stopped naming what it was given"


def test_an_unexpected_field_name_cannot_reach_the_message_raw() -> None:
    """Finding B: `extra="forbid"` makes the error location whatever key arrived.

    Only `name`, `version` and `scope` can reach `_explain` from `parse_npm_reference` today,
    and that is a fact about one call site rather than a property of the function. Stage 2.2
    will build a `PackageRef` from a registry response, and a `package.json` is free to call a
    key anything — including a name an ASCII terminal cannot print. The test drives the helper
    directly because the point is the rendering, not the route.
    """
    with pytest.raises(ValidationError) as excinfo:
        PackageRef(**{"\u00e9": 1, "name": "express", "version": "1.0.0"})

    error = excinfo.value
    assert error.errors()[0]["loc"][0] == "\u00e9", "pydantic stopped echoing the offending key"

    explained = _explain(error, "express@1.0.0", version="1.0.0")
    explained.encode("ascii")
    assert explained.isascii(), explained
    assert "\u00e9" not in explained, "the raw key reached the terminal"
    assert "\\xe9" in explained
    assert "Extra inputs are not permitted" in explained, "the refusal lost its reason"


def test_the_escapes_leave_useful_field_identification_alone() -> None:
    """`version:` should still read as a field name, not as `'version'` or `\x76ersion`.

    The escaping exists for characters above ASCII; ASCII words pass through untouched, which
    is what keeps the messages that were already legible legible.
    """
    with pytest.raises(InvalidPackageReferenceError) as excinfo:
        parse_npm_reference("express@^1.0.0")

    message = str(excinfo.value)
    assert "version: Value error" in message
    assert "'version':" not in message, "a field label gained quotes it did not have"
    assert _escape("version") == "version"
    assert _escape(MAX_NAME_LENGTH) == "214"


def test_the_escape_survives_a_lone_surrogate() -> None:
    """`backslashreplace` rather than a plain encode, for the input that cannot be encoded.

    A lone high surrogate is valid inside a Python `str` and undefined in every fixed-width
    encoding, so `\\ud800`.encode("ascii") raises the very error the escaping is here to
    prevent. pydantic has been seen to carry such values in a location tuple.

    The escape is doubled in this prose but single in the assertion below, and the
    difference is load-bearing rather than typographic. A docstring is a compiled constant,
    so a live `\\ud800` escape there puts an unencodable character into the module, and
    pytest's assertion rewriting — which re-compiles and caches the source — dies during
    collection with a `UnicodeEncodeError` about the test file itself. Prose must write the
    surrogate as text; only a runtime value may hold one.
    """
    escaped = _escape("\ud800name")

    assert escaped == "\\ud800name"
    assert escaped.isascii()


# ------------------------------------------------------------------- the model boundary


@pytest.mark.parametrize(
    "fields",
    [
        {"name": "../x", "version": "1.0.0"},
        {"name": "a/b", "version": "1.0.0"},
        {"name": "..", "version": "1.0.0"},
        {"name": "express", "version": "^1.0.0"},
        {"name": "express", "version": "latest"},
        {"name": "express", "version": "1.2"},
        {"name": "express", "version": "١.٢.٣"},
        {"name": "express", "version": "1.0.0", "scope": ".."},
        {"name": "express", "version": "1.0.0", "scope": "."},
        {"name": ".", "version": "1.0.0", "scope": "s"},
        {"name": "..", "version": "1.0.0", "scope": "s"},
        {"name": ".hidden", "version": "1.0.0"},
        {"name": "_private", "version": "1.0.0"},
        {"name": "express", "version": "1.0.0+"},
        {"name": "express", "version": "1.0.0+a..b"},
        {"name": "express", "version": "1.0.0+und_score"},
        {"name": "node_modules", "version": "1.0.0"},
        {"name": 42, "version": "1.0.0"},
        {"name": None, "version": "1.0.0"},
    ],
)
def test_the_model_refuses_unsafe_fields_constructed_directly(fields: object) -> None:
    """Defence in depth for Stage 2.2.

    The metadata client will build a `PackageRef` from a registry response rather than from
    a string this parser vetted, and registry JSON describes a package anyone may publish
    under any spelling. The grammar lives on the fields so that route is bounded by the same
    rules instead of relying on every caller remembering to check.
    """
    with pytest.raises(ValidationError):
        PackageRef(**fields)  # type: ignore[arg-type]


def test_the_scope_dependent_rule_holds_on_the_direct_route_too() -> None:
    """The leading-character rule is the one thing here that reads two fields at once, so it
    is the one that could have been left in the parser by mistake. It is not: a registry
    response supplying `name` and `scope` separately is bound by the same pair of rules."""
    scoped = PackageRef(name="_private", version="1.0.0", scope="scope")
    assert scoped.name == "_private"
    assert scoped.fullname == "@scope/_private"

    with pytest.raises(ValidationError, match="unscoped"):
        PackageRef(name="_private", version="1.0.0")


@pytest.mark.parametrize("version", ["1.0.0+build.5", "1.0.0-rc.1+build.5", "1.0.0+001"])
def test_the_model_admits_build_metadata_constructed_directly(version: str) -> None:
    assert PackageRef(name="pkg", version=version).version == version


def test_a_parsed_reference_is_immutable_comparable_and_usable_as_a_key() -> None:
    """Findings are grouped by package, so a `PackageRef` has to be hashable, and nothing
    may re-point a pinned scan at a different artifact after the fact."""
    first = parse_npm_reference("express@4.21.2")
    second = parse_npm_reference("express@4.21.2")

    assert first == second
    assert len({first, second}) == 1
    with pytest.raises(ValidationError):
        first.version = "4.21.1"  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    """A field nobody declared must not smuggle a value past the boundary."""
    with pytest.raises(ValidationError):
        smuggled = {"name": "express", "version": "4.21.2", "tarball": "http://evil/"}
        PackageRef(**smuggled)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [b"express", bytearray(b"express"), 42, 3.5, True])
def test_a_field_must_be_text_rather_than_something_decodable_into_text(value: object) -> None:
    """Pydantic's lax mode decodes `bytes` and `bytearray` into `str`; this model is strict.

    The coercion is the problem, not the decoded value. A field that arrived as bytes would
    become a package name after PackProof picked a decoding for it — and `b"\xff"` fails
    inside that decoding rather than as a refusal this module wrote. Refusing the type keeps
    the decision where the boundary is.

    `b"../x"` was refused before this change and is refused now, one step earlier: every
    field validator still runs on whatever is left, so strictness narrows what enters without
    relaxing anything once inside.
    """
    with pytest.raises(ValidationError):
        PackageRef(name=value, version="1.0.0")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PackageRef(name="express", version=value)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PackageRef(name="node", version="1.0.0", scope=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, b"express", 42, 3.5, True])
def test_a_required_field_may_not_be_nothing(value: object) -> None:
    """`None` is refused where the field is required and accepted where it is not.

    Keeping `None` out of `name` and `version` matters for a different reason than keeping
    `bytes` out: `scope` is `str | None`, so `None` is a meaningful value for one field and
    nonsense for the other two, and the model's own types are what say which is which.
    """
    with pytest.raises(ValidationError):
        PackageRef(name=value, version="1.0.0")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PackageRef(name="express", version=value)  # type: ignore[arg-type]


def test_an_absent_or_none_scope_is_still_allowed() -> None:
    """`strict=True` applies to the `str` half of `str | None`, not to the `None`."""
    assert PackageRef(name="node", version="1.0.0").scope is None
    assert PackageRef(name="node", version="1.0.0", scope=None).scope is None


def test_ordinary_text_still_enters_on_every_route() -> None:
    """Keyword construction, `model_validate` from a decoded payload, and the parser.

    Strict mode is what Stage 2.2 will meet first: registry JSON decodes to a dict, and a
    registry that answered `{"name": 42}` has not supplied a package name. Coercing it into
    one would be the same silent substitution the look-alike digits are.
    """
    assert PackageRef(name="express", version="4.21.2").fullname == "express"
    assert PackageRef.model_validate({"name": "express", "version": "4.21.2"}).name == "express"
    assert (
        PackageRef.model_validate({"name": "node", "version": "1.0.0", "scope": "types"}).scope
        == "types"
    )
    assert parse_npm_reference("@types/node@22.10.2").scope == "types"

    for payload in (
        {"name": 42, "version": "4.21.2"},
        {"name": "express", "version": 421.2},
        {"name": "express", "version": "4.21.2", "scope": b"types"},
    ):
        with pytest.raises(ValidationError):
            PackageRef.model_validate(payload)


def test_a_str_subclass_is_normalised_to_plain_text() -> None:
    """Strict mode keeps the property lax mode had, which is the one that matters.

    A `str` subclass may define `__eq__` to answer `True` for everything. If the subclass
    survived into a frozen model, two different packages would compare equal and dedupe into
    one finding — so normalising to exact `str` is validation, and it holds on both modes.
    """

    class AlwaysEqual(str):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return 1

    first = PackageRef(name=AlwaysEqual("express"), version="1.0.0")
    second = PackageRef(name="lodash", version="1.0.0")

    assert type(first.name) is str
    assert first != second
    assert len({first, second}) == 2
