"""Validated application settings.

`Settings` is immutable and validated, so a bad value fails when the settings object is
built rather than halfway through a download. Nothing constructs it yet: the CLI has no
command that needs a workspace, so this becomes an operating reality when a `scan`
command calls `Settings.from_env()`. Environment overrides are read explicitly in
`from_env()` instead of through a settings library: the values that need configuring are
few, and an implicit binding would put configuration in two places to look for.

Size limits
-----------
`max_download_bytes` and `max_metadata_bytes` bound different objects and are not
interchangeable. The download ceiling covers a package tarball — an archive PackProof
writes to disk and later unpacks, so its size tracks what a package contains. The
metadata ceiling covers a registry packument, the JSON document listing every published
version of a package, whose size tracks how *long* a package has been published instead.
Measured against the public npm registry with no compression requested, `express` is
823 KB while `typescript` (15.9 MB), `@types/node` (11.2 MB) and `aws-sdk` (10.6 MB) are
each an order of magnitude larger. Reusing the tarball limit would set a metadata bound
from an archive's expected weight — too generous against a registry that streams forever,
and too tight for an old, heavily published package. The default therefore sits at roughly
twice the largest packument measured, and its ceiling stays well under the tarball's
because the entire document is accumulated in memory before a parser sees it.

Workspace policy
----------------
The workspace holds untrusted package content that PackProof has downloaded but not
analyzed. It must never overlap the code PackProof trusts and runs — writing scan
input into the installation directory or the source checkout would put downloaded
bytes next to the interpreter's own import path.

Therefore `workspace_dir` is rejected when it:

- is empty, whitespace-only, or names the current directory (as in "" or ".");
- contains a `..` component, at any position, in any spelling;
- carries a `~user` prefix that names no account, which is reported as a configuration
  error instead of surfacing the `RuntimeError` the standard library raises for it;
- resolves to something that already exists and is not a directory — including a symlink
  that points at nothing, because a file is not a place to extract into and a `mkdir`
  failing at extraction time is the worst moment to learn that;
- resolves (after `~` expansion and symlink resolution) to a protected root or to
  anything beneath one;
- contains a protected root, since a workspace that holds the installation lets every
  archive member name reach PackProof's own code — the filesystem root is the extreme
  case, and so is a home directory that holds a checkout.

Protected roots come from the running package and the running interpreter, never from
the current working directory:

- the `packproof` package directory, always;
- the interpreter's package roots (`sysconfig`'s `purelib` and `platlib`), whenever the
  interpreter reports usable ones. The environment imports from those directories
  whether or not PackProof was installed into them, which is what puts an editable
  install under the same rule as a wheel: untrusted content written there can shadow a
  real module either way;
- the repository root, only when three signals agree on the same directory: a
  `pyproject.toml` declaring this project, repository metadata (`.git`, which is a
  directory in a normal clone and a file in a worktree or submodule), and the package
  directory actually sitting at that candidate's `src/packproof`. Neither of the first
  two establishes identity alone — a manifest name is a copyable string, and a `.git`
  entry only says that *some* tree is versioned — and both can be dropped into any
  directory that holds a virtualenv, which is why the walk also asks whether this
  installation lives where the candidate claims. That is a structural heuristic, not
  cryptographic verification, and it recognises one layout: the `src/packproof` tree
  `pyproject.toml` builds.

The interpreter rule is asked of `sysconfig` rather than inferred from the package's
parent directory, so an unusual install whose parent is a home or working directory does
not end up banning everything beneath it. The repository rule refuses to guess, so a
wheel install or an unpacked release tarball gets package-and-environment protection
instead of a fabricated checkout root — and the default workspace stays valid there.

A workspace is refused rather than accepted when a filesystem check needed to answer any
of this fails: PackProof cannot confirm such a path is outside the protected locations,
and a permission error is not evidence of a safe directory. Each refusal names the role
of the root it reached — package directory, source checkout, or interpreter package root
— because the three call for different fixes from the operator, and the role travels with
the root from `_protected_roots_with_roles()` rather than being re-derived from the
filesystem while the message is being written: building a refusal must not add a
filesystem operation that can fail.

Two refusals that look alike are not alike. A workspace that cannot be inspected is a
property of the configured value, and naming a different directory may well fix it.
Protected locations that cannot be listed are a property of the installation, and no
`PACKPROOF_WORKSPACE_DIR` value helps while they stay unreadable — the two messages say
which of those the operator is looking at.

The default workspace is a per-platform cache directory so it does not move when the
tool is invoked from a different directory. Constructing `Settings` only computes and
validates a path: it never creates a directory. That happens later, at the operational
boundary that actually needs it.

Configuration validation is not archive-extraction security. A member path inside a
tarball is chosen by the attacker at extraction time, so the extractor must
independently bound each destination, reject unsafe members, and handle symlinks;
checking one configured directory here cannot do that job.

Secrets do not belong in this model yet. When registry or AI credentials arrive they
must be added as `pydantic.SecretStr` fields, which keeps them out of `repr()` and
therefore out of logs.
"""

import functools
import os
import sys
import sysconfig
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

APP_DIR_NAME = "PackProof"
WORKSPACE_DIR_NAME = "scan-workspace"

DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_METADATA_BYTES = 32 * 1024 * 1024

#: The whole packument is held in memory before it is parsed, so a metadata limit has to
#: be smaller than a tarball limit that streams to disk. See "Size limits" above.
MAX_METADATA_BYTES_CEILING = 256 * 1024 * 1024

ENV_HTTP_TIMEOUT_SECONDS = "PACKPROOF_HTTP_TIMEOUT_SECONDS"
ENV_MAX_DOWNLOAD_BYTES = "PACKPROOF_MAX_DOWNLOAD_BYTES"
ENV_MAX_METADATA_BYTES = "PACKPROOF_MAX_METADATA_BYTES"
ENV_WORKSPACE_DIR = "PACKPROOF_WORKSPACE_DIR"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
ENV_LOCALAPPDATA = "LOCALAPPDATA"

#: Settings field -> environment variable. The single place that binds the two, so an
#: error message can name the variable the operator actually set.
ENVIRONMENT_FIELDS: dict[str, str] = {
    "http_timeout_seconds": ENV_HTTP_TIMEOUT_SECONDS,
    "max_download_bytes": ENV_MAX_DOWNLOAD_BYTES,
    "max_metadata_bytes": ENV_MAX_METADATA_BYTES,
    "workspace_dir": ENV_WORKSPACE_DIR,
}

_GIBIBYTE = 1024**3


class ConfigurationError(ValueError):
    """A setting was present but unusable.

    Separate from `ValidationError` so the CLI can tell "the operator configured this
    wrongly" apart from "the code has a bug" and report an actionable message.
    """


def _acceptance_hint() -> str:
    """The part of every workspace error that tells the operator what to do instead.

    Deliberately avoids computing the real default: an error path must not depend on
    the same environment (`HOME`, cache variables) that may already be unusable.
    """
    return (
        "expected a dedicated directory for untrusted downloads, outside every location "
        f"PackProof protects, such as the per-user cache location, or set {ENV_WORKSPACE_DIR} to one"
    )


def _workspace_inspection_failure(configured: str, stage: str, exc: OSError) -> str:
    """Message text for the case where *the configured workspace* could not be inspected.

    One of the two inspection-failure builders, named after the thing that failed. This is
    the case a change of setting can fix: the operator pointed `workspace_dir` at a path
    whose location or contents this process cannot read, and pointing it somewhere else is
    a real remedy.

    `strerror` rather than `str(exc)`: the OSError's own text repeats a filename, and
    under an unreadable ancestor that filename is whatever the failing `stat` reached,
    not the path the operator wrote. The configured value is already quoted here, and
    quoting a second one invites a report that contradicts the configuration.

    The reason is *not* dropped: "Permission denied" and "File name too long" call for
    different fixes, and a refusal that hides which one happened is not actionable.
    """
    reason = exc.strerror or type(exc).__name__
    return (
        f"workspace_dir ({configured!r}) was refused because {stage}: {reason}. "
        "PackProof cannot confirm a path it cannot inspect is outside the locations it "
        f"protects, and refuses it rather than guessing; check the access permissions on "
        f"that path, or set {ENV_WORKSPACE_DIR} to a directory you can read"
    )


def _protected_roots_unavailable(configured: str, exc: OSError) -> str:
    """Message text for the case where *PackProof's own protected locations* could not be read.

    The other inspection-failure builder, and deliberately a different one: sharing a
    template with `_workspace_inspection_failure()` meant this case inherited the advice to
    repoint `PACKPROOF_WORKSPACE_DIR`, which cannot help. The workspace is not the thing
    that failed — the list of directories that must *not* become one did, and the next
    value would be validated against the same unreadable boundary. Advice that sends the
    operator to the wrong knob is worse than no advice, because it makes a working
    configuration look like the cause.

    So this names what could not be read, says in terms that the setting is not the fix,
    and points at the three locations whose permissions actually have to be checked. The
    configured value is still quoted: it is what a report has to show, and it is the value
    being refused either way.
    """
    reason = exc.strerror or type(exc).__name__
    return (
        f"workspace_dir ({configured!r}) is refused because the protected locations could "
        f"not be listed: {reason}. PackProof protects its own package directory, this "
        "install's source checkout when it is one, and the running interpreter's package "
        "directories, and it will not validate a workspace against a boundary it cannot "
        f"read rather than assume nothing is protected. Setting {ENV_WORKSPACE_DIR} to "
        "another directory does not help while those locations are unreadable — check the "
        "permissions on the PackProof installation, its checkout, and the interpreter's "
        "package directories"
    )


def _usable_absolute_path(raw: str) -> Path | None:
    """Return `raw` as an absolute path, or None if it is not usable as one.

    A relative or empty cache root is not a bug to raise over — the XDG and Windows
    specifications both treat it as meaningless, so falling back to the per-user
    default is the safe reading of a malformed environment.
    """
    if not raw.strip():
        return None
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else None


def default_workspace_dir() -> Path:
    """Where the workspace lives when the operator configures nothing.

    Per-platform application cache location, chosen at call time so tests can patch
    `sys.platform` and the environment instead of running on three operating systems.
    Standard library only — a package such as `platformdirs` would be one more
    dependency to audit for something `sys.platform` and two environment variables
    already express.
    """
    home = Path.home()

    if sys.platform == "darwin":
        base = home / "Library" / "Caches"
    elif sys.platform.startswith("win"):
        # LOCALAPPDATA is set by Windows for interactive users; a missing or relative
        # value means we are in a service context without one.
        base = (
            _usable_absolute_path(os.environ.get(ENV_LOCALAPPDATA, ""))
            or home / "AppData" / "Local"
        )
    else:
        # Linux and other POSIX systems: XDG_CACHE_HOME, else ~/.cache.
        base = _usable_absolute_path(os.environ.get(ENV_XDG_CACHE_HOME, "")) or home / ".cache"

    return base / APP_DIR_NAME / WORKSPACE_DIR_NAME


def _package_dir() -> Path:
    """The directory the running `packproof` package was imported from.

    The only place that turns `__file__` into a trust decision, so the whole policy is
    derived from the installation rather than the working directory.
    """
    return Path(__file__).resolve().parents[1]


def _manifest_declares_packproof(manifest: Path) -> bool:
    """Whether `manifest` is a readable project table naming this project.

    Every shape check here is load-bearing. `pyproject.toml` files above the package are
    not ours and can contain valid TOML with any type under `project` — a string, a
    number, a list — and `.get("project", {}).get("name")` on those raises
    `AttributeError`, which nothing in this module catches and which would therefore
    escape `Settings()` as a traceback instead of a configuration decision. Unrecognized
    is not an error, it is simply not PackProof, so this returns False.

    Only the exceptions a manifest read can genuinely produce are caught: `OSError` for
    unreadable files and directories, `UnicodeDecodeError` for a non-UTF-8 manifest, and
    `TOMLDecodeError` for malformed TOML.
    """
    if not manifest.is_file():
        return False
    try:
        declaration = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    project = declaration.get("project")
    if not isinstance(project, dict):
        return False
    name = project.get("name")
    return isinstance(name, str) and name.lower() == "packproof"


def _is_repository_directory(candidate: Path) -> bool:
    """Whether `candidate` is a working tree rather than a directory that merely exists.

    Version-control metadata is the signal that a directory *is* a repository of some
    kind: the manifest alone says what a project is called, which is a copyable string.
    `.git` is a directory in a normal clone and a file in a Git worktree or submodule, so
    either form counts, and its *contents* are never read — this answers "is there a
    working tree here", not "which repository does it belong to", and the second question
    is what the package-location signal below answers. No `.git` of any kind means no
    claim to a repository root is made.
    """
    metadata = candidate / ".git"
    return metadata.is_dir() or metadata.is_file()


def _holds_the_running_package(candidate: Path, package_dir: Path) -> bool:
    """Whether `candidate` is the checkout whose `src/packproof` *is* this installation.

    The third identity signal, and the only one that ties a claim to the code being
    executed. A manifest naming `packproof` and a `.git` entry can both be placed in a
    directory that has nothing to do with this package — a home directory holding a
    virtualenv is the ordinary case — and adopting such a directory as a protected root
    invalidates the default workspace, which always lives beneath that home.

    A path comparison, not an existence probe: `package_dir` is where the interpreter
    already loaded this package from, so it exists by construction, and a candidate that
    is its ancestor cannot be fictional. `Path.resolve()` on paths that do not exist
    would report them equal anyway, which is exactly why this check is written against
    the package directory rather than against a candidate `__init__.py` that could be
    absent and still compare equal.

    One layout is recognized, the one `pyproject.toml` builds with
    `packages = ["src/packproof"]`. A flat-layout checkout would not be adopted, and the
    loss runs in the safe direction: the package directory and the interpreter's roots
    stay protected, so refusing to claim a layout we do not build narrows the protected
    set instead of widening it.
    """
    return candidate / "src" / "packproof" == package_dir


@functools.cache
def _project_root() -> Path | None:
    """Repository root of a PackProof source checkout, or None when nothing proves one.

    Three signals must agree on the same ancestor directory: a manifest that declares
    this project, repository metadata, and the running package sitting at that
    directory's `src/packproof`. Each one on its own is copyable, generic, or unrelated
    to this installation, and any two of the three can be satisfied by a directory the
    package merely happened to be installed under. Requiring all three is what stops an
    unrelated `pyproject.toml` and a stray `.git` from promoting a home directory into a
    protected root — which would silently invalidate a perfectly ordinary default
    workspace. This is a structural heuristic, not cryptographic verification of
    repository identity; it establishes "this directory is the checkout this package was
    imported from" well enough to protect it, and nothing more.

    Walking up from the package rather than from the working directory keeps the answer a
    property of the installation. Reading the manifest with `tomllib` rather than matching
    a directory name is what makes the claim about content rather than naming: a folder
    called `PackProof` proves nothing.

    Checking the package location first also bounds the walk: the manifest is read only
    for the one ancestor that can possibly hold this installation, so an unrelated tree
    above a wheel install is refused without a single file being opened.

    Returns None for a wheel install and for an unpacked release archive, neither of
    which has repository metadata. Those installs are still protected by the package
    directory and the interpreter's package roots.

    The candidate is already resolved — `_package_dir()` resolves, and every ancestor of
    a resolved path is resolved — so it is returned unchanged rather than resolved twice.

    Memoized because the answer depends only on where the package was imported from,
    and validation runs for every Settings construction.
    """
    package_dir = _package_dir()
    for candidate in package_dir.parents:
        if not _holds_the_running_package(candidate, package_dir):
            continue
        if _manifest_declares_packproof(candidate / "pyproject.toml") and _is_repository_directory(
            candidate
        ):
            return candidate
    return None


def _interpreter_package_roots() -> tuple[Path, ...]:
    """Where this interpreter imports installed packages from.

    `sysconfig` is the standard library's own answer to that question, so the check
    needs no dependency and no heuristic about directory names, and it is asked of the
    *interpreter* rather than inferred from where PackProof happens to sit — which is
    what makes an editable install, whose source lives in a checkout, still report the
    environment it imports from. Both `purelib` and `platlib` are read because a
    distribution can place them in different directories.

    A relative or empty scheme value is dropped rather than guessed at, because there is
    no location to protect: the answer names nothing usable. Dropping it cannot widen the
    workspace boundary — protection is a list of refusals, and the package directory and
    repository root are still on it — so the policy weakens only in the sense that an
    unusable path never contributed anything.

    Not memoized: the answer describes the running interpreter, and a test must be able
    to replace it.
    """
    paths = sysconfig.get_paths()
    roots = (_usable_absolute_path(paths.get(key) or "") for key in ("purelib", "platlib"))
    return tuple(dict.fromkeys(root.resolve() for root in roots if root is not None))


def _protected_roots_with_roles() -> tuple[tuple[Path, str], ...]:
    """The protected directories, each paired with the role that explains why it is here.

    The single source of truth for the protected boundary: `protected_roots()` is this
    function's path-only projection, and `_validate_workspace()` consumes the pairs so a
    refusal can name the role without asking the filesystem anything.

    Roles are decided during discovery, where the identity of each path is already known
    from the moment it is produced, rather than reconstructed later by comparing the path
    against freshly computed answers. Two things follow. A refusal performs no filesystem
    operation, so the exception-safety rule that bounds every other filesystem step in the
    validator holds for diagnostic text too. And the role reported is the role the root was
    discovered with, not the role a second `resolve()` of the package path happens to
    produce at message time — those can disagree when a symlink changes underneath a
    long-lived `Settings`, and a security message that contradicts the check that produced
    it is worse than one with less detail.

    Ordered package, checkout, interpreter. A path discovered by more than one source keeps
    the first role, which is the most specific claim about it: a directory that is both the
    package and an interpreter root is reported as the package. `dict.setdefault` gives that
    while preserving insertion order, which is what `dict.fromkeys` did for the bare paths.

    Can raise `OSError`, for the same reason as `protected_roots()`, described there.
    """
    discovered: list[tuple[Path, str]] = [(_package_dir(), PACKAGE_ROLE)]

    project_root = _project_root()
    if project_root is not None:
        discovered.append((project_root, CHECKOUT_ROLE))

    discovered.extend((root, ENVIRONMENT_ROLE) for root in _interpreter_package_roots())

    unique: dict[Path, str] = {}
    for root, role in discovered:
        unique.setdefault(root, role)
    return tuple(unique.items())


def protected_roots() -> tuple[Path, ...]:
    """Directories that must never contain untrusted downloaded content.

    The path-only view of `_protected_roots_with_roles()`, kept as this module's stable
    interface: callers that only ask whether a directory is protected should not have to
    unpack roles they will not use. Code that *names* a root in a message must use the
    role-bearing helper instead, because the role is a fact established at discovery.

    Three sources, all independent of the operator's working directory: the package
    directory of the running `packproof`, the repository root when the installation is
    demonstrably a checkout, and the running interpreter's package roots. They are
    returned in that order, which is also the order the roles are reported in, so the
    most specific name for a directory that is more than one thing wins.

    The interpreter's roots are protected whatever contains them, not only when PackProof
    was installed into them. That distinction is the whole difference between a wheel and
    an editable install: an editable one keeps its source in the repository, so keying
    protection to the package's parent directory left the environment's `site-packages`
    open — and a workspace directory named `packproof`, or `pydantic`, inside a directory
    on `sys.path` shadows a real module just as effectively from either kind of install.

    Asking `sysconfig` instead of assuming a package's parent is a project root keeps the
    set as narrow as the claim: an ordinary parent directory — a home, a working
    directory, an interpreter installed somewhere unusual — is never protected as a side
    effect, and everything beneath it stays available as a workspace.

    Can raise `OSError`: both repository-metadata checks and the manifest probe read the
    filesystem, and an ancestor that was traversable when the package was imported can
    have become unreadable since. `_validate_workspace()` is the caller that has to turn
    that into a refusal.
    """
    return tuple(root for root, _ in _protected_roots_with_roles())


#: The three things a protected root can be, worded for an operator who has to act on
#: them: a checkout is fixed by pointing elsewhere, an environment by not naming a
#: sibling of an installed module, a package directory by leaving the installation alone.
PACKAGE_ROLE = "PackProof's own package directory"
CHECKOUT_ROLE = "the PackProof source checkout"
ENVIRONMENT_ROLE = "the running interpreter's package directory"


def _expand_user(value: Path) -> Path:
    """`Path.expanduser()` with its one failure mode turned into a configuration error.

    A leading `~name` is looked up in the account database, and when no such account
    exists the standard library raises `RuntimeError` — for the obvious `~bob/ws` and for
    spellings nobody intends, like `"~ /x"`, where the "user" is a single space.

    That matters here because of how the validator reports problems: pydantic converts a
    `ValueError` raised in a field validator into `ValidationError`, which is exactly why
    `ConfigurationError` subclasses `ValueError` and travels the whole path. It does not
    convert `RuntimeError`, so left alone that exception would escape `Settings()`,
    escape `from_env()`'s handler, and reach the operator as a traceback — a traceback
    that also names no configuration variable.

    Only `RuntimeError` is caught, and the original is chained with `raise ... from`, so
    `_validate_workspace()`'s own `ConfigurationError` carries it as `__cause__`. Nothing
    falls back to another directory: an unexpandable home is a refusal, not a hint.
    """
    try:
        return value.expanduser()
    except RuntimeError as exc:
        raise ConfigurationError(
            f"workspace_dir names a home directory that cannot be resolved ({str(value)!r}): "
            f"{exc}. Write '~' for this user's home, or give the path directly; "
            + _acceptance_hint()
        ) from exc


def _validate_workspace(value: Path) -> Path:
    """Normalize and vet a configured workspace path.

    Returns the operator's value with `~` expanded and made absolute — the *lexical*
    path, deliberately kept so that `Settings` reports what was configured rather than an
    unrelated directory the filesystem happened to resolve it to. Containment is checked
    against `Path.resolve()` output instead, which is the only spelling that answers the
    security question. `workspace_path` re-resolves on demand and is what filesystem
    operations must use; the two may disagree when a symlink is involved.

    Raises `ConfigurationError` with the reason when the value is unusable, including
    when a filesystem check cannot be completed at all. Raises nothing else: every
    operation below that touches the filesystem is bounded by a handler, because an
    `OSError` escaping a field validator is not converted by pydantic the way a
    `ValueError` is, and would reach the operator as a traceback naming no setting.

    Wrapping the calls is not enough on its own, because a refusal is assembled after the
    handlers have been left: text built by asking the filesystem again could fail in a
    place no handler covers. Every message here is assembled only from values already in
    hand — the resolved path, and the roots with the roles they were discovered carrying.
    """
    text = str(value)

    if not text.strip():
        raise ConfigurationError(
            f"workspace_dir is empty or whitespace-only ({text!r}); {_acceptance_hint()}"
        )

    if not value.parts:
        # Path('.'), Path('') and Path('./') all mean "here", which for a security tool
        # would drop untrusted downloads into whatever directory it happens to be run from.
        raise ConfigurationError(
            f"workspace_dir must name a directory, not the current directory ({text!r}); "
            + _acceptance_hint()
        )

    if ".." in value.parts:
        raise ConfigurationError(
            f"workspace_dir must not contain '..' components ({text!r}); give the directory "
            "itself rather than a path that walks into it from somewhere else"
        )

    expanded = _expand_user(value)

    # Two boundaries, not one broad `try`. `absolute()` asks the operating system for the
    # working directory — which fails outright if that directory has been deleted since
    # the process started — and `resolve()` walks the path component by component. Each
    # can raise `PermissionError`, `NotADirectoryError` or an `ENAMETOOLONG` `OSError`,
    # and a value whose location is unknown cannot be compared to anything, so this stage
    # has to be its own refusal rather than feeding an unnormalized path onward.
    try:
        absolute = expanded if expanded.is_absolute() else expanded.absolute()
        resolved = absolute.resolve()
    except OSError as exc:
        raise ConfigurationError(
            _workspace_inspection_failure(text, "the path could not be located", exc)
        ) from exc

    # Listing the protected roots reads the filesystem too (repository metadata, an
    # ancestor manifest, `sysconfig`). Failing to produce the list is a reason to refuse
    # the workspace, never a reason to accept it: an empty boundary would let any path
    # through, which is the opposite of what the check is for.
    #
    # This is the one filesystem failure in the validator that is *not* about the
    # operator's value, so it gets its own message builder — see
    # `_protected_roots_unavailable()` for why repointing the setting cannot fix it.
    try:
        roots = _protected_roots_with_roles()
    except OSError as exc:
        raise ConfigurationError(_protected_roots_unavailable(text, exc)) from exc

    # Everything below is a comparison between paths that are already in hand. No
    # filesystem operation, and no `_package_dir()`-style re-resolution, including in the
    # text of a refusal: the role arrived with the root, so a message that failed to be
    # built could not fail for a reason the boundary above does not already cover.
    for root, role in roots:
        if resolved == root or resolved.is_relative_to(root):
            raise ConfigurationError(
                f"workspace_dir resolves to {resolved}, inside {role} at {root}. "
                "Untrusted package content must not be written there; " + _acceptance_hint()
            )
        if root.is_relative_to(resolved):
            # Containment has to be checked in both directions: a workspace that
            # *holds* the installation is as unsafe as one inside it, because every
            # member name the archive supplies would then be free to land on
            # PackProof's own code. `/` is the extreme case.
            raise ConfigurationError(
                f"workspace_dir resolves to {resolved}, which contains {role} "
                f"at {root}. Extracted members could reach that code from here; "
                + _acceptance_hint()
            )

    # A configured workspace need not exist yet — validation must not create it — but
    # whatever is already there has to be something that can hold extracted files. Both
    # checks run on the resolved location, so a link that reaches a regular file is
    # refused for what it reaches rather than for how it was spelled, and no extension is
    # consulted: `.txt` proves nothing about a filesystem type, and a directory called
    # `x.txt` is perfectly usable.
    #
    # The dangling case has to be asked of `absolute` rather than `resolved`, because
    # `resolve()` follows the link and returns its target — which for a link to nothing
    # looks exactly like a path that was never created.
    #
    # All three probes are evaluated together, before any decision is taken, so an
    # ancestor that cannot be searched fails once and loudly. Short-circuiting between
    # them would leave the second and third raising on their own. `is_dir()` reports False
    # for a path that does not exist, so the predicates below still mean what they say.
    try:
        is_link = absolute.is_symlink()
        exists = resolved.exists()
        is_directory = resolved.is_dir()
    except OSError as exc:
        raise ConfigurationError(
            _workspace_inspection_failure(text, "what is already there could not be inspected", exc)
        ) from exc

    if is_link and not exists:
        raise ConfigurationError(
            f"workspace_dir resolves to {resolved}, a symlink to a location that does not "
            f"exist. Point it at a real directory or remove it; {_acceptance_hint()}"
        )
    if exists and not is_directory:
        raise ConfigurationError(
            f"workspace_dir resolves to {resolved}, which exists and is not a directory. "
            f"Extracted packages need a directory to be created in; {_acceptance_hint()}"
        )

    return absolute


class Settings(BaseModel):
    """Runtime limits for retrieving and unpacking candidate packages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    http_timeout_seconds: float = Field(
        default=DEFAULT_HTTP_TIMEOUT_SECONDS,
        gt=0,
        le=300,
        description="Per-request timeout for registry and source-host calls.",
    )
    max_download_bytes: int = Field(
        default=DEFAULT_MAX_DOWNLOAD_BYTES,
        gt=0,
        le=2 * _GIBIBYTE,
        description="Hard ceiling on a downloaded tarball before the stream is abandoned.",
    )
    max_metadata_bytes: int = Field(
        default=DEFAULT_MAX_METADATA_BYTES,
        gt=0,
        le=MAX_METADATA_BYTES_CEILING,
        description=(
            "Hard ceiling on a registry metadata response, enforced while reading the body "
            "rather than by trusting a Content-Length header."
        ),
    )
    workspace_dir: Path = Field(
        default_factory=default_workspace_dir,
        validate_default=True,
        description=(
            "Directory for downloaded, untrusted package content, as configured: `~` "
            "expanded and made absolute, but not symlink-resolved. Use workspace_path "
            "for filesystem work."
        ),
    )

    @field_validator("workspace_dir")
    @classmethod
    def _workspace_is_safe(cls, value: Path) -> Path:
        return _validate_workspace(value)

    @property
    def workspace_path(self) -> Path:
        """The workspace as the filesystem sees it right now: absolute, symlinks resolved.

        This is the value filesystem security checks must compare against. Resolved
        rather than merely absolute, because a containment decision made on lexical text
        answers the wrong question — `/tmp/a/../../etc` and `/etc` are different strings
        for the same directory. Consumers that check archive members must resolve the
        member path and compare against this value.

        It is deliberately *not* the same thing as `workspace_dir`, which keeps the
        operator's spelling so configuration can be reported back as it was written. The
        two differ whenever a symlink is involved, and this one is re-resolved on every
        access: the validation done at construction time describes the filesystem as it
        was then, so a link swapped in afterwards changes what this returns without
        changing any decision that was already made. That gap is not closed here and
        cannot be — whoever writes into the workspace must resolve and check again at the
        moment of writing, per member, which is what the extraction boundary is for.
        """
        return self.workspace_dir.expanduser().resolve()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from recognised `PACKPROOF_*` variables only.

        Unrecognised variables are ignored on purpose; a stray environment variable
        must not silently change a security limit.
        """
        source = os.environ if env is None else env
        overrides: dict[str, Any] = {}
        for field, variable in ENVIRONMENT_FIELDS.items():
            value = source.get(variable)
            if value is not None:
                overrides[field] = value

        try:
            return cls(**overrides)
        except ValidationError as exc:
            raise ConfigurationError(_describe(overrides, exc)) from exc


def _describe(overrides: Mapping[str, Any], error: ValidationError) -> str:
    """Name the environment variables the operator set, not just the failing field."""
    variables = ", ".join(sorted(ENVIRONMENT_FIELDS[name] for name in overrides)) or "none"
    return f"invalid PackProof configuration (overridden: {variables})\n{error}"
