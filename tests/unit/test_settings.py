"""Settings: defaults, bounds, immutability, environment overrides, workspace policy."""

import errno
import os
import re
import shutil
import stat
import sys
import sysconfig
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

from packproof.config import ConfigurationError, Settings, default_workspace_dir, settings
from packproof.config.settings import (
    APP_DIR_NAME,
    CHECKOUT_ROLE,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_MAX_DOWNLOAD_BYTES,
    DEFAULT_MAX_METADATA_BYTES,
    ENV_HTTP_TIMEOUT_SECONDS,
    ENV_LOCALAPPDATA,
    ENV_MAX_DOWNLOAD_BYTES,
    ENV_MAX_METADATA_BYTES,
    ENV_WORKSPACE_DIR,
    ENV_XDG_CACHE_HOME,
    ENVIRONMENT_FIELDS,
    ENVIRONMENT_ROLE,
    MAX_METADATA_BYTES_CEILING,
    PACKAGE_ROLE,
    WORKSPACE_DIR_NAME,
    _interpreter_package_roots,
    _protected_roots_with_roles,
    _validate_workspace,
    protected_roots,
)

# These tests describe a PackProof *source checkout*, so both roots are taken by role
# rather than by position in `protected_roots()` — that tuple now carries the
# interpreter's package roots too, and reading it by index would silently rebind
# PROJECT_ROOT to `site-packages` and make a dozen tests assert the wrong thing. The
# assertion below fails loudly, with a reason, if the suite is ever run from an install
# that is not a checkout.
PACKAGE_ROOT = settings._package_dir()
PROJECT_ROOT = settings._project_root()
assert PROJECT_ROOT is not None, "these tests must run against a PackProof source checkout"
assert (PROJECT_ROOT / ".git").exists(), "the checkout must carry repository metadata"


def simulate_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    home: Path,
    cache_variable: str | None = None,
    cache_value: str | None = None,
) -> None:
    """Pretend to run on `platform` with `home`, without leaving this machine.

    `sys.platform` is read at call time by `default_workspace_dir()`, and `Path.home()`
    reads `HOME`/`USERPROFILE`, so the whole platform decision is patchable.
    """
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv(ENV_XDG_CACHE_HOME, raising=False)
    monkeypatch.delenv(ENV_LOCALAPPDATA, raising=False)
    if cache_variable is not None:
        assert cache_value is not None
        monkeypatch.setenv(cache_variable, cache_value)


# --------------------------------------------------------------------------- existing


def test_defaults_are_valid_without_any_environment_input() -> None:
    settings = Settings()

    assert settings.http_timeout_seconds == DEFAULT_HTTP_TIMEOUT_SECONDS == 30.0
    assert settings.max_download_bytes == DEFAULT_MAX_DOWNLOAD_BYTES == 50 * 1024 * 1024
    assert settings.max_metadata_bytes == DEFAULT_MAX_METADATA_BYTES == 32 * 1024 * 1024
    assert settings.http_timeout_seconds > 0
    assert settings.max_download_bytes > 0
    assert settings.max_metadata_bytes > 0
    assert settings.workspace_dir.is_absolute()
    assert settings.workspace_dir.parent.name == APP_DIR_NAME
    assert settings.workspace_dir.name == WORKSPACE_DIR_NAME


def test_settings_are_immutable_once_built() -> None:
    """A scanned package must not be able to influence a limit mid-run, and neither
    must a stray assignment in some future module."""
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.max_download_bytes = 10**12  # type: ignore[misc]


def test_unknown_setting_names_are_rejected() -> None:
    """Typos in configuration fail loudly instead of being silently ignored."""
    with pytest.raises(ValidationError):
        Settings(http_timeout=30)  # type: ignore[call-arg]


@pytest.mark.parametrize("timeout", [0, -1.5, 301, 10_000])
def test_http_timeout_is_bounded(timeout: float) -> None:
    with pytest.raises(ValidationError):
        Settings(http_timeout_seconds=timeout)


@pytest.mark.parametrize("size", [0, -1, 3 * 1024**3])
def test_download_size_limit_is_bounded(size: int) -> None:
    with pytest.raises(ValidationError):
        Settings(max_download_bytes=size)


@pytest.mark.parametrize("size", [0, -1, MAX_METADATA_BYTES_CEILING + 1, 2 * 1024**3])
def test_metadata_size_limit_is_bounded(size: int) -> None:
    """A metadata ceiling that could be lifted to a tarball's size would let a registry
    stream until the process ran out of memory, so the bound is part of the policy."""
    with pytest.raises(ValidationError):
        Settings(max_metadata_bytes=size)


def test_metadata_and_download_limits_are_separate_settings() -> None:
    """Moving one limit must not move the other: they bound different objects."""
    settings = Settings(max_metadata_bytes=1024)

    assert settings.max_metadata_bytes == 1024
    assert settings.max_download_bytes == DEFAULT_MAX_DOWNLOAD_BYTES


def test_workspace_dir_rejects_the_current_directory() -> None:
    """Untrusted downloads must not land in the project tree."""
    for unsafe in ("", ".", "./"):
        with pytest.raises(ValidationError):
            Settings(workspace_dir=unsafe)


def test_workspace_path_is_absolute_and_expands_the_home_directory() -> None:
    settings = Settings(workspace_dir="~/packproof-scan")

    assert settings.workspace_path.is_absolute()
    assert str(Path.home()) in str(settings.workspace_path)


def test_relative_workspace_is_anchored_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`workspace_path` resolves against the caller's directory, not the package's."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(workspace_dir="scan-workspace")

    assert settings.workspace_path == Path.cwd() / "scan-workspace"
    assert Path(settings.workspace_path).parent == Path.cwd()


def test_environment_variables_are_namespaced_to_packproof() -> None:
    """A generic name like HTTP_TIMEOUT would let unrelated tooling move a security
    limit, so every binding must be prefixed."""
    assert ENVIRONMENT_FIELDS, "environment bindings disappeared"

    for variable in ENVIRONMENT_FIELDS.values():
        assert variable.startswith("PACKPROOF_"), variable


def test_every_environment_binding_names_a_real_setting_field() -> None:
    """`from_env()` builds `Settings(**overrides)` from these keys, so a binding left
    behind when a field is renamed would fail as an obscure validation error rather
    than as a missing configuration."""
    fields = set(Settings.model_fields)
    unbound = set(ENVIRONMENT_FIELDS) - fields

    assert not unbound, f"bound but not configured: {unbound}"


def test_from_env_applies_only_the_recognised_variables(tmp_path: Path) -> None:
    workspace = tmp_path / "scan-workspace"
    env = {
        ENV_HTTP_TIMEOUT_SECONDS: "45",
        ENV_MAX_DOWNLOAD_BYTES: "1048576",
        ENV_MAX_METADATA_BYTES: "2097152",
        ENV_WORKSPACE_DIR: str(workspace),
        "PATH": "/usr/bin",  # unrelated
        "PACKPROOF_HTTP_TIMEOUT": "999",  # near-miss name must not bind
    }

    settings = Settings.from_env(env)

    assert settings.http_timeout_seconds == 45.0
    assert settings.max_download_bytes == 1048576
    assert settings.max_metadata_bytes == 2097152
    assert settings.workspace_dir == workspace


def test_from_env_falls_back_to_defaults_when_nothing_is_set() -> None:
    assert Settings.from_env({}) == Settings()


def test_from_env_reports_an_actionable_error_instead_of_falling_back_silently() -> None:
    with pytest.raises(ConfigurationError, match=ENV_HTTP_TIMEOUT_SECONDS):
        Settings.from_env({ENV_HTTP_TIMEOUT_SECONDS: "not-a-number"})

    with pytest.raises(ConfigurationError, match="workspace_dir"):
        Settings.from_env({ENV_WORKSPACE_DIR: "."})


# ------------------------------------------------------------------ A. default location


def test_default_workspace_on_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "darwin", home)
    expected = home / "Library" / "Caches" / APP_DIR_NAME / WORKSPACE_DIR_NAME

    assert default_workspace_dir() == expected
    assert Settings().workspace_dir == expected


def test_default_workspace_on_linux_with_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cache = tmp_path / "shared-cache"
    simulate_platform(monkeypatch, "linux", home, ENV_XDG_CACHE_HOME, str(cache))

    assert Settings().workspace_dir == cache / APP_DIR_NAME / WORKSPACE_DIR_NAME


def test_default_workspace_on_linux_falls_back_to_user_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "linux", home)

    assert Settings().workspace_dir == home / ".cache" / APP_DIR_NAME / WORKSPACE_DIR_NAME


def test_default_workspace_ignores_a_relative_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """XDG requires an absolute path; a relative one is not a location we can trust."""
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "linux", home, ENV_XDG_CACHE_HOME, "relative/cache")

    assert Settings().workspace_dir == home / ".cache" / APP_DIR_NAME / WORKSPACE_DIR_NAME


def test_default_workspace_ignores_an_empty_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "linux", home, ENV_XDG_CACHE_HOME, "   ")

    assert Settings().workspace_dir == home / ".cache" / APP_DIR_NAME / WORKSPACE_DIR_NAME


def test_default_workspace_on_windows_with_localappdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    local = tmp_path / "AppData" / "Local"
    simulate_platform(monkeypatch, "win32", home, ENV_LOCALAPPDATA, str(local))

    assert Settings().workspace_dir == local / APP_DIR_NAME / WORKSPACE_DIR_NAME


def test_default_workspace_on_windows_without_localappdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "win32", home)

    assert (
        Settings().workspace_dir == home / "AppData" / "Local" / APP_DIR_NAME / WORKSPACE_DIR_NAME
    )


def test_default_workspace_does_not_depend_on_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this milestone removes: the same install, run from two directories."""
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "linux", home)
    first_cwd, second_cwd = tmp_path / "one", tmp_path / "two"
    first_cwd.mkdir()
    second_cwd.mkdir()

    monkeypatch.chdir(first_cwd)
    from_first = Settings().workspace_dir
    monkeypatch.chdir(second_cwd)
    from_second = Settings().workspace_dir

    assert from_first == from_second == home / ".cache" / APP_DIR_NAME / WORKSPACE_DIR_NAME


def test_app_directory_name_is_used_consistently_across_platforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each platform gets its conventional cache base, but the application-specific
    tail is the same, so an operator can find the workspace however they were told the
    platform looks."""
    home = tmp_path / "home"
    observed = []
    for platform in ("darwin", "linux", "win32"):
        simulate_platform(monkeypatch, platform, home)
        observed.append(default_workspace_dir())

    assert observed == [
        home / "Library" / "Caches" / APP_DIR_NAME / WORKSPACE_DIR_NAME,
        home / ".cache" / APP_DIR_NAME / WORKSPACE_DIR_NAME,
        home / "AppData" / "Local" / APP_DIR_NAME / WORKSPACE_DIR_NAME,
    ]
    assert {path.parent.name for path in observed} == {APP_DIR_NAME}
    assert {path.name for path in observed} == {WORKSPACE_DIR_NAME}
    assert len(set(observed)) == 3, "the platforms must not collide onto one location"


def test_a_default_that_resolves_inside_the_project_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`validate_default=True` means the platform default is held to the same policy."""
    simulate_platform(monkeypatch, "linux", tmp_path, ENV_XDG_CACHE_HOME, str(PROJECT_ROOT))

    with pytest.raises(ValidationError, match=f"inside {CHECKOUT_ROLE}"):
        Settings()


# ----------------------------------------------------------------------- B. invalid paths


@pytest.mark.parametrize(
    "unsafe",
    [
        pytest.param("", id="empty-string"),
        pytest.param("  ", id="whitespace-only"),
        pytest.param("\t ", id="tab-and-space"),
        pytest.param(".", id="dot"),
        pytest.param("./", id="dot-slash"),
        pytest.param("..", id="parent"),
        pytest.param("../outside", id="parent-then-name"),
        pytest.param("src/packproof/cli/../main.py", id="traversal-inside-package"),
    ],
)
def test_workspace_values_that_cannot_hold_untrusted_content(unsafe: str) -> None:
    with pytest.raises(ValidationError):
        Settings(workspace_dir=unsafe)


def test_traversal_spelling_cannot_reach_outside_and_back(tmp_path: Path) -> None:
    """A path with `..` may resolve anywhere, so the component is rejected rather than
    resolved and hoped to land somewhere safe."""
    outside = tmp_path / "elsewhere" / ".." / "elsewhere"

    with pytest.raises(ValidationError, match=r"\.\."):
        Settings(workspace_dir=str(outside))


@pytest.mark.parametrize(
    "build,expectation",
    [
        pytest.param(
            lambda project: project / "src",
            f"contains {PACKAGE_ROLE}",
            id="source-directory",
        ),
        pytest.param(
            lambda project: project / "src" / "packproof",
            f"inside {PACKAGE_ROLE}",
            id="package-directory",
        ),
        pytest.param(
            lambda project: project / "src" / "packproof" / "cli",
            f"inside {PACKAGE_ROLE}",
            id="inside-package",
        ),
        pytest.param(lambda project: project / "tests", f"inside {CHECKOUT_ROLE}", id="test-tree"),
        pytest.param(lambda project: project, f"contains {PACKAGE_ROLE}", id="project-root-itself"),
    ],
)
def test_workspace_inside_the_source_checkout_is_rejected(build, expectation: str) -> None:
    """Every part of a checkout is unreachable, and the refusal says which root it hit.

    The role is not decoration: `src` is refused because it *holds* the package, while
    `tests` is refused because it sits inside the checkout, and an operator reading the
    message needs to know which of the two rules fired. Which root is named is
    deterministic — the first one in `protected_roots()` that is related to the path.
    """
    with pytest.raises(ValidationError, match=expectation):
        Settings(workspace_dir=str(build(PROJECT_ROOT)))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: PROJECT_ROOT.parent, id="directory-holding-the-checkout"),
        pytest.param(lambda: PACKAGE_ROOT.parents[-1], id="filesystem-or-drive-root"),
    ],
)
def test_a_workspace_that_would_contain_packproof_is_rejected(build) -> None:
    """Containment is checked from both sides.

    `/` is not *inside* the installation, so the first rule lets it through — yet a
    workspace holding the installation makes the extraction boundary as wide as the
    filesystem, and a member named `Users/you/…/PackProof/src/evil.js` would land on
    PackProof's own code.

    The package directory is the root reported, because it is the first source in
    `protected_roots()` and any directory that holds the checkout necessarily holds the
    package inside it — there is no spelling that contains the checkout and not the
    package, so the checkout role cannot appear on this side of the check.
    """
    ancestor = build()
    if ancestor in protected_roots():
        pytest.skip("this layout leaves nothing above the protected roots to test")

    with pytest.raises(ValidationError, match=f"contains {PACKAGE_ROLE}") as excinfo:
        Settings(workspace_dir=str(ancestor))

    assert str(ancestor) in str(excinfo.value)


def test_relative_spelling_of_a_protected_directory_is_also_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejection is by resolved location, so `src` from the checkout root is caught the
    same way the absolute path is — no literal string matching involved."""
    monkeypatch.chdir(PROJECT_ROOT)

    with pytest.raises(ValidationError, match=f"contains {PACKAGE_ROLE}"):
        Settings(workspace_dir="src")
    with pytest.raises(ValidationError, match=f"inside {PACKAGE_ROLE}"):
        Settings(workspace_dir="./src/packproof")


def test_protected_roots_are_derived_from_the_package_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patching the cwd must not widen or narrow what is considered trusted."""
    monkeypatch.chdir(tmp_path)
    roots_before = protected_roots()

    monkeypatch.chdir(PROJECT_ROOT)
    assert protected_roots() == roots_before
    assert PACKAGE_ROOT in roots_before


def test_project_root_is_identified_by_the_packproof_manifest() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert PACKAGE_ROOT.is_relative_to(PROJECT_ROOT)
    # Three signals, not one: the manifest says what the project is called, and a name is
    # a copyable string; repository metadata says a directory is a working tree, and any
    # directory can hold one; the package's own location is what ties both claims to the
    # code actually running.
    assert (PROJECT_ROOT / ".git").exists()
    assert PROJECT_ROOT / "src" / "packproof" == PACKAGE_ROOT


def test_a_wheel_install_protects_the_interpreters_package_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simulate_package_dir
) -> None:
    """An installed wheel has no checkout, but it does sit inside the directory the
    interpreter imports from — and untrusted content written beside it could shadow a
    real module. `sysconfig` names that directory, so the rule needs no guesswork."""
    site_packages = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    simulate_package_dir(site_packages / "packproof")
    monkeypatch.setattr(
        sysconfig,
        "get_paths",
        lambda: {"purelib": str(site_packages), "platlib": str(site_packages)},
    )

    assert site_packages.resolve() in protected_roots()

    for sibling in (
        site_packages / "packproof-0.1.0.dist-info",
        site_packages / "leftover-scan",
    ):
        with pytest.raises(ValidationError, match=f"inside {ENVIRONMENT_ROLE}"):
            Settings(workspace_dir=sibling)

    # Above the package root, not inside it: refused for the other reason.
    with pytest.raises(ValidationError, match=f"contains {PACKAGE_ROLE}"):
        Settings(workspace_dir=site_packages.parent)


def test_an_ordinary_parent_directory_is_not_promoted_to_a_protected_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flat install whose parent happens to be a home or working directory must not
    ban everything beneath it.

    The environment is protected because `sysconfig` names it, exactly and only; its
    parent is an ordinary directory and stays available. Under the corrected policy the
    interpreter's package root is on the list even though this checkout's package is not
    inside it — that is the editable-install rule doing its job, and it must not be
    allowed to expand upward into `tmp_path` itself.
    """
    elsewhere = tmp_path / "interpreter-elsewhere"
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(elsewhere)})

    assert protected_roots() == (PACKAGE_ROOT, PROJECT_ROOT, elsewhere.resolve())
    assert elsewhere.parent not in protected_roots()
    assert tmp_path not in protected_roots()
    assert Settings(workspace_dir=tmp_path / "scan").workspace_dir == tmp_path / "scan"


def test_a_relative_or_missing_interpreter_package_path_is_not_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sysconfig` is consulted, not trusted: a scheme that yields nothing usable adds
    nothing to the protected set rather than raising during configuration."""
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda: {"purelib": "relative/site-packages", "platlib": ""}
    )

    assert _interpreter_package_roots() == ()


def test_error_message_names_the_value_and_the_variable_to_set(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(workspace_dir=str(PACKAGE_ROOT))

    message = str(excinfo.value)
    assert "workspace_dir" in message
    assert str(PACKAGE_ROOT) in message  # which value
    assert PACKAGE_ROLE in message  # which root, by what it is
    assert ENV_WORKSPACE_DIR in message  # how to fix it


# -------------------------------------------------------------------------- C. valid paths


def test_external_absolute_workspace_is_accepted_and_stored_verbatim(tmp_path: Path) -> None:
    workspace = tmp_path / "scans"

    settings = Settings(workspace_dir=workspace)

    assert settings.workspace_dir == workspace
    assert settings.workspace_path == workspace.resolve()


@pytest.mark.parametrize(
    "name",
    [
        "scan workspace",
        "scan-workspace (2026)",
        "scan_workspace.v1~final",
        "scans, with commas",
        "café-scans",
    ],
)
def test_ordinary_characters_in_a_workspace_name_are_not_rejected(
    tmp_path: Path, name: str
) -> None:
    """Spaces, punctuation and non-ASCII names are legitimate; only the traversal
    component and protected locations are refused."""
    assert Settings(workspace_dir=str(tmp_path / name)).workspace_dir == tmp_path / name


def test_nonexistent_external_workspace_is_accepted_without_being_created(
    tmp_path: Path,
) -> None:
    """Validation must not touch the filesystem: the workspace is created later, by the
    code that actually needs it."""
    workspace = tmp_path / "does" / "not" / "exist"

    settings = Settings(workspace_dir=workspace)

    assert settings.workspace_path == tmp_path.resolve() / "does" / "not" / "exist"
    assert not settings.workspace_path.exists()


def test_construction_never_creates_the_default_workspace_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "linux", home)

    assert not Settings().workspace_dir.exists()


# ---------------------------------------------------------------------- D. symlink handling


def _make_symlink(link: Path, target: Path) -> bool:
    """Create `link -> target`, returning False when this platform forbids it."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):  # Windows needs privileges; some FS lack links
        return False
    return True


def _make_symlink_to(link: Path, target: Path) -> bool:
    """`_make_symlink` for a target that may be a file.

    `target_is_directory` is only a hint, and asserting it about a regular file is how a
    test ends up skipped — or failing — for reasons that have nothing to do with whether
    the platform supports symlinks at all.
    """
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        return False
    return True


def _swap_directory_for_symlink(directory: Path, target: Path) -> bool:
    """Turn an existing directory into a symlink, modelling a swap after validation."""
    try:
        directory.rmdir()
        directory.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    return True


def test_a_symlink_cannot_route_the_workspace_into_the_package(tmp_path: Path) -> None:
    """The lexical path is outside the package; the resolved location is not."""
    link = tmp_path / "innocent-name"
    if not _make_symlink(link, PACKAGE_ROOT):
        pytest.skip("this filesystem does not allow creating directory symlinks")

    with pytest.raises(ValidationError, match=f"inside {PACKAGE_ROLE}"):
        Settings(workspace_dir=link)


def test_a_symlink_chain_cannot_reach_a_protected_root(tmp_path: Path) -> None:
    middle = tmp_path / "middle"
    if not _make_symlink(middle, PACKAGE_ROOT):
        pytest.skip("this filesystem does not allow creating directory symlinks")
    (tmp_path / "outer").mkdir()
    outer = tmp_path / "outer" / "deeper"
    if not _make_symlink(outer, middle):
        pytest.skip("this filesystem does not allow creating directory symlinks")

    with pytest.raises(ValidationError, match=f"inside {PACKAGE_ROLE}"):
        Settings(workspace_dir=outer)


def test_symlink_pointing_outside_the_project_is_accepted_and_resolved(tmp_path: Path) -> None:
    target = tmp_path / "real-workspace"
    target.mkdir()
    link = tmp_path / "link"
    if not _make_symlink(link, target):
        pytest.skip("this filesystem does not allow creating directory symlinks")

    settings = Settings(workspace_dir=link)

    assert settings.workspace_path == target.resolve()


# --------------------------------------------------------------------- E. environment input


def test_workspace_override_from_the_environment_is_validated(tmp_path: Path) -> None:
    settings = Settings.from_env({ENV_WORKSPACE_DIR: str(tmp_path / "override")})

    assert settings.workspace_dir == tmp_path / "override"


def test_environment_workspace_inside_the_checkout_is_rejected_with_a_clear_error() -> None:
    with pytest.raises(ConfigurationError, match=ENV_WORKSPACE_DIR):
        Settings.from_env({ENV_WORKSPACE_DIR: str(PACKAGE_ROOT)})


def test_from_env_reads_the_real_environment_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The production path: the CLI calls `from_env()` with no argument."""
    monkeypatch.setenv(ENV_WORKSPACE_DIR, str(tmp_path / "from-real-env"))
    monkeypatch.setenv(ENV_HTTP_TIMEOUT_SECONDS, "12")

    settings = Settings.from_env()

    assert settings.workspace_dir == tmp_path / "from-real-env"
    assert settings.http_timeout_seconds == 12.0


def test_a_real_environment_cannot_move_a_bare_settings_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`from_env()` is the only door `PACKPROOF_*` values come through.

    This matters for both test reliability and production safety: if a future change
    bound the environment inside the model, a developer's shell would silently alter
    every `Settings()`, and code that forgot to call `from_env()` would pick up
    whatever the operator's terminal happened to export.
    """
    home = tmp_path / "home"
    simulate_platform(monkeypatch, "linux", home)
    monkeypatch.setenv(ENV_WORKSPACE_DIR, str(tmp_path / "from-a-stray-shell"))
    monkeypatch.setenv(ENV_HTTP_TIMEOUT_SECONDS, "99")

    assert Settings().workspace_dir == home / ".cache" / APP_DIR_NAME / WORKSPACE_DIR_NAME
    assert Settings().http_timeout_seconds == DEFAULT_HTTP_TIMEOUT_SECONDS


# ------------------------------------------------------------------ F. ancestor manifests


@pytest.fixture
def simulate_package_dir(monkeypatch: pytest.MonkeyPatch):
    """Report the running package as living somewhere else, for the duration of one test.

    `_project_root()` is memoized, because its answer depends only on where the package
    was imported from. Patching `_package_dir` without dropping that cache would leave a
    test asserting the *real* checkout's answer while believing it was testing a
    simulation — a silent pass, which is the worst kind.
    """

    def adopt(package_dir: Path) -> Path:
        resolved = package_dir.resolve()
        monkeypatch.setattr(settings, "_package_dir", lambda: resolved)
        settings._project_root.cache_clear()
        return resolved

    yield adopt
    settings._project_root.cache_clear()


def build_installation(
    base: Path, *, manifest: str | None, repository: str | None, package: str = "src/packproof"
) -> Path:
    """Create an installation rooted at `base`, returning the package directory.

    `repository` selects the version-control marker: "directory" for a normal clone,
    "file" for a Git worktree or submodule, None for an install with no repository
    metadata at all — a wheel, or an unpacked release tarball.
    """
    package_dir = base / Path(package)
    package_dir.mkdir(parents=True)
    if manifest is not None:
        (base / "pyproject.toml").write_text(manifest, encoding="utf-8")
    if repository == "directory":
        (base / ".git").mkdir()
    elif repository == "file":
        (base / ".git").write_text("gitdir: /elsewhere/.git/worktrees/one\n", encoding="utf-8")
    return package_dir


PACKPROOF_MANIFEST = '[project]\nname = "packproof"\nversion = "0.1.0"\n'

#: Every manifest shape that is valid TOML, or unreadable, but is not a usable claim to
#: be this project. `.get("project", {}).get("name")` used to raise `AttributeError` on
#: the first three, which nothing in the module catches.
NON_MATCHING_MANIFESTS = [
    pytest.param('project = "unexpected"\n', id="project-is-a-string"),
    pytest.param("project = 42\n", id="project-is-a-number"),
    pytest.param('project = ["unexpected"]\n', id="project-is-a-list"),
    pytest.param("project = true\n", id="project-is-a-boolean"),
    pytest.param("[project]\n", id="project-table-with-no-name"),
    pytest.param("[project]\nname = 42\n", id="name-is-a-number"),
    pytest.param('[project]\nname = ["packproof"]\n', id="name-is-a-list"),
    pytest.param('[project]\nname = "totally-other"\n', id="unrelated-project"),
    pytest.param("[tool.ruff]\nline-length = 100\n", id="no-project-table"),
    pytest.param("this is not = = valid toml\n", id="unparseable-toml"),
    pytest.param("", id="empty-manifest"),
]


@pytest.mark.parametrize("manifest", NON_MATCHING_MANIFESTS)
def test_a_non_matching_ancestor_manifest_is_neither_adopted_nor_fatal(
    manifest: str, simulate_package_dir, tmp_path: Path
) -> None:
    """The regression this covers: a file PackProof does not own, in a directory it
    merely walks past, used to turn configuration into a traceback.

    Repository evidence is present on purpose, so the *only* reason these roots are not
    adopted is the manifest check. Without it the test would pass for the wrong reason.
    """
    base = tmp_path / "install"
    package = build_installation(base, manifest=manifest, repository="directory")
    simulate_package_dir(package)

    assert settings._project_root() != base.resolve()
    # Reaching here at all is the assertion: no `AttributeError`, no `TypeError`.
    assert package.resolve() in protected_roots()
    assert Settings().workspace_dir.name == WORKSPACE_DIR_NAME


def test_a_valid_packproof_manifest_does_identify_the_repository_root(
    simulate_package_dir, tmp_path: Path
) -> None:
    """The other side of the same rule — the checks must not reject the real thing.

    The layout built here is the one `pyproject.toml` builds: `src/packproof` directly
    under a directory carrying the manifest and the repository metadata. All three
    signals line up, so this is the case the identity rule exists to accept.
    """
    base = tmp_path / "checkout"
    package = build_installation(base, manifest=PACKPROOF_MANIFEST, repository="directory")
    simulate_package_dir(package)

    assert settings._project_root() == base.resolve()
    assert base.resolve() in protected_roots()
    with pytest.raises(ValidationError, match=f"inside {CHECKOUT_ROLE}"):
        Settings(workspace_dir=base / "tests")


# ------------------------------------------------------------------- G. repository identity


def test_a_manifest_without_repository_metadata_is_not_a_repository_root(
    simulate_package_dir, tmp_path: Path
) -> None:
    """A directory can hold a `pyproject.toml` naming `packproof` without being the
    checkout it came from — a name is a copyable string. Requiring version-control
    evidence is what stops an unrelated ancestor becoming a protected root."""
    base = tmp_path / "stray"
    package = build_installation(base, manifest=PACKPROOF_MANIFEST, repository=None)
    simulate_package_dir(package)

    assert settings._project_root() is None
    assert base.resolve() not in protected_roots()


def test_a_git_worktree_file_counts_as_repository_evidence(
    simulate_package_dir, tmp_path: Path
) -> None:
    """`.git` is a directory in a clone and a one-line file in a worktree or submodule,
    and a developer working in a worktree has the same code to protect."""
    base = tmp_path / "worktree"
    package = build_installation(base, manifest=PACKPROOF_MANIFEST, repository="file")
    simulate_package_dir(package)

    assert (base / ".git").is_file()
    assert settings._project_root() == base.resolve()


def test_an_unrelated_ancestor_manifest_with_metadata_is_still_not_adopted(
    simulate_package_dir, tmp_path: Path
) -> None:
    """Both signals present, but pointing at someone else's project."""
    base = tmp_path / "other-repo"
    package = build_installation(
        base, manifest='[project]\nname = "other"\n', repository="directory"
    )
    simulate_package_dir(package)

    assert settings._project_root() is None


def test_a_wheel_install_without_repository_metadata_stays_functional(
    simulate_package_dir, tmp_path: Path
) -> None:
    """No manifest and no `.git` anywhere in the ancestry: protection still covers the
    installed package and the environment, and a normal external workspace is accepted."""
    site_packages = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    package = build_installation(site_packages, manifest=None, repository=None, package="packproof")
    simulate_package_dir(package)

    assert settings._project_root() is None
    assert package.resolve() in protected_roots()

    external = tmp_path / "work" / "scans"
    assert Settings(workspace_dir=external).workspace_dir == external


def test_a_false_repository_root_cannot_invalidate_the_default_workspace(
    simulate_package_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concrete failure the identity rule prevents — with *both* copyable signals faked.

    A manifest naming `packproof` dropped in a home directory used to promote that home
    to a protected root. The per-user default workspace is always beneath the home
    directory, so `Settings()` then refused its *own default* — a tool failing with
    nothing configured by the operator, which is indistinguishable from a broken install.

    Repository metadata is present here on purpose. The two-signal policy this replaces was
    satisfied by exactly this layout, so a test that left `.git` out would pass against the
    defective rule and prove nothing about the one that fixed it. What refuses the claim now
    is that the running package sits at `.venv/lib/…/packproof` rather than at the
    candidate's `src/packproof` — and the protection that matters, the package directory and
    the interpreter's own roots, is unaffected by the refusal.
    """
    home = tmp_path / "home"
    package = build_installation(
        home,
        manifest=PACKPROOF_MANIFEST,
        repository="directory",
        package=".venv/lib/python3.13/site-packages/packproof",
    )
    site_packages = package.parents[1]
    simulate_package_dir(package)
    monkeypatch.setattr(
        sysconfig,
        "get_paths",
        lambda: {"purelib": str(site_packages), "platlib": str(site_packages)},
    )
    simulate_platform(monkeypatch, "linux", home)

    assert settings._project_root() is None
    assert home.resolve() not in protected_roots()
    assert package.resolve() in protected_roots()
    assert site_packages.resolve() in protected_roots()
    assert Settings().workspace_dir == home / ".cache" / APP_DIR_NAME / WORKSPACE_DIR_NAME


@pytest.mark.parametrize(
    "package,adopted",
    [
        pytest.param("src/packproof", True, id="src-layout-checkout"),
        pytest.param("packproof", False, id="flat-layout-tree"),
        pytest.param(
            "venv/lib/python3.13/site-packages/packproof",
            False,
            id="wheel-below-the-ancestor",
        ),
    ],
)
def test_the_package_location_decides_when_manifest_and_metadata_already_agree(
    package: str, adopted: bool, simulate_package_dir, tmp_path: Path
) -> None:
    """F-20: the third signal is the one that separates a checkout from a host directory.

    Every case below has a `pyproject.toml` naming `packproof` *and* a `.git` directory in
    `base`, so the two signals the previous policy required are satisfied everywhere and
    cannot decide between them. Only the location of the code actually running differs:
    `base/src/packproof` is the layout `pyproject.toml` builds, and is a checkout;
    `base/packproof` is a flat tree this project does not build, and
    `base/venv/…/site-packages/packproof` is an installed wheel that merely happens to
    live underneath.

    The flat case is deliberately *not* adopted rather than supported speculatively. Adding
    it would widen every future claim about "the repository", and if the project ever moves
    to a flat layout the consequence of leaving it out is that the checkout is protected by
    the package-directory rule alone — a narrower refusal, never an unsafe one.
    """
    base = tmp_path / "claimed-root"
    built = build_installation(
        base, manifest=PACKPROOF_MANIFEST, repository="directory", package=package
    )
    package_dir = simulate_package_dir(built)

    if adopted:
        assert settings._project_root() == base.resolve()
    else:
        assert settings._project_root() is None
        assert base.resolve() not in protected_roots()
        # Refusing the claim must not cost the operator the directory: it stays usable.
        assert Settings(workspace_dir=base / "scans").workspace_dir == base / "scans"

    assert package_dir in protected_roots()


def test_the_real_checkout_is_identified_from_the_running_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-20 requirements A and H, with nothing simulated.

    An editable installation from a checkout is the layout a developer actually runs, and
    the one where dropping the repository root would leave the source tree unprotected
    beside the interpreter's own environment. The previous test suite reached this case
    only through the module-level `PROJECT_ROOT`, so the identity claim itself — three
    signals agreeing on *this* directory — was never asserted unpatched.
    """
    settings._project_root.cache_clear()  # no fixture patched this one; drop any stale memo
    assert settings._project_root() == PROJECT_ROOT
    assert PROJECT_ROOT / "src" / "packproof" == PACKAGE_ROOT
    assert (PROJECT_ROOT / ".git").exists()

    # The checkout stays refused, and is named as what it is (W-04).
    with pytest.raises(ValidationError, match=f"inside {re.escape(CHECKOUT_ROLE)}") as excinfo:
        Settings(workspace_dir=PROJECT_ROOT / "tests")
    assert str((PROJECT_ROOT / "tests").resolve()) in str(excinfo.value)


def test_an_unadopted_ancestor_is_never_named_in_a_refusal(
    simulate_package_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W-04: a message can only describe a root that was actually adopted.

    The misleading wording this replaces came from the F-20 defect — a falsely adopted
    ancestor was reported as PackProof's own files, so an operator was told that a
    directory holding their virtualenv belonged to the tool. Once the claim is refused, the
    directory cannot appear in a message at all, and this asserts that rather than trusting
    it not to.
    """
    base = tmp_path / "false-ancestor"
    built = build_installation(
        base,
        manifest=PACKPROOF_MANIFEST,
        repository="directory",
        package="venv/lib/python3.13/site-packages/packproof",
    )
    site_packages = built.parents[1]
    simulate_package_dir(built)
    monkeypatch.setattr(
        sysconfig,
        "get_paths",
        lambda: {"purelib": str(site_packages), "platlib": str(site_packages)},
    )

    with pytest.raises(ValidationError, match=f"inside {re.escape(ENVIRONMENT_ROLE)}") as excinfo:
        Settings(workspace_dir=site_packages / "leftover-scan")

    message = str(excinfo.value)
    # The root named in the message, captured rather than substring-searched: the ancestor
    # is a *prefix* of the environment path, so `"base" not in message` would fail for a
    # reason that has nothing to do with what this test is about.
    named = re.search(r"at (\S+?)\. Untrusted package content", message)
    assert named is not None, f"no protected root named in {message!r}"
    assert Path(named.group(1)) == site_packages.resolve()
    assert CHECKOUT_ROLE not in message, "a refused checkout claim must not reach the message"
    assert PACKAGE_ROLE not in message


# ------------------------------------------------------- H. interpreter package roots (F-10)


def environment_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *names: str
) -> tuple[Path, ...]:
    """Point `sysconfig` at one or more package roots under `tmp_path`.

    Two names model a distribution that separates pure and platform library dirs; one
    models the common case where they are the same directory.
    """
    roots = tuple((tmp_path / "env" / "lib" / "python3.13" / name).resolve() for name in names)
    schemes = {"purelib": str(roots[0]), "platlib": str(roots[-1])}
    monkeypatch.setattr(sysconfig, "get_paths", lambda: schemes)
    return roots


def test_an_editable_install_protects_the_environment_it_imports_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap this closes: an editable install keeps `packproof` in the checkout, so the
    package's own parent directory is *not* the environment — and protection used to be
    keyed to exactly that parent. Meanwhile `sys.path` still points at a `site-packages`
    that had no guard at all.

    The policy is now a property of the interpreter, not of where the source happens to
    live, which is what makes these assertions hold without PackProof being inside them.
    """
    (purelib,) = environment_at(tmp_path, monkeypatch, "site-packages")

    assert purelib in protected_roots()
    assert purelib in _interpreter_package_roots()

    for shadow in (purelib, purelib / "packproof", purelib / "pydantic", purelib / "leftover"):
        with pytest.raises(ValidationError, match=f"inside {ENVIRONMENT_ROLE}"):
            Settings(workspace_dir=shadow)

    with pytest.raises(ValidationError, match=f"contains {ENVIRONMENT_ROLE}"):
        Settings(workspace_dir=purelib.parent)


def test_separate_purelib_and_platlib_are_both_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A distribution can place pure and platform libraries in different directories;
    both are on the import path, so both need the same refusal."""
    purelib, platlib = environment_at(tmp_path, monkeypatch, "site-packages", "plat")

    assert purelib != platlib
    assert {purelib, platlib}.issubset(set(protected_roots()))
    for location in (purelib / "pkg", platlib / "pkg"):
        with pytest.raises(ValidationError, match=f"inside {ENVIRONMENT_ROLE}"):
            Settings(workspace_dir=location)


def test_duplicate_interpreter_paths_are_protected_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Most installs report the same directory twice; the root list is a set of refusals
    and a duplicate would make `protected_roots()` describe a policy it does not have."""
    (purelib,) = environment_at(tmp_path, monkeypatch, "site-packages")

    roots = protected_roots()
    assert roots.count(purelib) == 1
    assert len(roots) == len(set(roots))


def test_a_relative_or_absent_interpreter_scheme_adds_no_protected_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheme that names nothing usable protects nothing — and must not be a crash.

    Dropping it cannot widen the boundary: the package directory and any identified
    repository root are still refused, so the worst outcome is one fewer refusal rather
    than an unhandled exception at configuration time.
    """
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda: {"purelib": "relative/site-packages", "platlib": ""}
    )

    assert _interpreter_package_roots() == ()
    assert protected_roots() == (PACKAGE_ROOT, PROJECT_ROOT)
    assert Settings(workspace_dir=tmp_path / "scan").workspace_dir == tmp_path / "scan"


def test_an_editable_style_install_in_this_checkout_still_protects_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three sources at once: the running package, the checkout it lives in, and the
    environment the interpreter names — none of the three conditional on the others.

    The refusals also name which of the three was reached (W-04). This is the case that
    matters most for wording: all three roots are live at once here, and describing the
    interpreter's directory as PackProof's own file is exactly the diagnostic that sends
    an operator looking in the wrong place.
    """
    (purelib,) = environment_at(tmp_path, monkeypatch, "site-packages")

    assert protected_roots() == (PACKAGE_ROOT, PROJECT_ROOT, purelib)
    for unsafe, role in (
        (PACKAGE_ROOT / "cli", PACKAGE_ROLE),
        (PROJECT_ROOT / "tests", CHECKOUT_ROLE),
        (purelib / "packproof", ENVIRONMENT_ROLE),
    ):
        with pytest.raises(ValidationError, match=f"inside {re.escape(role)}") as excinfo:
            Settings(workspace_dir=unsafe)
        assert str(unsafe.resolve()) in str(excinfo.value)


# ------------------------------------------------------------ I. unexpandable home paths (F-09)


# A username no account database is likely to define; `~name` is looked up, not guessed.
UNKNOWN_USER = "~b0bn0nexistentuser/ws"


def test_an_unknown_user_home_prefix_raises_a_configuration_error(tmp_path: Path) -> None:
    """`Path.expanduser()` answers a missing account with `RuntimeError`, and pydantic
    converts `ValueError` — not `RuntimeError` — into `ValidationError`. Left unhandled,
    that exception escaped `Settings()` and `from_env()` as a traceback that named no
    configuration variable, breaking the contract `ConfigurationError` exists to keep.
    """
    with pytest.raises(ValidationError, match="home directory that cannot be resolved") as excinfo:
        Settings(workspace_dir=UNKNOWN_USER)

    assert ENV_WORKSPACE_DIR in str(excinfo.value)


def test_a_tilde_followed_by_a_space_is_not_silently_a_relative_name() -> None:
    """The spelling nobody intends: the "user" here is a single space, and the value must
    still be refused rather than turned into a directory named `~`."""
    with pytest.raises(ValidationError, match="home directory that cannot be resolved"):
        Settings(workspace_dir="~ /x")


def test_the_expansion_error_keeps_the_original_as_its_cause(tmp_path: Path) -> None:
    """`raise ... from` is what makes a reportable refusal diagnosable.

    The chaining is asserted at `_validate_workspace()` because that is where it is
    created: pydantic's core builds its `ValidationError` from the message alone, so the
    exception *object* never reaches the outer error even though its text does — a
    detail worth knowing before anyone writes a handler that expects `__cause__`.
    """
    with pytest.raises(ConfigurationError) as excinfo:
        _validate_workspace(Path(UNKNOWN_USER))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert ENV_WORKSPACE_DIR in str(excinfo.value)


def test_from_env_reports_an_unexpandable_home_as_configuration_error() -> None:
    """The production path, and the one that must not fall back to the default workspace:
    a configured value this operator cannot use is an error, not a hint."""
    with pytest.raises(ConfigurationError, match=ENV_WORKSPACE_DIR) as excinfo:
        Settings.from_env({ENV_WORKSPACE_DIR: UNKNOWN_USER})

    message = str(excinfo.value)
    assert "home directory that cannot be resolved" in message
    assert isinstance(excinfo.value.__cause__, ValidationError), "not a bare string error"


# ---------------------------------------------- J. a workspace that exists as the wrong thing


def test_an_existing_regular_file_is_rejected_as_a_workspace(tmp_path: Path) -> None:
    """A file is not a directory, and `mkdir` is not going to fix that.

    The failure used to arrive at extraction time as `FileExistsError` or
    `NotADirectoryError`, which is the worst possible moment for a security tool to
    discover its output location is unusable — after untrusted bytes have been fetched.
    """
    regular = tmp_path / "scan-workspace.txt"
    regular.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValidationError, match="exists and is not a directory"):
        Settings(workspace_dir=regular)


def test_the_check_uses_the_filesystem_type_and_not_the_name(tmp_path: Path) -> None:
    """Extensions are a naming convention, not a filesystem fact.

    A directory called `x.txt` is a perfectly good workspace, and a file called `scan`
    with no extension at all is not. Deciding on the suffix would reject the first and
    accept the second — the opposite of correct.
    """
    as_a_directory = tmp_path / "x.txt"
    as_a_directory.mkdir()
    assert Settings(workspace_dir=as_a_directory).workspace_dir == as_a_directory

    as_a_file = tmp_path / "scan-workspace"
    as_a_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValidationError, match="exists and is not a directory"):
        Settings(workspace_dir=as_a_file)


def test_a_symlink_to_a_regular_file_is_rejected_on_its_resolved_target(tmp_path: Path) -> None:
    """The typed spelling looks like a workspace path; the location it reaches is a file.

    The decision is taken after resolution, which is the only point at which it can be
    taken correctly.
    """
    regular = tmp_path / "payload.bin"
    regular.write_text("x", encoding="utf-8")
    link = tmp_path / "looks-like-a-directory"
    if not _make_symlink_to(link, regular):
        pytest.skip("this filesystem does not allow creating symlinks")

    with pytest.raises(ValidationError, match="exists and is not a directory"):
        Settings(workspace_dir=link)


def test_a_dangling_symlink_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    """A link to nothing reports itself as absent, so `exists()` alone would wave it
    through — and the extraction that followed would fail on the entry that is there."""
    link = tmp_path / "points-nowhere"
    try:
        link.symlink_to(tmp_path / "never-created")
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not allow creating symlinks")

    assert not link.exists() and link.is_symlink()
    with pytest.raises(ValidationError, match="symlink to a location that does not exist"):
        Settings(workspace_dir=link)


def test_an_existing_directory_and_an_absent_one_are_both_accepted(tmp_path: Path) -> None:
    """Rejecting the wrong thing costs the operator a working configuration.

    An external directory already in use is legitimate — it may hold a previous scan —
    and a path that does not exist yet must stay valid, because validation is not the
    step that creates it.
    """
    existing = tmp_path / "previous-scans"
    existing.mkdir()
    absent = tmp_path / "first" / "second" / "scan-workspace"

    assert Settings(workspace_dir=existing).workspace_path == existing.resolve()
    assert Settings(workspace_dir=absent).workspace_dir == absent
    assert not absent.exists(), "validation must not create the workspace"


# ------------------------------------------------- K. what the two workspace accessors mean


def test_workspace_dir_keeps_the_configuration_and_workspace_path_the_location(
    tmp_path: Path,
) -> None:
    """Two accessors, two questions, deliberately different answers.

    `workspace_dir` answers "what did the operator configure", which is what an error
    message and a report have to show; `workspace_path` answers "where is this on the
    filesystem", which is what a containment check has to compare against. A symlinked
    workspace is exactly where they part company, and this is the case a future reader
    needs spelled out — the resolved path was checked at construction, and nothing
    rechecks it later.
    """
    target = tmp_path / "real-workspace"
    target.mkdir()
    link = tmp_path / "configured-name"
    if not _make_symlink_to(link, target):
        pytest.skip("this filesystem does not allow creating directory symlinks")

    settings = Settings(workspace_dir=link)

    assert settings.workspace_dir == link
    assert settings.workspace_path == target.resolve()
    assert settings.workspace_dir != settings.workspace_path


def test_workspace_path_reevaluates_the_filesystem_on_every_access(tmp_path: Path) -> None:
    """The limit configuration validation cannot cross, stated as an executable fact.

    A symlink swapped in after construction changes what `workspace_path` returns with no
    revalidation of any kind. Recording this as a test is not a fix — it is the reason the
    extractor must resolve and check each member destination itself, at the moment it
    writes.
    """
    leaf = tmp_path / "scan-workspace"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    leaf.mkdir()

    settings = Settings(workspace_dir=leaf)
    assert settings.workspace_path == leaf.resolve()

    if not _swap_directory_for_symlink(leaf, outside):
        pytest.skip("this filesystem does not allow replacing a directory with a symlink")

    assert settings.workspace_path == outside.resolve()
    assert settings.workspace_dir == leaf, "the configured spelling is kept, not re-checked"


# ----------------------------------------------------------- L. the error contract end to end


def test_configuration_error_is_a_value_error_by_necessity() -> None:
    """Not a stylistic choice: this subclassing is the whole reason a refusal is reachable.

    A field validator's `ValueError` is what pydantic converts into `ValidationError`,
    which `from_env()` then converts into the operator-facing `ConfigurationError`. Re-base
    that class on `Exception` and every workspace refusal turns into an unhandled traceback
    — the exact shape of the `expanduser()` defect this milestone closed, arriving from the
    other direction.
    """
    assert issubclass(ConfigurationError, ValueError)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda root: str(root / "with" / ".." / "back"), id="parent-component"),
        pytest.param(lambda root: ".", id="current-directory"),
        pytest.param(lambda root: "   ", id="whitespace-only"),
        pytest.param(lambda root: UNKNOWN_USER, id="unexpandable-home"),
        pytest.param(lambda root: str(root / "afile.txt"), id="existing-regular-file"),
        pytest.param(lambda root: str(root / "dangling"), id="dangling-symlink"),
        pytest.param(lambda root: str(PACKAGE_ROOT), id="inside-the-package"),
        pytest.param(lambda root: "/", id="filesystem-root"),
    ],
)
def test_every_workspace_refusal_reaches_from_env_as_a_configuration_error(
    build, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One door, one error type, whatever the reason for the refusal.

    Each of these values fails a different check, and an operator who sets one of them gets
    the same class of error naming `PACKPROOF_WORKSPACE_DIR` — no traceback, and no silent
    fall back to the default workspace. The filesystem-dependent cases are created here
    rather than imagined, so a refusal that only fires against a real entry cannot hide.
    """
    (tmp_path / "afile.txt").write_text("x", encoding="utf-8")
    (tmp_path / "dangling").symlink_to(tmp_path / "never-created")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError, match=ENV_WORKSPACE_DIR) as excinfo:
        Settings.from_env({ENV_WORKSPACE_DIR: build(tmp_path)})

    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["afile.txt", "dangling"], (
        "a refusal must not create anything"
    )


# ------------------------------------------------ M. filesystem inspection failures (F-19)


def _raise_as(exc: BaseException):
    """A replacement callable that fails the way the filesystem would.

    Used only where a real kernel condition cannot be provoked portably, and only against
    the call that raises it, so the handler under test is exercised rather than avoided.
    """

    def raiser(*args: object, **kwargs: object) -> None:
        raise exc

    return raiser


def _directory_search_is_enforced(root: Path) -> bool:
    """Whether removing the search bit from a directory really denies inspection to us.

    Checked rather than assumed: the mode bits are POSIX-only, the super-user is exempt, and
    a filesystem mounted with permissions ignored answers lookups anyway. A test that skipped
    none of that would report a permission refusal it never observed.
    """
    probe = root / "search-probe"
    probe.mkdir()
    with _unsearchable(probe):
        try:
            (probe / "inside").exists()
        except OSError:
            return True
    return False


@contextmanager
def _unsearchable(path: Path) -> Iterator[Path]:
    """Take the search bit off `path`, then put back exactly the mode it had.

    Restoring the recorded mode rather than writing 0o755 keeps a test's teardown from
    silently changing a directory's permissions, and a private temporary tree does not need
    the group and world bits to be readable to be usable.
    """
    mode = stat.S_IMODE(path.lstat().st_mode)
    os.chmod(path, 0o000)
    try:
        yield path
    finally:
        os.chmod(path, mode)


@pytest.fixture
def unreadable_directory(tmp_path: Path) -> Iterator[Path]:
    """`tmp_path/locked`, with the search bit removed and restored whatever the test does.

    The restore is a `finally` rather than a last line because a leftover 000 directory makes
    pytest's own `tmp_path` teardown raise, and an error in teardown would hide whether the
    validation itself behaved.
    """
    locked = tmp_path / "locked"
    (locked / "inner").mkdir(parents=True)
    if not _directory_search_is_enforced(tmp_path):
        pytest.skip("this platform or user is not denied directory search")

    with _unsearchable(locked):
        yield locked


def test_an_unreadable_ancestor_becomes_a_configuration_error(
    unreadable_directory: Path,
) -> None:
    """F-19/T-02 A, against a real kernel refusal.

    `Path.is_symlink()`, `exists()` and `is_dir()` re-raise `EACCES` rather than reporting
    `False` — unlike `resolve()`, which walks past an unreadable component and answers for
    the path anyway, so the leak is *after* normalization and not visible in it. Left
    unhandled, `OSError` is not converted by pydantic the way a `ValueError` is: it escaped
    `Settings()`, escaped `from_env()`'s handler, and reached the operator as a traceback
    naming no configuration variable.
    """
    workspace = unreadable_directory / "inner" / "scan-workspace"

    with pytest.raises(ConfigurationError) as excinfo:
        _validate_workspace(workspace)

    assert isinstance(excinfo.value.__cause__, PermissionError), "the OS error is preserved"
    message = str(excinfo.value)
    assert "workspace_dir" in message
    assert ENV_WORKSPACE_DIR in message
    assert "Permission denied" in message

    # The same refusal through both public doors, and no silent second choice of directory.
    with pytest.raises(ValidationError):
        Settings(workspace_dir=workspace)
    with pytest.raises(ConfigurationError, match=ENV_WORKSPACE_DIR):
        Settings.from_env({ENV_WORKSPACE_DIR: str(workspace)})


def test_the_default_workspace_beneath_an_unreadable_home_is_refused(
    unreadable_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-19/T-02 B: the case that needs no configuration at all, which is why it matters.

    The report this came from pointed `HOME` at a directory the process could not enter. The
    default workspace is derived from `HOME`, so the failure arrives while computing the
    fallback — the one value an operator cannot avoid by not setting anything.
    """
    simulate_platform(monkeypatch, "linux", unreadable_directory)

    with pytest.raises(ConfigurationError) as excinfo:
        Settings.from_env({})

    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert "Permission denied" in str(excinfo.value)
    assert ENV_WORKSPACE_DIR in str(excinfo.value)


def test_an_unreadable_repository_root_is_a_configuration_error(
    simulate_package_dir, tmp_path: Path
) -> None:
    """F-19: `protected_roots()` inspects the filesystem too, from inside the same validator.

    An ancestor that was traversable when the package was imported can be unreadable by the
    time a workspace is checked, and the manifest probe and the `.git` probe then raise. The
    only safe reading of a boundary that cannot be listed is "refuse": accepting the path
    because protection could not be computed would widen the boundary at exactly the moment
    it stopped being verifiable.
    """
    base = tmp_path / "checkout"
    package = build_installation(base, manifest=PACKPROOF_MANIFEST, repository="directory")
    simulate_package_dir(package)
    if not _directory_search_is_enforced(tmp_path):
        pytest.skip("this platform or user is not denied directory search")

    with _unsearchable(base):
        with pytest.raises(ConfigurationError) as excinfo:
            _validate_workspace(base / "scans")

        assert "protected locations could not be listed" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, PermissionError)


def test_a_relative_path_in_a_deleted_working_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-19: `Path.absolute()` is a filesystem call, and the working directory can vanish.

    `os.getcwd()` raises `FileNotFoundError` once the directory the process stood in has been
    deleted, which a relative `PACKPROOF_WORKSPACE_DIR` reaches on its way to becoming
    absolute. Deleted here rather than simulated: it is a real kernel condition on every
    platform that lets the deletion happen.
    """
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    try:
        shutil.rmtree(gone)
    except OSError:
        pytest.skip("this platform will not delete the directory the process is standing in")

    with pytest.raises(ConfigurationError) as excinfo:
        _validate_workspace(Path("scan-workspace"))

    assert "the path could not be located" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_an_overlong_workspace_path_is_refused_rather_than_raising(tmp_path: Path) -> None:
    """F-19/T-02 C, where the platform cooperates: `ENAMETOOLONG` is a bare `OSError`.

    `Path`'s own error filtering swallows `ENOENT`, `ENOTDIR`, `EBADF` and `ELOOP`, and this
    is none of them, so the probes raise. Skips, and says so, on any filesystem that answers
    quietly — the simulated case below is what always runs.
    """
    overlong = tmp_path / ("scan-" + "x" * 4000) / "workspace"
    try:
        overlong.is_symlink()
    except OSError:
        pass
    else:
        pytest.skip("this filesystem raises nothing for a name this long")

    with pytest.raises(ValidationError) as excinfo:
        Settings(workspace_dir=str(overlong))

    message = str(excinfo.value)
    assert "workspace_dir" in message
    assert ENV_WORKSPACE_DIR in message


def test_a_simulated_name_too_long_is_refused_at_the_inspection_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-19/T-02 C, *simulated*: a test of the handler, not of a platform.

    The real case above can only run where the kernel raises, so this one always runs and
    proves the narrower point — an `ENAMETOOLONG` `OSError` reaching the inspection boundary
    is converted rather than propagated. It is labelled as a simulation so nobody reads it as
    evidence about Windows, Linux or macOS.
    """
    monkeypatch.setattr(
        Path, "is_symlink", _raise_as(OSError(errno.ENAMETOOLONG, "File name too long"))
    )

    with pytest.raises(ConfigurationError) as excinfo:
        _validate_workspace(tmp_path / "workspace")

    assert isinstance(excinfo.value.__cause__, OSError)
    assert "File name too long" in str(excinfo.value)


def test_the_filesystem_handlers_do_not_swallow_other_kinds_of_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-19 requirement 8: `except OSError` is the whole handler, and nothing wider.

    A `TypeError` from the same call is a programming error, not a filesystem condition.
    Turning it into a message about the operator's configuration would hide a bug in
    PackProof behind an instruction to change a setting.
    """
    monkeypatch.setattr(Path, "is_symlink", _raise_as(TypeError("not a filesystem condition")))

    with pytest.raises(TypeError):
        _validate_workspace(tmp_path / "workspace")


def test_the_inspection_boundaries_leave_every_other_decision_untouched(
    tmp_path: Path,
) -> None:
    """F-19/T-02 D and E: handlers wrap reads; they moved no rule and softened none.

    The rewrite evaluates `is_link`, `exists` and `is_directory` together, before deciding,
    so that one unreadable ancestor fails once instead of halfway through a short-circuit.
    That is the change this test guards: every value below is refused or accepted for the same
    reason it always was, in the same order — a dangling link before the wrong-type message,
    containment advice before either, and a plain external directory still accepted.
    """
    existing = tmp_path / "previous-scans"
    existing.mkdir()
    regular = tmp_path / "afile.txt"
    regular.write_text("x", encoding="utf-8")

    link = tmp_path / "dangling"
    links_supported = True
    try:
        link.symlink_to(tmp_path / "never-created")
    except (OSError, NotImplementedError):
        links_supported = False

    refusals = [
        ("..", r"must not contain '\.\.' components"),
        (".", "not the current directory"),
        ("   ", "empty or whitespace-only"),
        (UNKNOWN_USER, "home directory that cannot be resolved"),
        (str(regular), "exists and is not a directory"),
        (str(PACKAGE_ROOT), re.escape(f"inside {PACKAGE_ROLE}")),
        (str(PROJECT_ROOT.parent), re.escape(f"contains {PACKAGE_ROLE}")),
    ]
    if links_supported:
        refusals.insert(5, (str(link), "symlink to a location that does not exist"))

    for value, phrase in refusals:
        with pytest.raises(ValidationError, match=phrase):
            Settings(workspace_dir=value)

    assert Settings(workspace_dir=existing).workspace_path == existing.resolve()
    absent = tmp_path / "does" / "not" / "exist"
    assert Settings(workspace_dir=absent).workspace_dir == absent
    assert not absent.exists(), "validation must still not create anything"


# ------------------- N. protected-root roles and diagnostic exception safety (F-21)


def _answer_once_then_fail(real, exc: OSError):
    """Let `real` answer its first call, then fail like a path that turned unreadable.

    Discovery is entitled to consult both sources once while listing the protected roots;
    what must not happen is a *second* consultation while a refusal is being written. This
    draws that line by call count rather than by reading the implementation, and the memo
    in `_project_root()` is warmed first so exactly one call per source belongs to discovery.
    """
    seen = 0

    def wrapper():
        nonlocal seen
        seen += 1
        if seen > 1:
            raise exc
        return real()

    return wrapper


def test_a_refusal_is_written_without_another_filesystem_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-21: naming the root in a message used to cost a filesystem walk.

    `_root_role()` decided what a protected root *was* by calling `_package_dir()` — which
    is `Path(__file__).resolve()`, a real walk, uncached — and `_project_root()`. It ran
    from inside the containment loop, after every `try` in `_validate_workspace()` had
    already closed. F-19's rule was that no filesystem operation in this validator is
    unguarded, and the diagnostic path broke it: a second failure there would escape as a
    raw `OSError` from the very code meant to explain the first one, and it could also
    report a role derived from a later look at the filesystem than the one that produced the
    refusal.

    The `OSError` below is a labelled simulation, not a reproduction: measured on this
    machine, `resolve(strict=False)` tolerates both a symlink loop and a 300-character
    component, so no kernel condition could be provoked against that particular path. What
    is asserted is the property that makes the question moot — a refusal performs no
    discovery call at all. Removing the attached roles brings the second call back, and this
    test then fails with a `PermissionError` escaping `_validate_workspace()`.
    """
    settings._project_root.cache_clear()
    assert _protected_roots_with_roles(), "warm the memo so discovery is the only real caller"

    denial = PermissionError(errno.EACCES, "Permission denied")
    monkeypatch.setattr(
        settings, "_package_dir", _answer_once_then_fail(settings._package_dir, denial)
    )
    monkeypatch.setattr(
        settings, "_project_root", _answer_once_then_fail(settings._project_root, denial)
    )

    with pytest.raises(ConfigurationError) as excinfo:
        _validate_workspace(PACKAGE_ROOT / "cli")

    assert f"inside {PACKAGE_ROLE}" in str(excinfo.value), "the role still has to be right"
    assert excinfo.value.__cause__ is None, (
        "this is a containment refusal; an inspection failure would mean a fresh call leaked"
    )


def test_each_protected_source_arrives_with_the_role_that_explains_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W-04's guarantee, stated where it is actually decided: at discovery.

    Three live sources at once — this checkout's package directory, the checkout itself,
    and the interpreter's package root — each paired with the role named for it, in the
    documented order. Assigning the role from the discovery step that produced the path
    removes the guess the old comparison made, and the same role is what reaches the
    operator, so the pair list and the refusal text cannot drift apart.
    """
    (purelib,) = environment_at(tmp_path, monkeypatch, "site-packages")

    assert _protected_roots_with_roles() == (
        (PACKAGE_ROOT, PACKAGE_ROLE),
        (PROJECT_ROOT, CHECKOUT_ROLE),
        (purelib, ENVIRONMENT_ROLE),
    )

    for location, role in (
        (PACKAGE_ROOT / "cli", PACKAGE_ROLE),
        (PROJECT_ROOT / "tests", CHECKOUT_ROLE),
        (purelib / "packproof", ENVIRONMENT_ROLE),
    ):
        with pytest.raises(ValidationError, match=f"inside {role}"):
            Settings(workspace_dir=location)


def test_a_wheel_installation_reports_no_checkout_role_at_all(
    simulate_package_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-21 must not reintroduce F-20: with no repository evidence, the checkout role is
    simply absent from the list, and the two remaining roots keep the roles that make them
    actionable. A wheel user told their `site-packages` is "PackProof's own" would go looking
    in the wrong tree; a wheel user whose home was adopted as a checkout would get no workspace
    at all.
    """
    site_packages = (tmp_path / "env" / "lib" / "python3.13" / "site-packages").resolve()
    environment_at(tmp_path, monkeypatch, "site-packages")
    package = simulate_package_dir(site_packages / "packproof")

    assert _protected_roots_with_roles() == (
        (package, PACKAGE_ROLE),
        (site_packages, ENVIRONMENT_ROLE),
    )


def test_a_directory_discovered_twice_keeps_the_first_and_most_specific_role(
    simulate_package_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deduplication now has a second job: choosing which role wins.

    The old `dict.fromkeys()` collapsed a repeated path silently, and `_root_role()` would
    have answered "package" for it anyway because that comparison came first. Carrying roles
    makes that precedence an explicit part of the list, so a directory that is both PackProof's
    package and an interpreter root is described by the narrower, more actionable claim — and
    still appears exactly once, which is what the refusal set means.
    """
    package = simulate_package_dir(tmp_path / "env" / "lib" / "python3.13" / "site-packages")
    (root,) = environment_at(tmp_path, monkeypatch, "site-packages")
    assert package == root, "the fixture must really produce one path from two sources"

    pairs = _protected_roots_with_roles()

    assert [role for path, role in pairs if path == root] == [PACKAGE_ROLE]
    assert len(pairs) == len({path for path, _ in pairs}), "no root may be listed twice"


def test_duplicate_interpreter_roots_appear_once_with_their_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary install: `sysconfig` reports the same directory for `purelib` and
    `platlib`, so the environment contributes one entry, and it carries the environment's
    role rather than inheriting whichever role happened to be attached last."""
    (purelib,) = environment_at(tmp_path, monkeypatch, "site-packages")

    pairs = _protected_roots_with_roles()

    assert [path for path, _ in pairs].count(purelib) == 1
    assert dict(pairs)[purelib] == ENVIRONMENT_ROLE


def test_protected_roots_is_still_the_path_only_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compatibility requirement, tested rather than asserted in prose.

    `_protected_roots_with_roles()` is private and additive; `protected_roots()` keeps its
    documented `tuple[Path, ...]`, its order, and its deduplication, so every existing caller
    and every existing assertion about membership and equality still describes the same
    object. If a future change slips a role tuple into this return value, the type check
    fails here instead of in whatever code unpacked it.
    """
    (purelib,) = environment_at(tmp_path, monkeypatch, "site-packages")

    roots = protected_roots()

    assert isinstance(roots, tuple)
    assert all(isinstance(root, Path) for root in roots), "no role may leak into this API"
    assert roots == tuple(path for path, _ in _protected_roots_with_roles())
    assert roots == (PACKAGE_ROOT, PROJECT_ROOT, purelib)


# ------------------------------------- O. which knob the operator should turn (F-22)


def test_a_workspace_that_cannot_be_inspected_blames_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-22 case A: the unreadable thing *is* the configured value, so repointing it is real
    advice and must stay in the message.

    Simulated against the call that fails, so the handler is exercised rather than avoided;
    the same wording is produced by the real `chmod 000` tests in section M, which run for
    real on this machine.
    """
    workspace = tmp_path / "scan-workspace"
    monkeypatch.setattr(
        Path, "is_symlink", _raise_as(PermissionError(errno.EACCES, "Permission denied"))
    )

    with pytest.raises(ConfigurationError) as excinfo:
        _validate_workspace(workspace)

    message = str(excinfo.value)
    assert "workspace_dir" in message
    assert ENV_WORKSPACE_DIR in message
    assert "Permission denied" in message
    assert f"set {ENV_WORKSPACE_DIR} to a directory you can read" in message
    assert "what is already there could not be inspected" in message
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_an_unreadable_boundary_does_not_tell_the_operator_to_repoint_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-22 case B: the failure is PackProof's own, and the old shared template said otherwise.

    Both inspection failures used to come from one builder, whose closing sentence always
    told the operator to set `PACKPROOF_WORKSPACE_DIR` to a readable directory. When
    protected-root *discovery* is what failed, that is wrong twice over: the workspace was
    never the unreadable path, and a new value is validated against the same unreadable
    boundary, so the operator is left changing a setting that changes nothing — the classic
    way to make a working configuration look like the cause of a failure it did not cause.

    Simulated by replacing one discovery source, which is where the real case comes from
    (`_interpreter_package_roots` reads `sysconfig`, and the manifest and `.git` probes read
    the ancestor tree; section M drives a real kernel refusal for that).
    """
    monkeypatch.setattr(
        settings, "_interpreter_package_roots", _raise_as(PermissionError(errno.EACCES, "denied"))
    )

    with pytest.raises(ConfigurationError) as excinfo:
        _validate_workspace(tmp_path / "scan-workspace")

    message = str(excinfo.value)
    assert "the protected locations could not be listed" in message
    assert "PackProof installation" in message
    assert "checkout" in message
    assert "interpreter" in message
    assert f"Setting {ENV_WORKSPACE_DIR} to another directory does not help" in message
    assert "to a directory you can read" not in message, "the workspace advice must not appear"
    assert isinstance(excinfo.value.__cause__, PermissionError), "chaining still holds"


def test_the_two_inspection_failures_do_not_share_their_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-22 C: the distinction has to be visible end to end, not just in two strings.

    Both messages are produced by the same validator call with the same `OSError`, so any
    overlap in their guidance is a bug in the template rather than in the test's choice of
    words. Each keeps the operating-system reason — "Permission denied" is the one detail
    that distinguishes the two fixes — and each names only the knob that can actually help.
    """
    denial = PermissionError(errno.EACCES, "Permission denied")
    workspace = tmp_path / "scan-workspace"

    monkeypatch.setattr(Path, "is_symlink", _raise_as(denial))
    with pytest.raises(ConfigurationError) as workspace_failure:
        _validate_workspace(workspace)
    monkeypatch.undo()

    monkeypatch.setattr(settings, "_interpreter_package_roots", _raise_as(denial))
    with pytest.raises(ConfigurationError) as boundary_failure:
        _validate_workspace(workspace)

    first, second = str(workspace_failure.value), str(boundary_failure.value)

    assert first != second
    assert "Permission denied" in first and "Permission denied" in second
    assert f"{ENV_WORKSPACE_DIR} to a directory you can read" in first
    assert f"{ENV_WORKSPACE_DIR} to a directory you can read" not in second
    assert "interpreter" not in first, "the workspace message must not describe the install"
    assert "interpreter" in second
    assert isinstance(workspace_failure.value.__cause__, PermissionError)
    assert isinstance(boundary_failure.value.__cause__, PermissionError)
