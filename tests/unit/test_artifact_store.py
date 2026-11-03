r"""The publication core: names this process invented, one commit syscall, one ledger.

These tests run against a real filesystem in a temporary directory, because the properties worth
proving here are kernel properties: `O_EXCL` refusing a planted symlink, `os.link` returning
`EEXIST` instead of clobbering, `O_NOFOLLOW` refusing to read through a link, and a directory
descriptor still naming the inode it was opened on after the path in front of it has been swapped.
A mock would assert the code's own beliefs back at it.

Faults are injected deterministically through the two seams `ArtifactStore` exists to expose
(`_link`, `_fsync_dir`), through `monkeypatch` on `os.urandom` / `os.write` / `os.fsync`, or
through the store's own `fact_sink`. Nothing here relies on timing except the two concurrency
tests, and those assert an outcome that holds whatever the interleaving was — one file, both
callers satisfied — rather than a particular order.

The precedence tests assert the round-8 policy exactly: **the first exception event of an
operation escapes, by identity, and every later event is recorded.** Identity is checked with
`is`, not by type, because a re-synthesised exception of the same class is precisely what the
policy forbids. Two limitations are asserted as limitations rather than as guarantees: the
cleanup runner absorbs at most `len(CLEANUP_STEPS)` events, and an interrupt delivered inside the
commit window leaves a state the ledger calls `SETTLEMENT_REQUIRED` — which is what recovery is
for, and not the same thing as atomicity.
"""

import errno
import hashlib
import os
import stat
import threading
from pathlib import Path

import pytest

from packproof.registry import artifact_store
from packproof.registry.artifact_store import (
    DESTINATION_OCCUPIED_FOREIGN,
    NOTHING_PUBLISHED,
    PREEXISTING_VERIFIED,
    PUBLISHED,
    PUBLISHED_UNSYNCED,
    SETTLEMENT_REQUIRED,
    ArtifactStore,
    Finalization,
    StagedFile,
    annotate_primary,
    finalize_artifact,
    in_flight,
    note_ledger,
)
from packproof.registry.errors import (
    ArtifactCollisionError,
    ArtifactInternalError,
    ArtifactSettlementError,
    ArtifactWriteError,
    WorkspaceDurabilityError,
    WorkspaceEntryInvalidError,
    WorkspaceOutOfSpaceError,
)

PAYLOAD = b"a tarball, hypothetically speaking"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
REF = "left-pad@1.3.0"
ROOTED = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="a superuser reads and writes regardless of the mode bits these tests set",
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def store(workspace: Path) -> ArtifactStore:
    with ArtifactStore(workspace) as opened:
        yield opened


@pytest.fixture
def record_ledger() -> Finalization:
    """A ledger with a digest but no staged file, so `finalize_artifact` fails with a definite
    primary and reaches its cleanup runner — which is where the precedence tests have to drive."""
    return Finalization(REF, DIGEST)


def _stage(store: ArtifactStore, payload: bytes = PAYLOAD) -> tuple[Finalization, StagedFile]:
    """A staged file with *payload* written into it, but no digest and no seal — the exact state a
    stream leaves behind before the last byte has been checked."""
    record = Finalization(REF)
    staged = store.stage(record)
    staged.write(payload)
    return record, staged


def _publish(store: ArtifactStore, payload: bytes = PAYLOAD) -> tuple[Finalization, str]:
    """The whole happy path, so the failure tests start from a real published file."""
    record, staged = _stage(store, payload)
    record.digest = hashlib.sha256(payload).hexdigest()
    staged.seal()
    return record, finalize_artifact(store, record)


class ExplodingSink:
    """A logger that is down. Every sink-failure case in this file is produced with this."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)
        raise RuntimeError("the log is down")


# --------------------------------------------------------------------------- construction


def test_a_relative_workspace_is_refused_before_anything_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkspaceEntryInvalidError) as caught:
        ArtifactStore("relative-workspace")
    assert "not an absolute path" in str(caught.value)


def test_a_workspace_that_does_not_exist_is_refused_with_the_operating_systems_reason(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "never-made"
    with pytest.raises(WorkspaceEntryInvalidError) as caught:
        ArtifactStore(missing)
    assert "could not be opened as a directory" in str(caught.value)


def test_a_workspace_that_is_a_file_is_refused_and_the_file_is_untouched(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "not-a-directory"
    plain.write_bytes(b"evidence")
    with pytest.raises(WorkspaceEntryInvalidError):
        ArtifactStore(plain)
    assert plain.read_bytes() == b"evidence"


def test_the_root_is_kept_as_configured_rather_than_resolved(workspace: Path) -> None:
    """`PackageArtifact.directory` is this value, and the model requires the path an operator
    configured — resolving symlinks here would report a location nobody asked for."""
    with ArtifactStore(workspace) as opened:
        assert opened.root == workspace


def test_the_context_manager_closes_and_never_swallows_the_exception_that_got_us_there(
    workspace: Path,
) -> None:
    with pytest.raises(ValueError), ArtifactStore(workspace) as opened:
        raise ValueError("a caller failed")
    assert not opened.is_open


def test_closing_is_idempotent_and_a_closed_store_refuses_every_operation(
    workspace: Path,
) -> None:
    opened = ArtifactStore(workspace)
    opened.close()
    opened.close()
    assert not opened.is_open
    with pytest.raises(ArtifactWriteError) as caught:
        opened.stage(Finalization(REF))
    assert "store is closed" in str(caught.value)


def test_a_closed_descriptor_cannot_be_reused_to_write_somewhere_else(workspace: Path) -> None:
    """The number a closed descriptor held is available for an unrelated `open()` in this
    process to take, which is why the refusal is a check and not a convention."""
    opened = ArtifactStore(workspace)
    fd = opened._fd
    opened.close()
    assert fd != opened._fd
    assert opened._fd == -1


# --------------------------------------------------------------------------- staging


def test_a_staged_name_is_generated_here_and_is_not_derived_from_anything_remote(
    store: ArtifactStore,
) -> None:
    record = Finalization(REF)
    staged = store.stage(record)
    assert staged.name.startswith(f".partial.{os.getpid()}.")
    assert len(staged.name.rsplit(".", 1)[1]) == 16
    assert record.staged_name == staged.name
    assert DIGEST not in staged.name


def test_two_stages_are_two_names_because_the_random_suffix_is_what_an_attacker_cannot_guess(
    store: ArtifactStore,
) -> None:
    first = store.stage(Finalization(REF))
    second = store.stage(Finalization(REF))
    assert first.name != second.name
    store.discard(first.name)
    store.discard(second.name)


def test_a_staged_file_is_created_private_to_this_process(store: ArtifactStore) -> None:
    record, staged = _stage(store)
    mode = (store.root / staged.name).stat().st_mode
    assert stat.S_IMODE(mode) == 0o600
    staged.close()
    store.discard(staged.name)


def test_a_planted_symlink_on_the_staged_name_is_refused_and_its_victim_survives(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`O_EXCL` is the entire symlink defence, and this is the test for it. The victim lives
    outside the workspace, so a write through the planted link would be visible as a changed
    file rather than as an argument about semantics."""
    outside = workspace.parent / "victim-outside-the-workspace"
    outside.write_bytes(b"original bytes")
    monkeypatch.setattr(os, "urandom", lambda size: b"\x01" * size)
    planted = workspace / f".partial.{os.getpid()}.{'01' * 8}"
    planted.symlink_to(outside)

    with pytest.raises(WorkspaceEntryInvalidError) as caught:
        store.stage(Finalization(REF))
    assert "could not be created in the workspace" in str(caught.value)
    assert outside.read_bytes() == b"original bytes"
    assert planted.is_symlink()


def test_a_dangling_symlink_on_the_staged_name_is_refused_the_same_way(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "urandom", lambda size: b"\x02" * size)
    (workspace / f".partial.{os.getpid()}.{'02' * 8}").symlink_to(workspace / "nothing-here")
    with pytest.raises(WorkspaceEntryInvalidError):
        store.stage(Finalization(REF))


def test_a_staged_name_refused_by_the_directory_is_a_refusal_not_an_orphan(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EEXIST means the entry is not ours. Deleting it would be deleting something PackProof did
    not make, so nothing is created, nothing is recorded, and the ledger stays empty."""
    monkeypatch.setattr(os, "urandom", lambda size: b"\x03" * size)
    taken = workspace / f".partial.{os.getpid()}.{'03' * 8}"
    taken.write_bytes(b"someone elses file")
    record = Finalization(REF)
    with pytest.raises(WorkspaceEntryInvalidError):
        store.stage(record)
    assert record.staged_name is None
    assert record.facts == []
    assert taken.read_bytes() == b"someone elses file"


@ROOTED
def test_a_workspace_that_cannot_be_written_to_is_refused_without_an_orphan(
    workspace: Path,
) -> None:
    locked = workspace / "locked"
    locked.mkdir()
    store = ArtifactStore(locked)
    locked.chmod(0o500)
    try:
        record = Finalization(REF)
        with pytest.raises(WorkspaceEntryInvalidError):
            store.stage(record)
        assert record.staged_name is None
    finally:
        locked.chmod(0o700)
        store.close()


def test_digest_of_reads_the_bytes_on_disk_and_follows_no_symlink(
    store: ArtifactStore, workspace: Path
) -> None:
    (workspace / "artifact.tgz").write_bytes(PAYLOAD)
    assert store.digest_of("artifact.tgz") == DIGEST
    (workspace / "link.tgz").symlink_to(workspace / "victim.tgz")
    (workspace / "victim.tgz").write_bytes(b"not the artifact")
    with pytest.raises(OSError) as caught:
        store.digest_of("link.tgz")
    assert caught.value.errno == errno.ELOOP


def test_a_discard_of_a_name_that_is_already_gone_is_success_not_a_failure(
    store: ArtifactStore,
) -> None:
    """The staged file being absent is the outcome this call wanted. Treating it as a failure
    would make double cleanup report an orphan that is not there."""
    store.discard("never-existed")


def test_a_discard_that_the_filesystem_refuses_reaches_the_caller_as_an_oserror(
    store: ArtifactStore, workspace: Path
) -> None:
    """`discard` raises the raw `OSError`: the caller that planted a directory in the way is the
    one that has to decide what a refusal means, and wrapping it here would take that choice away.
    Which of the two codes an unlink-on-a-directory returns is the operating system's — measured
    `EPERM` here, and `EISDIR` on some Linux builds — so the assertion is over both, not a guess
    about one."""
    (workspace / "subdir").mkdir()
    with pytest.raises(OSError) as caught:
        store.discard("subdir")
    assert caught.value.errno in {errno.EPERM, errno.EISDIR}


def test_discard_staged_reports_the_three_outcomes_and_only_unlinks_its_own_name(
    store: ArtifactStore,
) -> None:
    """The three outcomes are *nothing to do*, *gone*, and *refused* — and the third one needs a
    real refusal, because a name that is simply absent is the second."""
    record = Finalization(REF)
    assert store.discard_staged(record) == "skipped:no-staged-file"

    staged = store.stage(record)
    assert store.discard_staged(record) == "ok"
    assert record.staged_name is None
    assert not (store.root / staged.name).exists()

    record.staged_name = "a-name-this-store-never-generated"
    assert store.discard_staged(record) == "ok"
    assert record.orphaned_temp is False
    assert record.cleanup_failures == []

    (store.root / "held-open").mkdir()
    record.staged_name = "held-open"
    assert store.discard_staged(record) == "failed:discard-failed"
    assert record.staged_name == "held-open"
    assert record.orphaned_temp is True
    assert any(f.startswith("discard-failed:") for f in record.cleanup_failures)
    assert any(f.startswith("staged-orphan:") for f in record.facts)
    assert (store.root / "held-open").is_dir()


def test_discard_staged_leaves_a_published_file_alone(store: ArtifactStore) -> None:
    """Nothing in this module deletes a name it did not generate for this operation, and the
    guard is on the ledger, not on the path."""
    record, final = _publish(store)
    assert record.staged_name is None
    assert store.discard_staged(record) == "skipped:no-staged-file"
    assert (store.root / final).is_file()


# --------------------------------------------------------------------------- writing


def test_a_write_that_the_kernel_accepts_a_byte_at_a_time_still_writes_the_whole_chunk(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`os.write` may accept fewer bytes than asked for, and a loop that assumed it did not would
    silently truncate the artifact — publishing a file named for the digest of bytes nobody
    wrote. This is the test for the loop."""
    real_write = os.write
    calls: list[int] = []

    def one_byte(fd: int, data: memoryview | bytes) -> int:
        calls.append(1)
        return real_write(fd, data[:1])

    monkeypatch.setattr(os, "write", one_byte)
    record = Finalization(REF)
    staged = store.stage(record)
    assert staged.write(b"0123456789") == 10
    staged.seal()
    assert len(calls) == 10
    assert (store.root / staged.name).read_bytes() == b"0123456789"
    store.discard(staged.name)


def test_a_write_that_accepts_zero_bytes_stops_instead_of_spinning(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `<= 0` guard is what keeps the partial-write loop from running forever."""
    monkeypatch.setattr(os, "write", lambda fd, data: 0)
    record = Finalization(REF)
    staged = store.stage(record)
    with pytest.raises(ArtifactWriteError) as caught:
        staged.write(b"anything at all")
    assert "accepted no bytes at all" in str(caught.value)
    staged.close()
    store.discard(staged.name)


@pytest.mark.parametrize(
    ("number", "category", "wording"),
    [
        (errno.ENOSPC, WorkspaceOutOfSpaceError, "ran out of space"),
        (errno.EDQUOT, WorkspaceOutOfSpaceError, "ran out of space"),
        (errno.EFBIG, WorkspaceOutOfSpaceError, "ran out of space"),
        (errno.EACCES, ArtifactWriteError, "could not write"),
        (errno.EIO, ArtifactWriteError, "could not write"),
    ],
)
def test_a_write_failure_is_sorted_by_errno_into_the_class_the_operator_can_act_on(
    store: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    number: int,
    category: type,
    wording: str,
) -> None:
    def refuse(fd: int, data: memoryview | bytes) -> None:
        raise OSError(number, "the kernel said no")

    monkeypatch.setattr(os, "write", refuse)
    record = Finalization(REF)
    staged = store.stage(record)
    with pytest.raises(category) as caught:
        staged.write(b"bytes")
    assert wording in str(caught.value)
    assert staged.name in str(caught.value)
    assert str(store.root) not in str(caught.value)
    staged.close()
    store.discard(staged.name)


def test_an_oserror_built_without_a_number_is_a_write_failure_not_a_disk_full_report(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`errno is None` is falsy, and promoting it to "free space" would send an operator to
    check a volume that was never the problem."""

    def refuse(fd: int, data: memoryview | bytes) -> None:
        raise OSError("an exception built by hand")

    monkeypatch.setattr(os, "write", refuse)
    record = Finalization(REF)
    staged = store.stage(record)
    with pytest.raises(ArtifactWriteError) as caught:
        staged.write(b"bytes")
    assert "carrying no errno" in str(caught.value)
    staged.close()
    store.discard(staged.name)


def test_a_seal_that_cannot_flush_is_a_pre_commit_failure_and_the_descriptor_is_released(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is published when the contents may still be in the page cache, and the descriptor
    is closed anyway — a leaked fd is not a fact about a package."""
    record, staged = _stage(store)
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "gone")))
    with pytest.raises(ArtifactWriteError):
        staged.seal()
    assert staged._fd == -1
    store.discard(staged.name)


def test_sealing_twice_is_refused_because_the_second_one_has_nothing_to_flush(
    store: ArtifactStore,
) -> None:
    record, staged = _stage(store)
    staged.seal()
    with pytest.raises(ArtifactWriteError) as caught:
        staged.seal()
    assert "already closed" in str(caught.value)
    store.discard(staged.name)


def test_closing_a_staged_file_twice_is_silently_fine(store: ArtifactStore) -> None:
    record, staged = _stage(store)
    staged.close()
    staged.close()
    store.discard(staged.name)


# --------------------------------------------------------------------------- the commit point


def test_publishing_without_a_staged_file_is_refused(store: ArtifactStore) -> None:
    record = Finalization(REF, digest=DIGEST)
    with pytest.raises(ArtifactWriteError) as caught:
        store.finalize(record)
    assert "no staged artifact" in str(caught.value)


def test_publishing_without_a_digest_is_refused_rather_than_naming_a_file_after_nothing(
    store: ArtifactStore,
) -> None:
    """`finalize_artifact` is reached only after streaming, but the name is built from this
    field, and `None.tgz` in a content-addressed workspace would be an artifact named for
    nothing."""
    record, staged = _stage(store)
    staged.seal()
    assert record.digest is None
    with pytest.raises(ArtifactWriteError) as caught:
        store.finalize(record)
    assert "no content digest" in str(caught.value)


def test_the_store_publishes_under_the_name_it_is_handed_rather_than_one_it_checked(
    tmp_path: Path,
) -> None:
    """A measured limitation, recorded so nobody has to rediscover it.

    `finalize` requires a digest to *exist*; it never asks whether one *is* a digest. The name is
    built by string formatting and handed to `os.link` with the workspace descriptor as the
    directory, and a directory descriptor still honours `..` — while an absolute name ignores the
    descriptor completely. Both were run here before this test existed, and both put a file outside
    the workspace.

    This is not reachable from a download: `npm_download` assigns `hashlib.hexdigest()` as the
    digest, never anything read from the registry, and `PackageArtifact` refuses a `file_name` that
    is not `<64 hex>.tgz`, so an escaped name could not be reported even if it were created. What
    this test pins is where the guarantee lives — the caller and the model — instead of letting a
    reader infer a check from a store that does not perform one. If a later stage ever names an
    artifact from anything remote, this is the record that the store will not stop it.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with ArtifactStore(workspace) as store:
        climbing = Finalization(REF, digest="../escape")
        staged = store.stage(climbing)
        staged.write(PAYLOAD)
        staged.seal()
        published_climbing = finalize_artifact(store, climbing)

        absolute = Finalization(REF, digest=str(tmp_path / "absolute"))
        staged = store.stage(absolute)
        staged.write(PAYLOAD)
        staged.seal()
        published_absolute = finalize_artifact(store, absolute)

    assert published_climbing == "../escape.tgz"
    assert published_absolute == f"{tmp_path / 'absolute'}.tgz"
    assert (tmp_path / "escape.tgz").is_file()
    assert (tmp_path / "absolute.tgz").is_file()
    assert list(workspace.iterdir()) == [], "the workspace was supposed to hold both artifacts"


def test_a_published_artifact_is_the_payload_under_its_own_digest_with_no_litter(
    store: ArtifactStore, workspace: Path
) -> None:
    record, final = _publish(store)
    assert final == f"{DIGEST}.tgz"
    assert (workspace / final).read_bytes() == PAYLOAD
    assert record.state == PUBLISHED
    assert record.synced is True
    assert record.recovered is False
    assert not list(workspace.glob(".partial.*"))
    assert record.facts[-1] == f"published:{final}"


def test_a_published_artifact_keeps_the_private_mode_it_was_staged_in(
    store: ArtifactStore, workspace: Path
) -> None:
    """Measured rather than assumed: publication is a link to the same inode, so the 0o600 the
    staged file was created with is the mode the artifact keeps. PackProof does not hand a
    downloaded package to every reader on the machine."""
    _record, final = _publish(store)
    mode = (workspace / final).stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_the_link_happens_exactly_once_and_leaves_one_directory_entry(
    store: ArtifactStore,
) -> None:
    record, final = _publish(store)
    assert store.link_count(final) == 1


def test_the_intent_is_recorded_before_the_syscall_because_the_window_between_them_is_the_point(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`link_started` written after the call would describe an interrupted operation as "nothing
    happened", and settlement would never be asked. This captures the ledger at the moment the
    link is entered."""
    seen: list[bool] = []
    real_link = ArtifactStore._link

    def spy(self: ArtifactStore, rec: Finalization, staged: str, final: str) -> None:
        seen.append(rec.link_started)
        real_link(self, rec, staged, final)

    monkeypatch.setattr(ArtifactStore, "_link", spy)
    record, staged = _stage(store)
    record.digest = DIGEST
    staged.seal()
    store.finalize(record)
    assert seen == [True]


def test_only_eexist_may_ask_the_filesystem_about_the_destination(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign file is planted at the destination and the link fails with `EACCES`. Reading the
    destination after a failure that was not `EEXIST` would be inspecting an entry this process
    has no evidence it created, so the collision path must not run."""
    (workspace / f"{DIGEST}.tgz").write_bytes(b"foreign bytes")
    record, staged = _stage(store)
    record.digest = DIGEST
    staged.seal()

    def refuse(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(ArtifactStore, "_link", refuse)
    monkeypatch.setattr(
        ArtifactStore,
        "_verify_existing",
        lambda *a, **k: pytest.fail("EACCES must not enter existing-destination verification"),
    )
    with pytest.raises(ArtifactWriteError) as caught:
        store.finalize(record)
    assert "could not be published" in str(caught.value)
    assert record.state == NOTHING_PUBLISHED
    assert record.link_failed is True
    assert (workspace / f"{DIGEST}.tgz").read_bytes() == b"foreign bytes"


def test_an_out_of_space_link_is_a_space_failure_and_still_inspects_nothing(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, staged = _stage(store)
    record.digest = DIGEST
    staged.seal()

    def refuse(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(ArtifactStore, "_link", refuse)
    with pytest.raises(WorkspaceOutOfSpaceError):
        store.finalize(record)
    assert record.link_failed is True


# --------------------------------------------------------------------------- the destination held by someone else


def test_a_destination_that_already_holds_these_bytes_is_a_success_without_a_second_write(
    store: ArtifactStore, workspace: Path
) -> None:
    (workspace / f"{DIGEST}.tgz").write_bytes(PAYLOAD)
    before = (workspace / f"{DIGEST}.tgz").stat().st_ino
    record, final = _publish(store)
    assert final == f"{DIGEST}.tgz"
    assert record.state == PREEXISTING_VERIFIED
    assert record.verified_existing is True
    assert record.synced is True
    assert record.facts[-1] == f"reused-existing:{final}"
    assert (workspace / final).stat().st_ino == before


@pytest.mark.parametrize(
    ("kind", "wording"),
    [
        ("directory", "held by a directory"),
        ("symlink", "held by a symlink"),
        ("fifo", "held by a fifo"),
    ],
)
def test_a_destination_held_by_something_else_is_refused_by_what_it_is(
    store: ArtifactStore, workspace: Path, kind: str, wording: str
) -> None:
    """The kind is named because the action differs: a directory is a workspace someone pointed
    at the wrong place, a symlink is an attack, and a fifo is neither."""
    destination = workspace / f"{DIGEST}.tgz"
    if kind == "directory":
        destination.mkdir()
    elif kind == "symlink":
        destination.symlink_to(workspace / "victim.tgz")
        (workspace / "victim.tgz").write_bytes(b"victim")
    else:
        os.mkfifo(destination)

    record = Finalization(REF)
    staged = store.stage(record)
    staged.write(PAYLOAD)
    record.digest = DIGEST
    staged.seal()
    with pytest.raises(ArtifactCollisionError) as caught:
        finalize_artifact(store, record)
    assert wording in str(caught.value)
    assert record.state == DESTINATION_OCCUPIED_FOREIGN
    assert not list(workspace.glob(".partial.*"))
    if kind == "symlink":
        assert (workspace / "victim.tgz").read_bytes() == b"victim"
        assert destination.is_symlink()


def test_foreign_bytes_in_the_way_are_preserved_down_to_the_mtime(
    store: ArtifactStore, workspace: Path
) -> None:
    destination = workspace / f"{DIGEST}.tgz"
    destination.write_bytes(b"NOT THE SAME CONTENT")
    before = destination.stat()
    record = Finalization(REF)
    staged = store.stage(record)
    staged.write(PAYLOAD)
    record.digest = DIGEST
    staged.seal()
    with pytest.raises(ArtifactCollisionError) as caught:
        finalize_artifact(store, record)
    assert "holds a different payload" in str(caught.value)
    after = destination.stat()
    assert destination.read_bytes() == b"NOT THE SAME CONTENT"
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_size,
    )


@ROOTED
def test_a_regular_file_that_cannot_be_read_is_foreign_and_stays_there(
    store: ArtifactStore, workspace: Path
) -> None:
    """`_verify_existing` re-reads the destination because "the name exists" and "the name holds
    these bytes" are different claims. A file that cannot be read cannot make the second claim,
    so PackProof did not create it and does not get to decide what happens to it."""
    destination = workspace / f"{DIGEST}.tgz"
    destination.write_bytes(PAYLOAD)
    destination.chmod(0o000)
    record = Finalization(REF)
    staged = store.stage(record)
    staged.write(PAYLOAD)
    record.digest = DIGEST
    staged.seal()
    try:
        with pytest.raises(ArtifactCollisionError) as caught:
            finalize_artifact(store, record)
        assert "could not read to verify it" in str(caught.value)
    finally:
        destination.chmod(0o600)


def test_a_destination_that_vanishes_between_the_refusal_and_the_stat_is_a_write_failure(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`EEXIST` then `ENOENT` means the entry changed underneath this operation, which is a
    different fact than "the destination is foreign" and must not be reported as it."""
    record, staged = _stage(store)
    record.digest = DIGEST
    staged.seal()

    def link_then_vanish(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        raise OSError(errno.EEXIST, "file exists")

    monkeypatch.setattr(ArtifactStore, "_link", link_then_vanish)

    def vanish(self: ArtifactStore, name: str) -> os.stat_result:
        raise OSError(errno.ENOENT, "no such file or directory")

    monkeypatch.setattr(ArtifactStore, "digest_of", vanish)
    real_stat = os.stat

    def statting_through_a_hole(*args: object, **kwargs: object) -> os.stat_result:
        if str(args[0]).endswith(f"{DIGEST}.tgz"):
            raise OSError(errno.ENOENT, "no such file or directory")
        return real_stat(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", statting_through_a_hole)
    with pytest.raises(ArtifactWriteError) as caught:
        store.finalize(record)
    assert "changed under us" in str(caught.value)
    assert record.state == NOTHING_PUBLISHED


# --------------------------------------------------------------------------- durability after the commit


def test_a_directory_that_cannot_be_synced_keeps_the_artifact_and_says_so(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes are on the volume; only the durability of the *directory entry* is in question.
    Replacing that with a failed download would delete the one thing the operation proved it
    had, so the state is `PUBLISHED_UNSYNCED` and the exception carries the name."""

    def broken_sync(self: ArtifactStore) -> None:
        raise OSError(errno.EIO, "device went away")

    monkeypatch.setattr(ArtifactStore, "_fsync_dir", broken_sync)
    record = Finalization(REF)
    staged = store.stage(record)
    staged.write(PAYLOAD)
    record.digest = DIGEST
    staged.seal()
    with pytest.raises(WorkspaceDurabilityError) as caught:
        store.finalize(record)
    assert caught.value.state == PUBLISHED_UNSYNCED
    assert caught.value.final_name == f"{DIGEST}.tgz"
    assert (workspace / f"{DIGEST}.tgz").read_bytes() == PAYLOAD
    assert record.state == PUBLISHED_UNSYNCED


def test_the_cleanup_runner_retires_the_staged_file_left_by_a_failed_sync(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`finalize` raises before its own retire step, so without the runner the workspace would
    keep a `.partial.*` file nobody mentioned."""

    def broken_sync(self: ArtifactStore) -> None:
        raise OSError(errno.EIO, "device went away")

    monkeypatch.setattr(ArtifactStore, "_fsync_dir", broken_sync)
    with pytest.raises(WorkspaceDurabilityError):
        _publish(store)
    assert not list(workspace.glob(".partial.*"))


def test_a_sync_that_reports_errno_zero_is_still_a_failure(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`if sync_errno:` would read as false for a real error carrying no number, which is why
    the object is the predicate."""

    def broken_sync(self: ArtifactStore) -> None:
        raise OSError(0, "unnamed")

    monkeypatch.setattr(ArtifactStore, "_fsync_dir", broken_sync)
    with pytest.raises(WorkspaceDurabilityError):
        _publish(store)


# --------------------------------------------------------------------------- settlement


def _undetermined_record(store: ArtifactStore, workspace: Path) -> Finalization:
    """A ledger that says a link started and never came back: the state settlement exists for."""
    record = Finalization(REF)
    record.digest = DIGEST
    record.link_started = True
    staged = store.stage(record)
    staged.close()
    return record


def test_settlement_finds_absence_and_calls_it_definitive(
    store: ArtifactStore,
) -> None:
    record = _undetermined_record(store, store.root)
    assert store.settle(record) is None
    assert record.settlement_note == "absent"
    assert record.link_failed is True
    assert record.state == NOTHING_PUBLISHED
    assert "settled-absent" in record.facts[-1]


def test_settlement_recovers_a_publication_the_ledger_never_finished_recording(
    store: ArtifactStore, workspace: Path
) -> None:
    (workspace / f"{DIGEST}.tgz").write_bytes(PAYLOAD)
    record = _undetermined_record(store, workspace)
    assert store.settle(record) == f"{DIGEST}.tgz"
    assert record.recovered is True
    assert record.published_as == f"{DIGEST}.tgz"
    assert record.state == PUBLISHED_UNSYNCED, "recovered bytes were never synced by this operation"
    assert "settled-published" in record.facts[-1]


def test_settlement_is_read_only_and_idempotent(store: ArtifactStore, workspace: Path) -> None:
    (workspace / f"{DIGEST}.tgz").write_bytes(PAYLOAD)
    record = _undetermined_record(store, workspace)
    before = (workspace / f"{DIGEST}.tgz").stat()
    first = store.settle(record)
    second = store.settle(record)
    assert first == second == f"{DIGEST}.tgz"
    assert record.settlement_attempts == 1, "the second call short-circuits on the commit fact"
    assert (workspace / f"{DIGEST}.tgz").stat() == before


def test_settlement_that_hashes_a_different_payload_calls_it_foreign(
    store: ArtifactStore, workspace: Path
) -> None:
    (workspace / f"{DIGEST}.tgz").write_bytes(b"someone elses bytes")
    record = _undetermined_record(store, workspace)
    assert store.settle(record) is None
    assert record.destination_foreign is True
    assert record.settlement_note == "digest-mismatch"
    assert record.state == DESTINATION_OCCUPIED_FOREIGN


def test_settlement_of_a_symlink_destination_refuses_to_follow_it(
    store: ArtifactStore, workspace: Path
) -> None:
    """The link target holds the payload, so following it would "recover" a publication by
    reading a file outside the workspace."""
    (workspace / "victim.tgz").write_bytes(PAYLOAD)
    (workspace / f"{DIGEST}.tgz").symlink_to(workspace / "victim.tgz")
    record = _undetermined_record(store, workspace)
    assert store.settle(record) is None
    assert record.settlement_note == "foreign"
    assert "settled-foreign" in record.facts[-1]


def test_a_settlement_that_cannot_stat_is_an_error_and_not_a_guess(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SETTLEMENT_REQUIRED` with the read itself failing means nobody knows. Acting on either
    answer would be PackProof inventing a fact about the workspace."""
    record = _undetermined_record(store, store.root)

    # Any signature, because this replaces a function the whole interpreter shares: a stub that
    # mis-typed the call would raise `TypeError` out of `os.stat`, and `settle` only catches
    # `OSError` — the test would then be measuring pytest's own teardown, not this branch.
    def unreadable(*args: object, **kwargs: object) -> os.stat_result:
        raise OSError(errno.EIO, "cannot look")

    monkeypatch.setattr(os, "stat", unreadable)
    with pytest.raises(ArtifactSettlementError) as caught:
        store.settle(record)
    assert "could not determine" in str(caught.value)
    assert record.state == SETTLEMENT_REQUIRED
    assert "settled-undetermined" in record.facts[-1]


def test_settlement_is_skipped_when_the_ledger_already_knows(
    store: ArtifactStore,
) -> None:
    record = Finalization(REF)
    record.link_started = True
    record.link_failed = True
    record.digest = DIGEST
    assert store.settle(record) is None
    assert record.settlement_attempts == 0


def test_an_unreadable_recovered_file_is_recorded_as_unreadable_not_undetermined(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / f"{DIGEST}.tgz").write_bytes(PAYLOAD)
    monkeypatch.setattr(
        ArtifactStore,
        "digest_of",
        lambda self, name: (_ for _ in ()).throw(OSError(errno.EACCES, "denied")),
    )
    record = _undetermined_record(store, workspace)
    assert store.settle(record) is None
    assert record.settlement_note.startswith("unreadable:")
    assert "settled-unreadable" in record.facts[-1]


# --------------------------------------------------------------------------- precedence and cleanup


def test_the_primary_escapes_by_identity_and_the_publication_still_happens(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupt delivered after the link landed is the operation's answer: it escapes as the
    same object, with its own message, and the ledger's `SETTLEMENT_REQUIRED` window is closed by
    recovery rather than by a guess."""
    real_link = ArtifactStore._link
    # One object, so `is` below means something: a re-synthesised `KeyboardInterrupt` of the same
    # class would satisfy a `pytest.raises` type check and violate the precedence policy.
    interrupt = KeyboardInterrupt()

    def interrupt_after_linking(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        real_link(self, rec, src, dst)
        raise interrupt

    monkeypatch.setattr(ArtifactStore, "_link", interrupt_after_linking)
    with pytest.raises(KeyboardInterrupt) as caught:
        _publish(store)
    assert caught.value is interrupt
    assert str(caught.value) == ""
    assert (workspace / f"{DIGEST}.tgz").read_bytes() == PAYLOAD
    assert not list(workspace.glob(".partial.*"))


def test_a_secondary_interrupt_during_cleanup_never_replaces_the_primary(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact round-8 rule: a cleanup event that arrives while a primary owns the propagation
    slot is recorded in `displaced` and dropped. The caller sees the artifact error."""

    def refuse(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(ArtifactStore, "_link", refuse)
    monkeypatch.setattr(
        ArtifactStore,
        "discard_staged",
        lambda self, rec: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    # The ledger is built here rather than by `_publish` because the interesting facts live in it:
    # on the raising path the record never comes back to the caller, so a test that could not
    # reach it could only assert the type of the exception — and the type is not the policy.
    record = Finalization(REF)
    staged = store.stage(record)
    staged.write(PAYLOAD)
    record.digest = DIGEST
    staged.seal()

    with pytest.raises(ArtifactWriteError) as caught:
        finalize_artifact(store, record)

    assert caught.value.__class__ is ArtifactWriteError
    assert "could not be published" in str(caught.value)
    assert record.primary_kind == "ArtifactWriteError"
    assert record.displaced == ["discard-staged:KeyboardInterrupt"]
    assert "KeyboardInterrupt" in record.interrupts_seen
    # The dropped event still has to be true about the workspace: the staged file was never
    # unlinked, so the ledger says so instead of reading as though cleanup had finished.
    assert record.orphaned_temp is True
    assert any(f.startswith("staged-orphan:") for f in record.facts)
    assert (workspace / record.staged_name).is_file()


def test_the_cleanup_runner_defers_the_first_interrupt_only_when_nothing_owns_the_slot(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `deferred` half of the runner's contract, pinned on the runner itself.

    A `BaseException` from a cleanup step with no primary of its own must not be converted into an
    ordinary outcome — an interrupt must not turn an operation into a success. `finalize_artifact`
    cannot reach that branch: when it runs the steps, either a caught `Exception` or the
    interpreter's own unwinding state already owns the slot, so `primary` is never `None` there.
    Testing the public driver for this behaviour would mean faking `in_flight()`, so the contract is
    asserted where it lives, and the reachable half (a primary present) is asserted through
    `finalize_artifact` in the tests above and below this one.
    """
    interrupt = SystemExit("nothing owns the slot")
    monkeypatch.setattr(
        artifact_store,
        "_step",
        lambda store_arg, rec, held, step: (_ for _ in ()).throw(interrupt),
    )

    without_owner = Finalization(REF, DIGEST)
    deferred = artifact_store._run_steps(store, without_owner, None, None)
    assert deferred is interrupt
    # One slot, four steps: the first event is deferred and every later one is recorded, which is
    # the same bounded-absorption rule the next assertion measures for the owned-slot case.
    assert without_owner.displaced == [
        f"{step}:SystemExit" for step in artifact_store.CLEANUP_STEPS[1:]
    ]

    # Same event, one slot owner already present: recorded and dropped instead of deferred.
    primary = ArtifactWriteError("the operation's answer")
    with_owner = Finalization(REF, DIGEST)
    assert artifact_store._run_steps(store, with_owner, primary, primary) is None
    assert with_owner.displaced == [f"{step}:SystemExit" for step in artifact_store.CLEANUP_STEPS]


def test_a_settlement_interrupted_once_gets_its_one_retry_because_settling_never_writes(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry is safe for exactly one reason: settlement only reads, hashes and never unlinks. The
    retry repairs the ledger, and the interrupt still goes back out by identity.

    The scenario is the commit window: the link syscall really ran, the destination exists on disk,
    and an interrupt landed between the syscall returning and the ledger hearing about it.
    Settlement is the only thing that can say so.
    """
    record = Finalization(REF, DIGEST)
    staged = store.stage(record)
    staged.write(PAYLOAD)
    staged.seal()

    interrupt = KeyboardInterrupt()
    real_link = ArtifactStore._link

    def interrupt_inside_the_link(
        self: ArtifactStore, rec: Finalization, src: str, dst: str
    ) -> None:
        real_link(self, rec, src, dst)
        raise interrupt

    real_settle = ArtifactStore.settle
    calls: list[int] = []

    def interrupt_once(self: ArtifactStore, rec: Finalization) -> str | None:
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real_settle(self, rec)

    monkeypatch.setattr(ArtifactStore, "_link", interrupt_inside_the_link)
    monkeypatch.setattr(ArtifactStore, "settle", interrupt_once)

    with pytest.raises(KeyboardInterrupt) as caught:
        finalize_artifact(store, record)
    assert caught.value is interrupt, "the operation's own interrupt, not the cleanup one"
    assert len(calls) == 2, "one interrupt, one retry, and no loop"
    assert record.published_as == f"{DIGEST}.tgz"
    assert record.recovered is True
    assert record.settlement_incomplete is False
    assert record.accounted is True
    assert any(fact.startswith("settle-retry:") for fact in record.facts)
    assert "settled-published" in " ".join(record.facts)
    assert record.state == PUBLISHED_UNSYNCED


def test_the_cleanup_runner_absorbs_at_most_one_event_per_step(
    store: ArtifactStore, record_ledger: Finalization, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is `len(CLEANUP_STEPS)`, and it exists because a runner that retried an absorbed
    event could be interrupted forever."""
    monkeypatch.setattr(
        artifact_store,
        "_step",
        lambda store_arg, rec, held, step: (_ for _ in ()).throw(KeyboardInterrupt(step)),
    )
    with pytest.raises(ArtifactWriteError) as caught:
        finalize_artifact(store, record_ledger)
    assert "no staged artifact" in str(caught.value)
    assert len(record_ledger.interrupts_seen) == len(artifact_store.CLEANUP_STEPS)
    assert len(record_ledger.displaced) == len(artifact_store.CLEANUP_STEPS)
    assert [step for step, _ in record_ledger.steps] == list(artifact_store.CLEANUP_STEPS)


def test_an_ordinary_exception_from_a_cleanup_step_is_recorded_and_skipped(
    store: ArtifactStore, record_ledger: Finalization, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug in cleanup must never become the story of the operation, and must never be allowed
    to look like an interrupt."""

    def explode(*args: object) -> None:
        raise ValueError("a cleanup bug")

    monkeypatch.setattr(artifact_store, "_step", explode)
    with pytest.raises(ArtifactWriteError) as caught:
        finalize_artifact(store, record_ledger)
    assert "no staged artifact" in str(caught.value)
    assert record_ledger.interrupts_seen == []
    assert record_ledger.displaced == []
    assert [step for step, _ in record_ledger.steps] == list(artifact_store.CLEANUP_STEPS)
    assert all(outcome.startswith("failed:ValueError") for _, outcome in record_ledger.steps)
    assert record_ledger.pending_failure == "ArtifactWriteError"


def test_a_failed_discard_leaves_the_primary_telling_the_operator_about_the_file(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`WorkspaceEntryInvalidError` alone would say "the workspace refused the bytes"; the file
    is still there, and the exception is the only channel that reaches the caller."""

    def refuse(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(ArtifactStore, "_link", refuse)
    monkeypatch.setattr(
        ArtifactStore,
        "discard",
        lambda self, name: (_ for _ in ()).throw(OSError(errno.EBUSY, "device busy")),
    )
    with pytest.raises(ArtifactWriteError) as caught:
        _publish(store)
    message = str(caught.value)
    assert "was left in the workspace" in message
    assert ".partial." in message
    assert caught.value.staged_left_behind is True
    assert any(f.startswith("discard-failed:") for f in caught.value.cleanup_failures)


def test_the_orphan_clause_is_added_once_and_not_again(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`annotate_primary` runs twice by design — once as a step, once in the tail — and a clause
    appended twice would read as a corrupted message."""

    def refuse(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(ArtifactStore, "_link", refuse)
    monkeypatch.setattr(
        ArtifactStore,
        "discard",
        lambda self, name: (_ for _ in ()).throw(OSError(errno.EBUSY, "device busy")),
    )
    with pytest.raises(ArtifactWriteError) as caught:
        _publish(store)
    error = caught.value
    message = str(error)
    assert message.count("was left in the workspace") == 1
    assert error.staged_left_behind is True
    # The clause names the staged file, which this process invented — the one name in the whole
    # message that is not registry-supplied, and therefore the only path name safe to print.
    assert ".partial." in message
    assert any(f.startswith("discard-failed:") for f in error.cleanup_failures)


def test_a_transport_failure_gets_no_staged_file_clause(
    store: ArtifactStore,
) -> None:
    """`annotate_primary` is deliberately limited to the publication family. An artifact error
    about a workspace has a workspace to talk about; anything else does not."""
    error = ArtifactInternalError("something PackProof did not predict")
    record = Finalization(REF)
    assert annotate_primary(record, error) == "skipped:not-an-artifact-error"
    assert error.args[0] == "something PackProof did not predict"


def test_annotating_with_no_primary_is_a_no_op() -> None:
    assert annotate_primary(Finalization(REF), None) == "skipped:no-primary"


def test_a_successful_publish_never_annotates_anything(store: ArtifactStore) -> None:
    record, final = _publish(store)
    assert record.accounted is False
    assert not any(f.startswith("aborted-but-published") for f in record.facts)
    assert final == f"{DIGEST}.tgz"


def test_an_abort_after_a_committed_publish_is_accounted_for(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifact exists and the operation was interrupted: both are true, and the ledger has
    to say both, because a caller that deleted the file on seeing an interrupt would destroy a
    verified download."""
    record = Finalization(REF)
    record.digest = DIGEST
    staged = store.stage(record)
    staged.write(PAYLOAD)
    staged.seal()
    real_finalize = ArtifactStore.finalize
    interrupt = KeyboardInterrupt()

    def commit_then_be_interrupted(self: ArtifactStore, rec: Finalization) -> str:
        real_finalize(self, rec)
        raise interrupt

    monkeypatch.setattr(ArtifactStore, "finalize", commit_then_be_interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        finalize_artifact(store, record)
    assert caught.value is interrupt
    assert (workspace / f"{DIGEST}.tgz").read_bytes() == PAYLOAD
    assert record.published_as == f"{DIGEST}.tgz"
    assert record.accounted is True
    assert record.state == PUBLISHED
    assert any(f.startswith("aborted-but-published") for f in record.facts)


# --------------------------------------------------------------------------- the recording primitive


def test_an_unregistered_fact_kind_loses_its_detail(store: ArtifactStore) -> None:
    """A hastily added call site cannot put an arbitrary string in front of a terminal, which is
    the half of the policy that covers what gets *recorded*."""
    record = Finalization(REF)
    fact = store.record_fact(
        "some-new-kind", "https://registry.invalid/a.tgz?capability=zz", record
    )
    assert fact == "unregistered-fact:'some-new-kind'"
    assert "registry.invalid" not in " ".join(record.facts)
    assert "capability" not in " ".join(record.facts)
    assert record.facts[-1] == fact


def test_a_registered_fact_reaches_the_ledger_before_the_sink(
    store: ArtifactStore, workspace: Path
) -> None:
    """The ledger is the record; the sink is a copy of it. A sink that raises therefore cannot
    lose a fact."""
    store.fact_sink = ExplodingSink()
    record = Finalization(REF)
    store.record_fact("published", "x.tgz", record)
    assert record.facts == ["published:x.tgz"]
    assert record.sink_failed is True
    assert record.cleanup_failures == ["sink-failed:RuntimeError"]
    assert record.sink_attempts == 1


def test_a_broken_sink_is_not_touched_again_for_the_same_operation(workspace: Path) -> None:
    sink = ExplodingSink()
    with ArtifactStore(workspace, fact_sink=sink) as store:
        record = Finalization(REF)
        for _ in range(3):
            store.record_fact("published", "x.tgz", record)
        assert len(sink.lines) == 1
        assert record.sink_attempts == 1
        assert record.facts == ["published:x.tgz"] * 3


def test_a_publication_survives_a_broken_sink(workspace: Path) -> None:
    """A dead logger is not a failed download: the artifact is still committed and the ledger
    still tells the whole story, because the sink is a copy of the record and not the record."""
    sink = ExplodingSink()
    with ArtifactStore(workspace, fact_sink=sink) as store:
        record, final = _publish(store)
        assert final == f"{DIGEST}.tgz"
        assert record.facts[-1] == f"published:{final}"
        assert record.sink_failed is True
        assert (workspace / final).read_bytes() == PAYLOAD


def test_an_interrupt_inside_the_sink_is_not_converted_into_a_recorded_failure(
    store: ArtifactStore,
) -> None:
    """The honest edge of the non-throwing promise: `record_fact` catches `Exception`, so a real
    interrupt delivered on the way to the logger keeps propagating. Pretending a logger could
    swallow an interrupt would be a false guarantee."""

    class InterruptingSink:
        def append(self, line: str) -> None:
            raise KeyboardInterrupt

    store.fact_sink = InterruptingSink()
    with pytest.raises(KeyboardInterrupt):
        store.record_fact("published", "x.tgz", Finalization(REF))


def test_a_fact_recorded_without_a_ledger_still_reaches_the_sink(store: ArtifactStore) -> None:
    assert store.record_fact("settled-absent", "x") == "settled-absent:x"


def test_the_recording_vocabulary_is_a_closed_set() -> None:
    """`artifact_store` and this file agree on the kinds, and a kind outside it is a bug rather
    than a new fact."""
    assert "unregistered-fact" in artifact_store.FACT_KINDS
    assert "attempt-failed" in artifact_store.FACT_KINDS
    assert "close-failed" in artifact_store.FACT_KINDS
    assert all(isinstance(kind, str) for kind in artifact_store.FACT_KINDS)


def test_note_ledger_writes_the_ledger_and_touches_nothing_else() -> None:
    """The handlers that defend the primary's slot may only do work that cannot fail, which is
    why the secondary events are ledger-side and never reach the sink."""
    record = Finalization(REF)
    assert note_ledger(record, "displaced", "in-handler:RuntimeError") == (
        "displaced:in-handler:RuntimeError"
    )
    assert record.facts == ["displaced:in-handler:RuntimeError"]
    assert record.sink_attempts == 0


def test_in_flight_names_the_exception_the_interpreter_is_propagating() -> None:
    """Measured across the supported interpreters and load-bearing here: the cleanup runner uses
    this to name the primary without holding onto the object."""
    seen: list[object] = []
    try:
        raise ValueError("the primary")
    except ValueError:
        try:
            raise StopIteration
        except StopIteration:
            seen.append(in_flight())
            return_value = None
        finally:
            seen.append(in_flight())
    assert [type(item).__name__ for item in seen] == ["StopIteration", "ValueError"]
    assert return_value is None


def test_the_state_is_derived_from_the_fields_and_never_stored() -> None:
    """There is no `state` attribute to fall out of sync with the facts, so a half-written ledger
    cannot misreport an outcome."""
    assert not hasattr(Finalization(REF), "__dict__")
    record = Finalization(REF)
    record.published_as = "x.tgz"
    record.synced = False
    assert record.state == PUBLISHED_UNSYNCED
    record.synced = True
    assert record.state == PUBLISHED
    record.verified_existing = True
    assert record.state == PREEXISTING_VERIFIED


def test_five_of_the_six_states_describe_a_workspace_that_holds_something() -> None:
    """Why a boolean `published` would be a lie: "held by foreign bytes" and "nothing happened"
    both answer `False` while meaning opposite things to whoever has to clean up."""
    states = artifact_store.PUBLICATION_STATES
    assert len(states) == 6
    assert NOTHING_PUBLISHED in states
    assert SETTLEMENT_REQUIRED in states


def test_the_ledger_repr_names_no_url(store: ArtifactStore) -> None:
    record = Finalization("left-pad@1.3.0")
    record.digest = DIGEST
    rendered = repr(record)
    assert "https://" not in rendered
    assert "left-pad@1.3.0" in rendered
    assert DIGEST[:12] in rendered


# --------------------------------------------------------------------------- concurrency


def test_two_threads_publishing_the_same_bytes_agree_on_one_file(
    workspace: Path,
) -> None:
    """Each operation owns its own ledger and its own staged name, and the commit is one syscall
    that either created the destination or refused with `EEXIST`. So whatever the interleaving,
    exactly one ledger may claim it created the file and the other must reach existing-destination
    verification. That invariant holds by construction rather than by timing, which is why the
    barrier only encourages the two to overlap instead of being the thing under test.

    The barrier is waited on by both workers, never by the main thread: a `Barrier(2)` that the
    spawned threads do not reach leaves the *test process* parked forever, which is a hang rather
    than a failure, and every timeout here is bounded so a broken race reports instead of stalling.
    """
    outcomes: list[tuple[str, str]] = []
    errors: list[BaseException] = []
    meet = threading.Barrier(2)

    def publish() -> None:
        try:
            with ArtifactStore(workspace) as own_store:
                record, staged = _stage(own_store)
                record.digest = DIGEST
                staged.seal()
                meet.wait(timeout=10)
                outcomes.append((finalize_artifact(own_store, record), record.facts[-1]))
        except BaseException as error:  # noqa: BLE001 - the assertion is made by the caller
            errors.append(error)

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert [final for final, _ in outcomes] == [f"{DIGEST}.tgz", f"{DIGEST}.tgz"]
    assert sorted(fact.split(":", 1)[0] for _, fact in outcomes) == [
        "published",
        "reused-existing",
    ]
    assert sorted(p.name for p in workspace.iterdir()) == [f"{DIGEST}.tgz"]


def test_two_threads_publishing_different_bytes_do_not_share_a_ledger(
    workspace: Path,
) -> None:
    """The store holds no publication state, which is the property that makes the test above
    meaningful rather than lucky."""
    payloads = [b"first package bytes", b"second package bytes"]
    digests = sorted(hashlib.sha256(payload).hexdigest() for payload in payloads)
    results: list[str] = []
    errors: list[BaseException] = []

    def publish(payload: bytes) -> None:
        try:
            with ArtifactStore(workspace) as own_store:
                _record, final = _publish(own_store, payload)
                results.append(final)
        except BaseException as error:  # noqa: BLE001 - the assertion is made by the caller
            errors.append(error)

    threads = [threading.Thread(target=publish, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert sorted(results) == [f"{digest}.tgz" for digest in digests]


# --------------------------------------------------------------------------- the documented limits


def test_a_held_descriptor_keeps_naming_the_original_directory_after_the_path_is_swapped(
    tmp_path: Path,
) -> None:
    """This is the guarantee, stated exactly: the descriptor makes a swap *detectable*, not
    prevented. The workspace is renamed, a fresh directory takes its name, and the bytes still
    land in the original inode — so no artifact is written into a directory nobody configured."""
    original = tmp_path / "configured"
    original.mkdir()
    replacement = tmp_path / "looked-like-the-workspace"
    replacement.mkdir()
    store = ArtifactStore(original)
    original.rename(tmp_path / "renamed-away")
    (tmp_path / "configured").symlink_to(replacement)
    try:
        _record, final = _publish(store)
        assert (tmp_path / "renamed-away" / final).is_file()
        assert list(replacement.iterdir()) == []
    finally:
        store.close()


def test_the_commit_window_is_not_atomic_and_says_what_it_became_instead(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 8's preserved limitation: an asynchronous interrupt delivered between two bytecodes
    of the commit sequence cannot be made atomic in Python, and this project does not claim it
    is. What the design does guarantee is that the ledger names the uncertainty —
    `SETTLEMENT_REQUIRED` — and that recovery is read-only. U9/S-11 stays open because of this
    window."""
    record = Finalization(REF)
    record.digest = DIGEST
    real_link = ArtifactStore._link

    def interrupt_between_the_ledger_and_the_fact(
        self: ArtifactStore, rec: Finalization, src: str, dst: str
    ) -> None:
        rec.link_started = True
        real_link(self, rec, src, dst)
        raise KeyboardInterrupt

    monkeypatch.setattr(ArtifactStore, "_link", interrupt_between_the_ledger_and_the_fact)
    staged = store.stage(record)
    staged.write(PAYLOAD)
    staged.seal()
    with pytest.raises(KeyboardInterrupt):
        store.finalize(record)
    assert record.link_started is True
    assert record.published_as is None
    assert record.state == SETTLEMENT_REQUIRED
    assert record.outcome_undetermined is True


def test_a_state_is_never_claimed_for_bytes_the_sink_could_not_be_told_about(
    store: ArtifactStore, workspace: Path
) -> None:
    """A recording failure is a durability detail, not a different publication outcome: the
    artifact is `PUBLISHED` whether or not anybody was told, and pretending otherwise would make
    the report depend on the logger."""
    store.fact_sink = ExplodingSink()
    record, final = _publish(store)
    assert record.state == PUBLISHED
    assert (workspace / final).is_file()
    assert record.sink_failed is True
