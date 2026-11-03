r"""The publication core: one staged file, one commit point, one ledger per operation.

This is the half of Stage 2.3 that touches the filesystem, and it is the only half that
needs to. Everything here was decided by a measurement, in the round-6/7/8 design
probes, rather than by reasoning about what POSIX "should" do — several of those
reasonings were wrong when they were first written, which is why the shapes below look
fussy in places. The fussy places are the load-bearing ones.

**Names are generated, never derived.** `dir_fd` anchors a path but does not confine it:
`open("../outside", dir_fd=fd)` succeeds. So the guarantee comes entirely from the name
being one this process invented — `.partial.<pid>.<8 random bytes, hex-encoded>` while staging,
`<sha256>.tgz` once published — and from `O_EXCL` on the staged name, which is what turns
a planted symlink (live or dangling) into `EEXIST` instead of a write through the link.
The registry's URL path is never an input to either name, so a `..` in `dist.tarball` has
nowhere to land.

**Publication is a link, not a rename.** `os.replace` silently replaces a symlink's *name*
and has no `dir_fd` argument at all; `os.link` never clobbers and never follows a symlink
destination, returning `EEXIST` with the link intact. Atomicity therefore means "one
syscall that either created the destination or refused", and the only error worth
inspecting the destination for is `EEXIST`. Every other errno from `os.link` is a plain
failure, and treating, say, `EACCES` as "maybe it already exists" would read a file this
process did not create and then decide whether it counts.

**The state is derived, never stored.** `Finalization.state` is computed from the ledger's
fields, so a half-written ledger cannot misreport an outcome — there is no `state` field to
get out of sync with the facts. Five of the six states describe a workspace that now holds
something, which is why a boolean "published" would be a lie: "the destination is held by
foreign bytes" and "nothing was published" both answer `False` while meaning opposite
things to whoever has to clean up.

**Recording is one primitive with a closed vocabulary.** `record_fact` writes to the
operation's ledger *before* it touches the sink, so a sink that raises cannot lose a fact;
after the first failure the sink is not called again for that operation, and the primitive
never reports its own failure through itself. `detail` is only ever a generated name, a
bare class name or `errno_text()` output — never an exception object, never `str(exc)`,
never a URL — and an unregistered `kind` is recorded as `unregistered-fact` with its detail
dropped, so a new call site cannot leak by accident. That is the "recorded" half of the
policy; whether a fact may be *acted on* is decided downstream.

**Exception precedence, which is what rounds 7 and 8 were for:** the first exception event
of an operation escapes, by identity, and every later event is recorded. The cleanup runner
achieves that by construction — when a primary exists, the `finally` it lives in contains
no reachable `raise`, and raising from a `finally` is precisely what would displace the
primary (and, measured, would also leave it with no `__context__`). Secondary failures are
appended to the ledger instead. The boundary of that guarantee is stated plainly in
`docs/architecture.md`: an asynchronous interrupt delivered between two bytecodes of the
commit sequence cannot be made atomic in Python, and U9/S-11 stays open because of it.
"""

import contextlib
import errno as os_errno
import hashlib
import logging
import os
import stat as stat_module
import sys
from pathlib import Path
from typing import Protocol

from packproof.registry.errors import (
    ArtifactCollisionError,
    ArtifactSettlementError,
    ArtifactWriteError,
    WorkspaceDurabilityError,
    WorkspaceEntryInvalidError,
    WorkspaceOutOfSpaceError,
    errno_text,
    quote_value,
)

LOGGER = logging.getLogger("packproof.registry.artifact_store")

# --------------------------------------------------------------------------- publication states

#: Committed by this operation and the workspace directory synced.
PUBLISHED = "PUBLISHED"
#: Committed by this operation; the directory entry could not be synced. The artifact is
#: there. A power loss may not be. Never reported as a failed download.
PUBLISHED_UNSYNCED = "PUBLISHED_UNSYNCED"
#: The destination already held exactly these bytes; nothing was written.
PREEXISTING_VERIFIED = "PREEXISTING_VERIFIED"
#: The destination is held by something else, and it is still held by that something else.
DESTINATION_OCCUPIED_FOREIGN = "DESTINATION_OCCUPIED_FOREIGN"
#: A link was started and this operation never learned whether it landed.
SETTLEMENT_REQUIRED = "SETTLEMENT_REQUIRED"
#: No commit happened, and the workspace is as this operation found it.
NOTHING_PUBLISHED = "NOTHING_PUBLISHED"

#: The whole vocabulary, so a report can enumerate states instead of guessing at them.
PUBLICATION_STATES = (
    PUBLISHED,
    PUBLISHED_UNSYNCED,
    PREEXISTING_VERIFIED,
    DESTINATION_OCCUPIED_FOREIGN,
    SETTLEMENT_REQUIRED,
    NOTHING_PUBLISHED,
)

# --------------------------------------------------------------------------- recording vocabulary

#: Every fact this module may record. A `kind` outside this set is recorded as
#: `unregistered-fact` with its detail dropped, which is what keeps a hastily added call
#: site from putting an arbitrary string in front of a terminal.
FACT_KINDS = frozenset(
    {
        "published",
        "published-unsynced",
        "reused-existing",
        "staged-orphan",
        "settled-published",
        "settled-absent",
        "settled-foreign",
        "settled-unreadable",
        "settled-undetermined",
        "settle-interrupted",
        "settle-retry",
        "operation-failed",
        "attempt-failed",
        "aborted-but-published",
        "close-failed",
        "sink-failed",
        "unregistered-fact",
        "step-skipped",
        "step-failed",
        "displaced",
        # The object that owns the propagation slot, named by class at catch time.
        "primary-in-flight",
    }
)

# --------------------------------------------------------------------------- chunking

#: Bytes per staged write. The same size the metadata client reads with, so one ceiling on
#: "how much of a hostile response exists in memory at once" applies to both halves.
ARTIFACT_WRITE_CHUNK_BYTES = 64 * 1024

# errno values that mean "the volume is full or the file is too big", as opposed to
# "the workspace is not a shape we can write into". Classifying them apart is the only
# reason an operator gets told to free space rather than to repoint `workspace_dir`.
_OUT_OF_SPACE_ERRNOS = frozenset(
    code
    for name in ("ENOSPC", "EDQUOT", "EFBIG")
    if (code := getattr(os_errno, name, None)) is not None
)


def _is_out_of_space(error: OSError) -> bool:
    """Did this `OSError` come from a volume that had no room left?

    The number is the predicate, and a missing attribute is not a number: `None` is never in
    the set, so an `OSError` built by hand classifies as an ordinary write failure rather
    than being promoted to a disk-full report the operator cannot act on.
    """
    return error.errno in _OUT_OF_SPACE_ERRNOS


# --------------------------------------------------------------------------- the sink contract


class FactSink(Protocol):
    """Where recorded facts go. A `list` satisfies it, and so does the logging adapter below.

    The store's sink is an object with `append`, not a function, because "the logger is down"
    has to be injectable — replacing this with something that raises on `append` is how every
    sink-failure case in the tests is produced, at every recording site, without any
    production call site needing to know.
    """

    def append(self, line: str) -> None: ...


# --------------------------------------------------------------------------- the ledger


class Finalization:
    """The ledger of one publication attempt. Owned by the caller, never by the store.

    Every field is a fact about *this operation*. The store holds none of them, which is
    what makes two concurrent downloads of two different packages unable to corrupt each
    other's accounting — and what lets an interrupted operation be described completely by
    an object that a caller can print, keep, or throw away.
    """

    __slots__ = (
        "ref",
        "digest",
        "staged_name",
        "link_started",  # intent, written BEFORE the syscall
        "link_failed",  # definite failure, or settlement proved absence
        "published_as",  # THE commit fact: assigned at most once
        "synced",  # pessimistic default False
        "verified_existing",
        "destination_foreign",
        "recovered",  # settled from disk after an interruption
        "orphaned_temp",
        "accounted",
        "cleanup_failures",
        "settlement_note",
        "pending_failure",  # primary class name, recorded at catch time
        "facts",  # every recorded fact, ledger-side, before the sink
        "sink_failed",  # the sink is known broken: never call it again
        "sink_attempts",  # how many times it was actually touched
        "steps",  # per-step outcome trace of the cleanup runner
        "settlement_attempts",
        "settlement_incomplete",  # settlement never reached a verdict
        "interrupts_seen",  # every BaseException cleanup had to catch, in order
        "displaced",  # every cleanup event that LOST the propagation slot
        "primary_kind",  # class name of the exception that owns the slot, or None
    )

    def __init__(self, ref: str, digest: str | None = None) -> None:
        """One ledger for one operation, opened before a single byte has arrived.

        `digest` is optional because a streamed download cannot know it at the start: the
        content-addressed name is a fact about the last byte, and this object has to exist before
        the first one to record the staging, the write failures and the cleanup. It is assigned
        once, by the code that finished hashing, and `finalize` refuses to publish without it.
        """
        self.ref = ref
        self.digest = digest
        self.staged_name: str | None = None
        self.link_started = False
        self.link_failed = False
        self.published_as: str | None = None
        self.synced = False
        self.verified_existing = False
        self.destination_foreign = False
        self.recovered = False
        self.orphaned_temp = False
        self.accounted = False
        self.cleanup_failures: list[str] = []
        self.settlement_note: str | None = None
        self.pending_failure: str | None = None
        self.facts: list[str] = []
        self.sink_failed = False
        self.sink_attempts = 0
        self.steps: list[tuple[str, str]] = []
        self.settlement_attempts = 0
        self.settlement_incomplete = False
        self.interrupts_seen: list[str] = []
        self.displaced: list[str] = []
        self.primary_kind: str | None = None

    @property
    def outcome_undetermined(self) -> bool:
        """True only when the syscall may have created the destination AND the ledger never
        captured the result.

        One predicate drives both the reported state and whether settlement is worth running
        at all, so the two cannot disagree about what is unknown.
        """
        return (
            self.link_started
            and not self.link_failed
            and self.published_as is None
            and not self.destination_foreign
        )

    @property
    def state(self) -> str:
        """DERIVED. A partially-written ledger therefore cannot misreport an outcome."""
        if self.published_as is not None:
            if self.verified_existing:
                return PREEXISTING_VERIFIED
            return PUBLISHED if self.synced else PUBLISHED_UNSYNCED
        if self.destination_foreign:
            return DESTINATION_OCCUPIED_FOREIGN
        if self.outcome_undetermined:
            return SETTLEMENT_REQUIRED
        return NOTHING_PUBLISHED

    def __repr__(self) -> str:
        digest = "unset" if self.digest is None else self.digest[:12]
        return (
            f"Finalization(ref={quote_value(self.ref)}, digest={digest},"
            f" state={self.state}, published_as={quote_value(self.published_as)},"
            f" synced={self.synced}, recovered={self.recovered},"
            f" staged={quote_value(self.staged_name)}, accounted={self.accounted},"
            f" settle_attempts={self.settlement_attempts},"
            f" settle_incomplete={self.settlement_incomplete},"
            f" interrupts={self.interrupts_seen}, displaced={self.displaced},"
            f" primary={quote_value(self.primary_kind)},"
            f" cleanup={self.cleanup_failures})"
        )


# --------------------------------------------------------------------------- the staged file


class StagedFile:
    """One open staged file, owned by exactly one download attempt.

    The store deliberately does *not* hold this: an open descriptor is per-operation state,
    and a store that kept one would make two concurrent downloads overwrite each other's
    file. The name is public because it has to appear in a ledger and, if cleanup fails, in
    an error message — it is a generated name, so naming it discloses nothing a caller was
    not entitled to know.
    """

    __slots__ = ("name", "_fd")

    def __init__(self, name: str, fd: int) -> None:
        self.name = name
        self._fd = fd

    def write(self, chunk: bytes) -> int:
        """Append *chunk*, in full, and return how many bytes were added.

        `os.write` may accept fewer bytes than asked for, and a loop that assumed it did not
        would silently truncate the artifact — which, because the file is named for the
        digest of the bytes we *thought* we wrote, would publish a lie. The loop is the
        entire point; the `<= 0` guard is what stops a zero-length accept from spinning it
        forever.
        """
        remaining = memoryview(chunk)
        written_total = 0
        while remaining:
            try:
                written = os.write(self._fd, remaining)
            except OSError as error:
                raise _write_failure(self.name, error) from None
            if written <= 0:
                raise ArtifactWriteError(
                    f"the staged artifact {quote_value(self.name)} accepted no bytes at all, "
                    "so writing to it stopped"
                )
            written_total += written
            remaining = remaining[written:]
        return written_total

    def seal(self) -> None:
        """Flush the staged bytes to the volume and close the descriptor.

        fsync happens here, before the link, so a commit is never made to a destination
        whose contents may still be only in this machine's page cache. A failure is a
        pre-commit failure: nothing is published, and the workspace is as it was.
        """
        if self._fd < 0:
            raise ArtifactWriteError(
                f"the staged artifact {quote_value(self.name)} was already closed"
            )
        try:
            os.fsync(self._fd)
        except OSError as error:
            raise _write_failure(self.name, error) from None
        finally:
            self.close()

    def close(self) -> None:
        """Release the descriptor. Idempotent, and a failure here is not worth a new state.

        Suppressed rather than propagated because the caller is either publishing (the seal
        path, which fsynced and closed already) or unwinding an error it must not replace.
        A descriptor this process cannot close is a leak to be found by `lsof`, not a fact
        about a package.
        """
        fd, self._fd = self._fd, -1
        if fd < 0:
            return
        with contextlib.suppress(OSError):
            os.close(fd)


def _write_failure(name: str, error: OSError) -> ArtifactWriteError:
    """An OSError from a staged write, sorted into the class that says what to do about it.

    The staged name is included because it is generated and harmless; the workspace path is
    not, because `os.strerror` describes the *number* and a message built from it cannot
    carry a filename the way `str(error)` does.
    """
    detail = f"the staged artifact {quote_value(name)}: {errno_text(error)}"
    if _is_out_of_space(error):
        return WorkspaceOutOfSpaceError(f"PackProof ran out of space writing {detail}")
    return ArtifactWriteError(f"PackProof could not write {detail}")


# --------------------------------------------------------------------------- the production sink


class _LoggingSink:
    """The production sink: one DEBUG line per recorded fact, and nothing else.

    Facts arrive already reduced to the closed vocabulary above, so this is the one place
    the artifact path writes to the logging system — and the reason `record_fact` exists is
    that no other place has to be trusted to keep it that way.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def append(self, line: str) -> None:
        self._logger.debug("artifact fact: %s", line)


# --------------------------------------------------------------------------- the store


class ArtifactStore:
    """Holds NO publication state: only a workspace descriptor, a sink, and pure functions.

    The descriptor is opened once, on a path `config/settings.py` already vetted, and every
    operation below is relative to it. That is not a performance choice. A path-based
    `open(name)` would re-resolve the *directory* component of the name at every call, so a
    swap of the workspace to a symlink between two operations would be followed; with a held
    `dir_fd`, the components after the directory never get a chance to be re-interpreted.
    What the fd cannot do is stop the directory itself from being replaced underneath us —
    it keeps naming the *original* inode, which makes a swap detectable rather than
    prevented, and the difference is documented rather than oversold.
    """

    def __init__(self, root: str | os.PathLike[str], *, fact_sink: FactSink | None = None) -> None:
        # `root` is normalised here rather than at the call site because it is the value that
        # ends up in `PackageArtifact.directory`, and that model requires an absolute path in
        # strict mode. `is_absolute()` is string logic, not a syscall: the workspace is not
        # resolved, symlinks are not followed, and a path that does not exist fails one line
        # later at `os.open` with the operating system's own reason attached.
        configured = Path(os.fspath(root))
        if not configured.is_absolute():
            raise WorkspaceEntryInvalidError(
                f"the artifact workspace {quote_value(os.fspath(root))} is not an absolute path; "
                "a relative one names a directory that depends on where the process stands"
            )
        self.root = configured
        try:
            self._fd = os.open(configured, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as error:
            raise WorkspaceEntryInvalidError(
                f"the artifact workspace {quote_value(os.fspath(configured))} could not be opened "
                f"as a directory ({errno_text(error)})"
            ) from None
        self.fact_sink = _LoggingSink(LOGGER) if fact_sink is None else fact_sink
        self._closed = False

    # ---- lifecycle -----------------------------------------------------
    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Close the descriptor, and never swallow the exception that got us here."""
        self.close()
        return False

    def close(self) -> None:
        """Release the workspace descriptor. Idempotent, and never raises.

        The store's job ends with a published name; after this, its operations refuse. That
        refusal matters more than it looks: a closed descriptor's number can be reused by an
        unrelated `open()` in the same process, and a `dir_fd` pointing at that would let an
        artifact land somewhere nobody configured.
        """
        if self._closed:
            return
        self._closed = True
        fd, self._fd = self._fd, -1
        with contextlib.suppress(OSError):
            os.close(fd)

    @property
    def is_open(self) -> bool:
        return not self._closed

    def _dir_fd(self) -> int:
        if self._closed:
            raise ArtifactWriteError(
                "the artifact store is closed, so no workspace operation will be attempted"
            )
        return self._fd

    # ---- THE ONE RECORDING PRIMITIVE -----------------------------------
    #
    # Contract, all of it load-bearing:
    #   * never raises for an ordinary sink failure (catches `Exception`, not `BaseException`);
    #   * writes the ledger BEFORE attempting the sink, so a sink that raises cannot lose a
    #     fact — the ledger is the record, the sink is a copy of it;
    #   * after the first failure the sink is not touched again for this operation, and its
    #     own failure is never reported through itself (that would recurse);
    #   * `kind` comes from a closed vocabulary and `detail` is an already-safe token: never
    #     an exception object, never `str(exc)`, never a response URL;
    #   * does NOT catch `BaseException`. An interrupt delivered inside the sink is a real
    #     interrupt and keeps propagating; it is already in the ledger by then. Pretending a
    #     non-throwing logger could also swallow an interrupt would be a false guarantee, and
    #     saying so is the honest edge of this one.
    def record_fact(
        self, kind: str, detail: str | None = None, record: Finalization | None = None
    ) -> str:
        if kind in FACT_KINDS:
            fact = kind if detail is None else f"{kind}:{detail}"
        else:
            fact = f"unregistered-fact:{quote_value(kind)}"
        if record is not None:
            record.facts.append(fact)
            if record.sink_failed:
                return fact
            record.sink_attempts += 1
        try:
            self.fact_sink.append(fact)
        except Exception as error:
            if record is not None:
                record.sink_failed = True
                record.cleanup_failures.append(f"sink-failed:{type(error).__name__}")
        return fact

    # ---- staging -------------------------------------------------------
    def stage(self, record: Finalization) -> StagedFile:
        """Create a staged file under a name this process invented.

        `O_EXCL` is the whole symlink defence: measured, `O_WRONLY|O_CREAT` without it writes
        straight through a planted link, and with it the call returns `EEXIST` while the
        victim stays intact — for a dangling link too. The random suffix is why a second
        process cannot guess the name and win it first; the pid is why a human looking at the
        workspace can tell which run left an orphan behind.

        The digest cannot appear in the name here, because at this point no bytes have
        arrived. That is the reason staging and publishing are two names and one link.
        """
        name = f".partial.{os.getpid()}.{os.urandom(8).hex()}"
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self._dir_fd())
        except OSError as error:
            # Nothing is created by this call on any of these paths, so there is no name to
            # hand back and no orphan to record. EEXIST in particular means the entry is not
            # ours, and unlinking it would be deleting something PackProof did not make.
            raise WorkspaceEntryInvalidError(
                f"the staged artifact name {quote_value(name)} could not be created in the "
                f"workspace ({errno_text(error)})"
            ) from None
        record.staged_name = name
        return StagedFile(name, fd)

    def digest_of(self, name: str) -> str:
        """SHA-256 of what is actually on disk under *name*, following no symlink.

        Raises the raw `OSError` rather than a typed artifact error on purpose: its callers
        distinguish "could not read the destination" from "the destination is foreign" by
        catching `OSError`, and wrapping it here would turn an unreadable file into an
        unhandled exception type at exactly the two sites that must not do that.
        """
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._dir_fd())
        try:
            digest = hashlib.sha256()
            while chunk := os.read(fd, ARTIFACT_WRITE_CHUNK_BYTES):
                digest.update(chunk)
            return digest.hexdigest()
        finally:
            os.close(fd)

    def discard(self, name: str) -> None:
        """Unlink a staged name. Only ever a staged name — never a published one.

        A published artifact is content-addressed: another operation, or a later stage, may
        have linked it or be reading it. Nothing in this module deletes a name it did not
        generate for this operation, and the parameter is a staged name by contract.

        `ENOENT` is success. The staged file being already gone is the outcome this call
        wanted, and treating it as a failure would make double cleanup — which the runner
        can perform after an interrupted discard — report an orphan that is not there.
        """
        try:
            os.unlink(name, dir_fd=self._dir_fd())
        except OSError as error:
            if error.errno == os_errno.ENOENT:
                return
            raise

    def discard_staged(self, record: Finalization) -> str:
        """Unlink this operation's staged name, and record an orphan when the unlink fails.

        The one implementation of the `discard-staged` cleanup action, shared by the runner's
        step and by the transport's release path so the two cannot drift into offering a
        different definition of "cleaned up". Only ever a staged name: nothing in this module
        deletes a name it did not generate for this operation.

        `_retire_staged` deliberately stays separate rather than calling this. After a
        successful commit the orphan it records is not a failure anyone has to be told about —
        the download succeeded and the file is only workspace litter — so it must not join
        `cleanup_failures`, which the primary-annotation pass turns into an error clause.
        """
        if record.staged_name is None:
            return "skipped:no-staged-file"
        try:
            self.discard(record.staged_name)
        except OSError as error:
            record.orphaned_temp = True
            record.cleanup_failures.append(f"discard-failed:{errno_text(error)}")
            self.record_fact("staged-orphan", f"{record.staged_name} {errno_text(error)}", record)
            return "failed:discard-failed"
        record.staged_name = None
        return "ok"

    def link_count(self, name: str) -> int:
        """How many directory entries hold these bytes.

        Read rather than assumed because it is the cheapest evidence a test has that a link
        happened exactly once: `2` means a staged name still points at the published inode,
        which means the retire step did not run.
        """
        return os.stat(name, dir_fd=self._dir_fd(), follow_symlinks=False).st_nlink

    # ---- the two seams the tests inject through ------------------------
    def _link(self, record: Finalization, staged: str, final: str) -> None:
        os.link(staged, final, src_dir_fd=self._dir_fd(), dst_dir_fd=self._dir_fd())

    def _fsync_dir(self) -> None:
        fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY, dir_fd=self._dir_fd())
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    # ---- finalization --------------------------------------------------
    def finalize(self, record: Finalization) -> str:
        """Link the staged file to its content-addressed name. The one commit point.

        Three rules organise this function, and they are all measured rather than assumed:

        * `link_started` is written *before* the syscall, because the window between the two
          is where a `KeyboardInterrupt` can land with the destination created and this
          ledger still saying nothing happened. That is `SETTLEMENT_REQUIRED`, and it is the
          state settlement exists for.
        * The commit fact is one assignment to `published_as`, immediately after `os.link`
          returns. A failure of the directory sync afterwards does not undo it, and does not
          become a failed download: the bytes are on the volume, only the durability of the
          *directory entry* is in question.
        * Only `EEXIST` may enter existing-destination verification. Any other errno is a
          plain failure and asking the filesystem about the destination after it would be
          reading an entry this process has no evidence it created.
        """
        if record.staged_name is None:
            raise ArtifactWriteError(
                f"there is no staged artifact for {quote_value(record.ref)} to publish"
            )
        if record.digest is None:
            # Not reachable through the intended caller: the digest is what streaming produces,
            # and publishing is only asked for once a download completed. Checked anyway,
            # because `final` is built from it and a `None.tgz` sitting in a content-addressed
            # workspace would be an artifact named for nothing.
            raise ArtifactWriteError(
                f"no content digest was recorded for {quote_value(record.ref)}, so there is no "
                "content-addressed name this artifact could be published under"
            )
        final = f"{record.digest}.tgz"
        link_errno: int | None = None
        link_error: OSError | None = None

        record.link_started = True
        try:
            self._link(record, record.staged_name, final)
        except OSError as error:
            link_error = error
            link_errno = error.errno if isinstance(error.errno, int) else 0
        if link_error is not None:  # the OBJECT is the predicate, not the number
            if link_errno != os_errno.EEXIST:
                raise self._link_failure(record, final, link_error) from None
            del link_error
            return self._verify_existing(record, final)

        record.published_as = final

        sync_error: OSError | None = None
        try:
            self._fsync_dir()
        except OSError as error:
            sync_error = error
        if sync_error is not None:  # never `if sync_errno:`: errno 0 is falsy and real
            record.synced = False
            self.record_fact("published-unsynced", f"{final} {errno_text(sync_error)}", record)
            raise WorkspaceDurabilityError(
                f"the artifact for {quote_value(record.ref)} is committed as {final} but the "
                f"workspace directory could not be synced ({errno_text(sync_error)}); "
                f"PackProof keeps the artifact and reports state {PUBLISHED_UNSYNCED}",
                state=PUBLISHED_UNSYNCED,
                final_name=final,
            ) from None
        record.synced = True

        self._retire_staged(record)
        self.record_fact("published", final, record)
        return final

    def _link_failure(
        self, record: Finalization, final: str, link_error: OSError
    ) -> ArtifactWriteError:
        record.link_failed = True
        record.published_as = None
        detail = errno_text(link_error)
        message = (
            f"the artifact for {quote_value(record.ref)} could not be published as {final}: "
            f"{detail}"
        )
        if _is_out_of_space(link_error):
            return WorkspaceOutOfSpaceError(message)
        return ArtifactWriteError(message)

    def _verify_existing(self, record: Finalization, final: str) -> str:
        """ONLY `errno.EEXIST` reaches here: decide whether those bytes are already ours.

        Re-opened with `O_NOFOLLOW` and hashed, because "the name exists" and "the name holds
        these bytes" are different claims and only the second one makes a re-download a
        success. A directory, a symlink, a socket or an unreadable file is foreign and stays
        where it is: PackProof did not create the entry, so it does not get to decide what
        happens to it — which is also why this path raises instead of retrying under a new
        name. A content-addressed name is not negotiable; two artifacts with the same digest
        are the same artifact.
        """
        stat_error: OSError | None = None
        st: os.stat_result | None = None
        try:
            st = os.stat(final, dir_fd=self._dir_fd(), follow_symlinks=False)
        except OSError as error:
            stat_error = error
        if stat_error is not None:
            record.link_failed = True
            record.published_as = None
            raise ArtifactWriteError(
                f"the artifact for {quote_value(record.ref)} could not be published as "
                f"{final}: the destination changed under us ({errno_text(stat_error)})"
            ) from None
        if not stat_module.S_ISREG(st.st_mode):
            record.destination_foreign = True
            record.published_as = None
            raise ArtifactCollisionError(
                f"the artifact destination {final} for {quote_value(record.ref)} is held by "
                f"a {self._kind(st.st_mode)}, which PackProof will not replace"
            ) from None
        read_error: OSError | None = None
        existing: str | None = None
        try:
            existing = self.digest_of(final)
        except OSError as error:
            read_error = error
        if read_error is not None:
            record.destination_foreign = True
            record.published_as = None
            raise ArtifactCollisionError(
                f"the artifact destination {final} for {quote_value(record.ref)} is a regular "
                f"file that PackProof could not read to verify it ({errno_text(read_error)})"
            ) from None
        if existing == record.digest:
            record.verified_existing = True
            record.published_as = final
            record.synced = True
            self._retire_staged(record)
            self.record_fact("reused-existing", final, record)
            return final
        record.destination_foreign = True
        record.published_as = None
        raise ArtifactCollisionError(
            f"the artifact destination {final} for {quote_value(record.ref)} holds a different "
            "payload (digest mismatch), which PackProof will not replace"
        ) from None

    @staticmethod
    def _kind(mode: int) -> str:
        """What kind of thing is in the way. A name is not enough to say what to do about it."""
        if stat_module.S_ISDIR(mode):
            return "directory"
        if stat_module.S_ISLNK(mode):
            return "symlink"
        if stat_module.S_ISFIFO(mode):
            return "fifo"
        if stat_module.S_ISSOCK(mode):
            return "socket"
        if stat_module.S_ISBLK(mode):
            return "block device"
        if stat_module.S_ISCHR(mode):
            return "character device"
        return "non-regular file"

    def _retire_staged(self, record: Finalization) -> None:
        """Unlink the staged name after a successful commit, and record an orphan if we can't.

        Failure here is deliberately not an error to raise: the artifact is published. A
        leftover `.partial.*` costs the user a file; replacing a success with an exception
        because of it would cost them the download they have.
        """
        if record.staged_name is None:
            return
        unlink_error: OSError | None = None
        try:
            self.discard(record.staged_name)
        except OSError as error:
            unlink_error = error
        if unlink_error is not None:
            record.orphaned_temp = True
            self.record_fact(
                "staged-orphan", f"{record.staged_name} {errno_text(unlink_error)}", record
            )
            return
        record.staged_name = None

    # ---- settlement ----------------------------------------------------
    def settle(self, record: Finalization) -> str | None:
        """Ask the filesystem what the interrupted link did, without writing a byte.

        Read-only, content-addressed, idempotent, no unlink and no write syscall. That is
        what makes it safe to retry once after an interruption, and safe under concurrent
        downloads — and it is why settlement can recover a publication this ledger never
        finished recording.
        """
        if record.published_as is not None:
            return record.published_as
        if not record.outcome_undetermined:
            return None
        record.settlement_attempts += 1
        final = f"{record.digest}.tgz"
        st: os.stat_result | None = None
        stat_error: OSError | None = None
        try:
            st = os.stat(final, dir_fd=self._dir_fd(), follow_symlinks=False)
        except OSError as error:
            if error.errno == os_errno.ENOENT:
                # Absence is a DEFINITIVE answer, not an undetermined one.
                record.link_failed = True
                record.settlement_note = "absent"
                self.record_fact("settled-absent", final, record)
                return None
            stat_error = error
        if stat_error is not None:
            record.settlement_note = f"stat-failed:{errno_text(stat_error)}"
            self.record_fact("settled-undetermined", f"{final} {record.settlement_note}", record)
            raise ArtifactSettlementError(
                f"PackProof started publishing {quote_value(record.ref)} but could not "
                f"determine whether {final} exists ({errno_text(stat_error)}); the workspace "
                "needs reconciliation"
            ) from None
        if not stat_module.S_ISREG(st.st_mode):
            record.destination_foreign = True
            record.settlement_note = "foreign"
            self.record_fact("settled-foreign", f"{final} {self._kind(st.st_mode)}", record)
            return None
        read_error: OSError | None = None
        existing: str | None = None
        try:
            existing = self.digest_of(final)
        except OSError as error:
            read_error = error
        if read_error is not None:
            record.destination_foreign = True
            record.settlement_note = f"unreadable:{errno_text(read_error)}"
            self.record_fact("settled-unreadable", final, record)
            return None
        if existing != record.digest:
            record.destination_foreign = True
            record.settlement_note = "digest-mismatch"
            self.record_fact("settled-foreign", f"{final} digest mismatch", record)
            return None
        # The bytes for this operation's digest are on disk. Whether this operation linked
        # them or a concurrent one with the same digest did, the artifact is ours to hand
        # back — but we did not sync the directory, so the report stays pessimistic.
        record.published_as = final
        record.recovered = True
        record.synced = False
        record.settlement_note = "recovered"
        self.record_fact("settled-published", final, record)
        return final


# --------------------------------------------------------------------------- cleanup


#: Artifact errors the runner may complete with a cleanup clause. Deliberately the
#: publication family: a transport failure has no staged-file clause to add, because the
#: transport path builds its own message from the outcome it reduced.
ARTIFACT_ERRORS = (
    ArtifactWriteError,
    ArtifactCollisionError,
    WorkspaceDurabilityError,
    ArtifactSettlementError,
)

#: The cleanup steps, in the order they must be attempted. Named as a constant so the
#: tests can prove the runner does nothing but loop over this tuple — and so the bound on
#: absorbed events is `len(CLEANUP_STEPS)` rather than a promise.
CLEANUP_STEPS = ("settle", "discard-staged", "account-published", "annotate-primary")


def note_ledger(record: Finalization, kind: str, detail: str) -> str:
    """A ledger-only write, built from fixed strings and class names.

    The handlers that protect the primary's propagation slot may only do work that cannot
    fail. `store.record_fact()` cannot be used there: it touches the sink, and a sink that
    raises is a NEW event competing for the very slot the handler is defending. A plain
    `list.append` on the caller's own record is not a failure site. The sink still receives
    every primary fact through `record_fact`; only the secondary events stay ledger-side.
    """
    fact = f"{kind}:{detail}"
    record.facts.append(fact)
    return fact


def in_flight() -> BaseException | None:
    """The exception the interpreter is currently propagating through this frame.

    Measured on 3.11.15 / 3.12.13 / 3.13.6 / 3.14.4: inside a `finally`, `sys.exc_info()`
    DOES report the exception causing the unwinding, so the runner can name the primary.
    Used for two things only — the class name in the ledger, and the decision whether
    anything else already owns the propagation slot. The object itself is never formatted,
    never stored on the record and never raised here: to preserve it, the runner simply does
    not raise anything.
    """
    return sys.exc_info()[1]


def _run_steps(
    store: ArtifactStore,
    record: Finalization,
    held: BaseException | None,
    primary: BaseException | None,
) -> BaseException | None:
    """One loop over four named steps, and the primary keeps its slot.

    Three outcomes per step:

    * it returns `'ok'` or `'skipped:...'` — recorded as the step's outcome;
    * it raises an `Exception` — an ordinary bug in cleanup: recorded and skipped. It must
      never become the story of the operation;
    * it raises a `BaseException` — recorded in `interrupts_seen`. If nothing else owns the
      slot (`primary is None`) the FIRST one is returned as `deferred` for the caller to
      re-raise, because an interrupt must not turn an operation into a success. If a primary
      does own the slot it is recorded in `displaced` and dropped: the caller sees the
      primary, which is what the precedence policy asks for.

    `deferred` is a single slot, the loop runs a fixed four steps, and there is no retry
    anywhere in this function — so the number of events it can absorb is bounded by
    `len(CLEANUP_STEPS)`, and no sequence of interruptions can make it loop forever.
    """
    deferred: BaseException | None = None
    for step in CLEANUP_STEPS:
        try:
            ran = _step(store, record, held, step)
        except BaseException as error:  # precedence, not capture
            name = type(error).__name__  # the CLASS NAME only: never str(error)
            if isinstance(error, Exception):
                record.steps.append((step, f"failed:{name}"))
                record.cleanup_failures.append(f"step-failed:{step}:{name}")
                note_ledger(record, "step-failed", f"{step}:{name}")
                continue
            record.steps.append((step, f"interrupted:{name}"))
            record.interrupts_seen.append(name)
            if primary is None and deferred is None:
                deferred = error
            else:
                record.displaced.append(f"{step}:{name}")
                note_ledger(record, "displaced", f"{step}:{name}")
            continue
        record.steps.append((step, ran))
    # If the staged file is still named here, no step retired it: the discard either failed
    # or was interrupted midway. The file is in the workspace and belongs to this operation,
    # so the ledger must not read as though cleanup had finished. Ledger-only, because this
    # runs inside a `finally` whose purpose is to protect a slot from new events.
    if record.staged_name is not None and not record.orphaned_temp:
        record.orphaned_temp = True
        record.cleanup_failures.append("staged-orphan:cleanup-incomplete")
        note_ledger(record, "staged-orphan", f"{record.staged_name} cleanup-incomplete")
    return deferred


def _step(store: ArtifactStore, record: Finalization, held: BaseException | None, step: str) -> str:
    """Each step returns `'ok'` or `'skipped:<reason>`.

    Skipping is a decision made from the ledger, never from a guess about the filesystem.
    """
    if step == "settle":
        if record.published_as is not None:
            return "skipped:already-committed"
        if not record.outcome_undetermined:
            return "skipped:outcome-determined"
        try:
            store.settle(record)
        except BaseException as interrupt:
            # "Incomplete" has to mean "the ledger still cannot say what happened", not "the
            # call was interrupted": settle() writes its verdict into the record *before* it
            # records the fact, so an interrupt landing on that sink call leaves a complete
            # verdict behind, and flagging it incomplete would contradict `state`.
            record.settlement_incomplete = record.outcome_undetermined
            store.record_fact("settle-interrupted", type(interrupt).__name__, record)
            if isinstance(interrupt, Exception):
                # A settlement VERDICT failure — the stat could not be performed at all — is
                # not something a retry can fix, so it goes back out to the runner.
                raise
            # A process-control interrupt is exactly what one retry can fix, and the retry is
            # safe because settlement only reads, hashes and never unlinks.
            store.record_fact("settle-retry", record.digest[:12], record)
            store.settle(record)  # may itself raise; the runner defers it
            record.settlement_incomplete = record.outcome_undetermined
            # The retry repairs the LEDGER, not the signal. A SystemExit caught here and never
            # re-raised would be silently converted into an ordinary outcome, which the brief
            # forbids: it goes back out by identity, the same way CPython re-raises a second
            # SIGINT that arrives during cleanup.
            raise interrupt
        return "ok"
    if step == "discard-staged":
        return store.discard_staged(record)
    if step == "account-published":
        if record.published_as is None:
            if record.settlement_incomplete:
                record.settlement_note = record.settlement_note or "interrupted-before-verdict"
                return "skipped:settlement-incomplete"
            return "skipped:nothing-published"
        if record.accounted:
            return "skipped:already-accounted"
        record.accounted = True  # ledger fact FIRST
        store.record_fact(
            "aborted-but-published",
            f"{record.published_as} state={record.state} recovered={record.recovered}",
            record,
        )
        return "ok"
    if step == "annotate-primary":
        return annotate_primary(record, held)
    return "skipped:unknown-step"


def annotate_primary(record: Finalization, held: BaseException | None) -> str:
    """Complete the operational primary with what cleanup found. Idempotent by design.

    Called twice on purpose: once as the `annotate-primary` step, and once from
    `finalize_artifact`'s tail after the loop has finished. The second call is not decoration
    — the `staged-orphan` note is written when the loop is already over, and a discard that
    was *interrupted* rather than failing with an `OSError` leaves `cleanup_failures` empty
    while the step runs. Without the tail call the caller would receive an error that says
    nothing about the file the workspace still holds, and the record never reaches them: the
    exception is the only channel.

    Plain attribute and list work on the caller's own objects: no sink, no raise, nothing
    that could displace the exception about to escape.
    """
    if held is None:
        return "skipped:no-primary"
    if not isinstance(held, ARTIFACT_ERRORS):
        return "skipped:not-an-artifact-error"
    if (
        record.orphaned_temp
        and record.staged_name is not None
        and not getattr(held, "staged_left_behind", False)
        and isinstance(held.args, tuple)
        and held.args
        and isinstance(held.args[0], str)
    ):
        held.args = (
            held.args[0] + f"; the staged file {record.staged_name} was left in the workspace",
        )
        held.staged_left_behind = True
    if record.cleanup_failures:
        held.cleanup_failures = list(record.cleanup_failures)
    return "ok"


def finalize_artifact(store: ArtifactStore, record: Finalization) -> str:
    """Publish `record`, and clean up without ever replacing the first exception.

    The design's publication driver. Named for what it does rather than `download`, which is
    the transport half's public verb: nothing here reads a socket, and nothing that sees a
    URL is in scope.

    Precedence, in one sentence: **the first exception event of the operation escapes, by
    identity, and every later event is recorded.** Achieved by construction, not bookkeeping
    — when a primary exists, this `finally` contains no reachable `raise`, and the tail
    re-raises the object it caught rather than any substitute for it.
    """
    held: BaseException | None = None
    succeeded = False
    try:
        final = store.finalize(record)
        succeeded = True
        return final
    except Exception as error:  # NOT BaseException: an interrupt is not ours to reduce
        held = error
        # Wrapped, because the recording below touches the sink. If a signal lands on that
        # call while this clause is still handling `held`, the new exception would take the
        # slot AND the operation would report the interrupt instead of the artifact error.
        # The wrapped event is recorded ledger-only and the handler completes normally, so
        # `held` keeps precedence via the raise at the bottom.
        try:
            record.pending_failure = type(error).__name__
            store.record_fact("operation-failed", type(error).__name__, record)
        except BaseException as secondary:  # recorded, never propagated
            name = type(secondary).__name__
            record.interrupts_seen.append(name)
            record.displaced.append(f"in-handler:{name}")
            note_ledger(record, "displaced", f"in-handler:{name}")
    finally:
        if not succeeded:
            # The slot owner: the caught error, else whatever the interpreter is still
            # carrying through this finally (an interrupt the clause above never saw).
            primary = held if held is not None else in_flight()
            record.primary_kind = type(primary).__name__ if primary is not None else "none"
            note_ledger(record, "primary-in-flight", record.primary_kind)
            deferred = _run_steps(store, record, held, primary)
            # Only when NOTHING owns the slot does a cleanup interrupt get raised. When a
            # primary exists this raises nothing at all, which is precisely what lets the
            # primary escape by identity.
            if deferred is not None and primary is None:
                raise deferred  # the exact original object, by identity
    if held is not None:
        annotate_primary(record, held)
        raise held
    # There is no normal exit reaching here: `succeeded` is only True after `return final`
    # inside the try, so a trailing `return None` would be dead code that reads like a success
    # path. An orphaned staged file from a *successful* publish is already recorded as
    # `staged-orphan` and is deliberately not swept here — the orphan sweep is deferred, and
    # deleting names this process did not create is its own risky operation.
