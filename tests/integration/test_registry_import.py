"""Importing the registry layers must not do anything.

`packproof.registry` is imported by the CLI, by tests, and by anything that only wants to *parse*
a reference — so a module-level `build_opener()` would read proxy settings from the environment at
import, and a module-level request would be exactly the network side effect this stage promises not
to have. Neither is visible from inside a running test process, where the module was already
imported: the proof needs a fresh interpreter with an audit hook installed before the import, which
is why this file is in the integration tier and runs a subprocess.

The hook counts three families: anything socket-shaped, any file opened for writing, and any change
to the filesystem or start of a process. A package that only parses text and validates models
produces none of them.

Three counters that read `0 0 0` are a claim about an instrument as much as about a package, so the
rest of this file aims the same hook — the same source text, not a copy of it — at calls that must
register. A detector that cannot see anything prints the same numbers as a clean import, and only
the controls tell the two apart. Every name in either detection set was fired on CPython 3.13.6
before it was written down, and the names that did not fire were left out. What differs is how the
permanent control fires them:

* **The mutation names are fired by the calls themselves.** Each is a syscall the control can make
  inside its own temporary directory, so that control proves the detector against the real thing.
* **`urllib.Request` is fired by a real call too**, through urllib's own `file://` handler. That is
  the same `OpenerDirector.open` a registry fetch goes through, with no socket in it. The matching
  negative is asserted below as well: constructing a `Request` fires nothing.
* **The four socket-stack names are raised for the hook, not dialled.** A test in this stage may not
  reach DNS or open a socket, so `sys.audit` fires `http.client.connect`, `socket.__new__`,
  `socket.connect` and `socket.getaddrinfo`, which proves the hook records and reports each name it
  watches. That is a narrower claim than the two above, and it needs its other half: a scratch fetch
  of `http://127.0.0.1:1` — numeric, so no resolver runs, and refused by the kernel, so no packet
  leaves this machine — fires exactly `urllib.Request, http.client.connect, socket.getaddrinfo,
  socket.__new__, socket.connect`, in that order, on 3.11.15, 3.12.13, 3.13.6 and 3.14.4. This file
  does not re-run that fetch; it proves the detector can see the names, and records that a real
  connection produces them.

Four measured facts about the names that did not fire, kept because leaving a name out is only
honest if the reason is written down:

* **`os.replace` fires `os.rename`.** There is no `os.replace` event, so watching for one would be
  watching for nothing while a real `os.replace` went unreported under a name that looks covered.
* **`pathlib` fires `os.*`.** `Path.mkdir()` and `Path.unlink()` are `os.mkdir` and `os.remove`.
* **`os.spawnl` / `os.spawnv` fire `os.fork` and `os.exec`,** not `os.spawn`. `os.spawn` is left
  out, and the two names that caught the spawn calls are in.
* **`socket.send` and `socket.sendall` were silent** — on an `AF_UNIX` socketpair and on a
  connected `AF_INET` datagram. They are left out, and their absence costs no detection: a socket
  this process did not create cannot send through it, and `socket.__new__` fires for the creation.

The claim is about writes and changes, not about reading. Python's import machinery opens every
`.py` and `.pyc` it loads, and `os.scandir` fires for namespace packages, so a test that forbade
`open` outright would fail on the interpreter rather than on PackProof. The children also run with
`-B`: a bytecode cache invalidated by an unrelated edit is the interpreter writing its own files,
and letting that count as a package side effect would make this test fail for reasons it is not
about — or, worse, train whoever runs it to expect a failure that means nothing.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_SECONDS = 60

#: Events that mean a socket was touched, or an HTTP request was accepted by an opener. The
#: construction half of that distinction is asserted below rather than trusted: `urllib.Request`
#: fires when `OpenerDirector.open` accepts a request and not when the `Request` is built.
_NETWORK_EVENTS = (
    "http.client.connect",
    "socket.__new__",
    "socket.connect",
    "socket.getaddrinfo",
    "urllib.Request",
)

#: The subset of `_NETWORK_EVENTS` that only the socket stack raises. The control fires these with
#: `sys.audit`, because no call this suite may make — reach DNS or open a socket — is allowed to
#: produce them. `urllib.Request` is the one that is *not* here, and so is proved by a real call;
#: the equality that keeps the two tuples covering exactly one set is asserted in the control.
_RAISED_NETWORK_EVENTS = (
    "http.client.connect",
    "socket.__new__",
    "socket.connect",
    "socket.getaddrinfo",
)

#: Events that either change the filesystem or start a process. `os.exec`, `os.fork`, `os.system`
#: and `os.posix_spawn` are the Stage 2.3B promise that nothing a package ships is ever run: an
#: import that fired one of them would be executing, not describing.
_MUTATION_EVENTS = (
    "os.chmod",
    "os.chdir",
    "os.exec",
    "os.fork",
    "os.link",
    "os.mkdir",
    "os.posix_spawn",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.symlink",
    "os.system",
    "os.truncate",
    "shutil.rmtree",
)

#: Text-mode `open` modes that create or change something. Measured: the audit event reports the
#: normalised letter, so `open(path, "wb")` arrives as `"w"`.
_WRITE_MODES = ("w", "a", "x", "+")

#: The Stage 2.3B modules, named so a cold import cannot accidentally skip one by being renamed.
#: `packproof.registry` re-exports the clients, so the first line already pulls most of these in;
#: importing each by name is what makes "this module touched the network" a statement about it.
_STAGE_23_MODULES = (
    "packproof.registry",
    "packproof.registry.npm",
    "packproof.registry.npm_fetcher",
    "packproof.registry.npm_download",
    "packproof.registry.artifact_store",
    "packproof.registry.download_policy",
    "packproof.registry.errors",
    "packproof.schemas",
    "packproof.schemas.models",
    "packproof.schemas.artifacts",
)

#: What the clean import is supposed to produce, in the order the child prints it.
_ZERO = (0, 0, 0)


@dataclass(frozen=True)
class _ChildReport:
    """The three counters and the records behind them, exactly as one child reported them.

    `names` holds file paths for the write family and event names for the other two, because that
    is what each counter can say usefully about itself: which file, or which syscall.
    """

    counts: tuple[int, int, int]
    network: tuple[str, ...]
    writes: tuple[str, ...]
    mutations: tuple[str, ...]

    @property
    def summary(self) -> str:
        return (
            f"network={list(self.network)} writes={list(self.writes)} "
            f"mutations={list(self.mutations)}"
        )


def _child_script(body: str, root: Path) -> str:
    """One child source file: preamble, hook, caller's body, report.

    The hook lives here and nowhere else. Every test in this file runs this same text, so a control
    that proves the detector fires is proof about the instrument the import test actually used —
    and not about a second copy that could drift from it.
    """
    return "\n".join(
        [
            "import os",
            "import sys",
            f"ROOT = {str(root)!r}",
            f"NETWORK_EVENTS = {_NETWORK_EVENTS!r}",
            f"RAISED_NETWORK_EVENTS = {_RAISED_NETWORK_EVENTS!r}",
            f"MUTATION_EVENTS = {_MUTATION_EVENTS!r}",
            f"WRITE_MODES = {_WRITE_MODES!r}",
            # Computed in the child, from that interpreter's own `os` constants: the flag bits are
            # platform-specific (`os.O_EXCL` is 0x800 on macOS and 0x2 on Linux), and a list of
            # numbers copied from one machine would be wrong on the other.
            "WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND",
            "network, writes, mutations = [], [], []",
            "def hook(event, args):",
            "    if event in NETWORK_EVENTS:",
            "        network.append(event)",
            "    elif event in MUTATION_EVENTS:",
            "        mutations.append(event)",
            "    elif event == 'open':",
            # Measured: `io.open(path, 'wb')` reports the mode letter at args[1] and the flag bits
            # at args[2], while `os.open(path, O_WRONLY | O_CREAT, 0o600)` reports None at args[1]
            # and the flag bits at args[2]. Reading only the string would make every staged file
            # invisible, because Stage 2.3B stages with `os.open`.
            "        mode = args[1] if len(args) > 1 else None",
            "        flags = args[2] if len(args) > 2 else 0",
            "        writing = isinstance(flags, int) and bool(flags & WRITE_FLAGS)",
            "        if isinstance(mode, str):",
            "            writing = writing or any(character in mode for character in WRITE_MODES)",
            "        if writing:",
            # The whole path, not a prefix: a length cap measured earlier in this session truncated
            # two different staged names to the same directory, so the control asserted on a value
            # that identified neither file it claimed to have seen. Only the two characters the
            # report format uses as separators are replaced.
            "            name = str(args[0])",
            "            writes.append(name.replace('|', '!').replace('\\n', ' '))",
            "sys.addaudithook(hook)",
            body,
            "print(len(network), len(writes), len(mutations))",
            "print('|'.join(sorted(set(network))))",
            "print('|'.join(writes))",
            "print('|'.join(sorted(set(mutations))))",
        ]
    )


def _run_child(body: str, root: Path) -> _ChildReport:
    """Run one child and read its four report lines back.

    A non-zero exit is a failure of the test itself rather than of the claim: the body must reach
    the prints, so an exception part-way through a control body is reported here with its traceback
    instead of showing up as a suspiciously clean `0 0 0`.
    """
    result = subprocess.run(  # noqa: S603  (this interpreter, `-B`, and one literal script built
        # from constants in this file plus a pytest-owned directory; nothing derived from a scanned
        # package reaches it)
        [sys.executable, "-B", "-c", _child_script(body, root)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, f"the child failed:\n{result.stderr}"
    lines = result.stdout.splitlines()
    assert len(lines) >= 4, f"the child printed no report: {result.stdout!r}"
    counts, network, writes, mutations = lines[:4]
    return _ChildReport(
        counts=tuple(int(value) for value in counts.split()),
        network=tuple(network.split("|")) if network else (),
        writes=tuple(writes.split("|")) if writes else (),
        mutations=tuple(mutations.split("|")) if mutations else (),
    )


_IMPORT_BODY = "\n".join(f"import {name}" for name in _STAGE_23_MODULES)

# Two writes through the two APIs, then two reads through the same two APIs: the count has to come
# out at exactly two, which is what proves the reads are not being counted and the staged-file form
# is. A control that only asserted "the write was seen" would pass even if every read were too.
_WRITE_CONTROL_BODY = "\n".join(
    [
        "builtins_target = os.path.join(ROOT, 'builtins-open.bin')",
        "handle = open(builtins_target, 'wb')",
        "handle.write(b'bytes, as a report would name them')",
        "handle.close()",
        "staged_target = os.path.join(ROOT, 'os-open.bin')",
        "descriptor = os.open(staged_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)",
        "os.write(descriptor, b'the way Stage 2.3B stages a file')",
        "os.close(descriptor)",
        "reader = open(builtins_target, 'rb')",
        "reader.close()",
        "read_descriptor = os.open(staged_target, os.O_RDONLY)",
        "os.close(read_descriptor)",
    ]
)

# One call per name in `_MUTATION_EVENTS`. `os.exec` is the awkward one: a successful exec replaces
# the process image and takes the record with it, so the control execs a program that does not
# exist — the audit fires first, the syscall then fails, and the append survives in a process that
# carries on to print its report. `os.fork` is last and the child exits immediately, so only the
# parent reaches the prints and the report is not duplicated.
_MUTATION_CONTROL_BODY = "\n".join(
    [
        "import pathlib",
        "import shutil",
        "made = os.path.join(ROOT, 'made')",
        "os.mkdir(made)",
        "os.rmdir(made)",
        "pathlib_made = pathlib.Path(made)",
        "pathlib_made.mkdir()",
        "pathlib_made.rmdir()",
        "pathlib_target = pathlib.Path(os.path.join(ROOT, 'pathlib-unlink.bin'))",
        "pathlib_target.write_bytes(b'unlinked a moment later')",
        "pathlib_target.unlink()",
        "first = os.path.join(ROOT, 'first.bin')",
        "second = os.path.join(ROOT, 'second.bin')",
        "descriptor = os.open(first, os.O_WRONLY | os.O_CREAT, 0o600)",
        "os.write(descriptor, b'two artifacts so link and rename have something to move')",
        "os.close(descriptor)",
        "os.link(first, second)",
        "os.symlink(first, os.path.join(ROOT, 'a-link'))",
        "os.chmod(second, 0o644)",
        "os.truncate(second, 4)",
        "os.rename(first, os.path.join(ROOT, 'renamed.bin'))",
        "os.replace(second, os.path.join(ROOT, 'replaced.bin'))",
        "os.remove(os.path.join(ROOT, 'renamed.bin'))",
        "os.unlink(os.path.join(ROOT, 'replaced.bin'))",
        "os.mkdir(os.path.join(ROOT, 'tree'))",
        "shutil.rmtree(os.path.join(ROOT, 'tree'))",
        "os.chdir(ROOT)",
        "os.system('true')",
        # `/bin/sh` rather than a `true` binary: this macOS has no `/bin/true`, and the control's
        # job is to fire the event, not to prove where coreutils was installed.
        "os.posix_spawn('/bin/sh', ['/bin/sh', '-c', 'true'], os.environ)",
        "try:",
        "    os.execl('/bin/pp-not-a-real-program', '/bin/pp-not-a-real-program')",
        "except OSError:",
        "    pass",
        "child = os.fork()",
        "if child == 0:",
        "    os._exit(0)",
        "os.waitpid(child, 0)",
    ]
)

# The network family, proved two ways. `urllib.Request` fires from a real call: `file://` is handled
# by urllib's own `FileHandler`, so the request reaches `OpenerDirector.open` — the same call a
# registry fetch goes through — with no socket and no resolver lookup in it, and `/dev/null` opens
# read-only so the write counter stays clean. The four names below it belong to the socket stack,
# and this stage's brief forbids a test that opens a socket or asks for a name, so they are raised
# for the hook instead of dialled. What the split costs in directness is recorded in the module
# docstring, along with the scratch measurement that a real refused connection to `127.0.0.1:1`
# produces exactly these five names on all four supported interpreters.
_NETWORK_CONTROL_BODY = "\n".join(
    [
        "import urllib.request",
        # A literal `file://` path, no host, no name to resolve; the URL string is not a target.
        "handle = urllib.request.urlopen('file:///dev/null', timeout=5)",
        "handle.close()",
        "for name in RAISED_NETWORK_EVENTS:",
        "    sys.audit(name, 'pp-detector-control')",
    ]
)

# The other half of the same fact: a `Request` that never reaches an opener is invisible to this
# detector. Without the negative, "no `urllib.Request` fired during import" would be evidence about
# a name that might be raised by anything, and a module that built requests at import — wrong, but
# silently — would still pass. The host sits under the `.invalid` TLD that RFC 2606 reserves and the
# DNS is required never to define, and construction performs no lookup in any case.
_REQUEST_CONSTRUCTION_BODY = "\n".join(
    [
        "import urllib.request",
        "urllib.request.Request('https://registry.invalid/pp-detector-negative')",
    ]
)


def test_importing_the_registry_opens_no_sockets_writes_no_files_and_creates_nothing(
    tmp_path: Path,
) -> None:
    """The Stage 2.3 packages, imported cold: zero socket events, zero writes, zero mutations.

    `packproof.registry` re-exports the clients, so importing it is the widest thing a cold import
    can pull in — if the fetcher had a module-level opener, or the artifact store a module-level
    workspace, this is where it would fire. The three controls below are what make the zeros
    evidence rather than an absence of instrumentation.
    """
    report = _run_child(_IMPORT_BODY, tmp_path)

    assert report.counts == _ZERO, report.summary


def test_the_write_detector_sees_a_staged_file_and_ignores_a_read(tmp_path: Path) -> None:
    """`os.open(path, O_WRONLY | O_CREAT | O_EXCL)` is counted; `O_RDONLY` and `'rb'` are not.

    This is the control for the counter that matters most to Stage 2.3B, because the publication
    path stages exactly that way. Without the flag-bit read, a write opened through `os.open`
    arrives with `None` where the mode should be and the counter stays at zero forever — a test that
    passes whether or not the module it guards ever wrote a file.
    """
    report = _run_child(_WRITE_CONTROL_BODY, tmp_path)

    assert report.counts[0] == 0, f"the write control touched a socket: {report.summary}"
    assert report.counts[2] == 0, f"the write control mutated the filesystem: {report.summary}"
    assert [Path(path).name for path in sorted(report.writes)] == [
        "builtins-open.bin",
        "os-open.bin",
    ], f"expected exactly the two writes, got {report.summary}"


def test_every_mutation_event_the_detector_watches_can_actually_fire(tmp_path: Path) -> None:
    """No name in `_MUTATION_EVENTS` is dead weight.

    Each is fired by a call that should produce it, and the test asserts the set the child saw
    contains every name in the tuple. A name that cannot fire would let "the set was empty" report a
    protection that was never there, so this is the test that makes the zeros above mean something.
    It also asserts no socket event slipped in, because a control that drifted into the network
    family would be a control that no longer tested what its name says.
    """
    report = _run_child(_MUTATION_CONTROL_BODY, tmp_path)

    assert report.counts[0] == 0, f"the mutation control touched a socket: {report.summary}"
    unseen = sorted(set(_MUTATION_EVENTS) - set(report.mutations))
    assert unseen == [], f"{unseen} never fired, so the detector cannot detect it: {report.summary}"


def test_the_detector_records_every_network_name_it_watches(tmp_path: Path) -> None:
    """No name in `_NETWORK_EVENTS` is dead weight, and the control writes and changes nothing.

    One of the five names comes from a real call: urllib's `file://` opener, which opens no socket.
    The other four are raised with `sys.audit`, because this stage forbids a test that touches the
    network. The split is asserted to be exhaustive, so the two tuples cannot drift into covering
    something the import test never watches, or leaving a name controlled by nobody, and the raw
    event count holds the injection to exactly one event per name.
    """
    watched = tuple(sorted(_RAISED_NETWORK_EVENTS)) + ("urllib.Request",)
    assert watched == _NETWORK_EVENTS, "the two tuples no longer cover the watched names"

    report = _run_child(_NETWORK_CONTROL_BODY, tmp_path)

    assert report.counts[1] == 0, f"the network control wrote a file: {report.summary}"
    assert report.counts[2] == 0, f"the network control changed the filesystem: {report.summary}"
    unseen = sorted(set(_NETWORK_EVENTS) - set(report.network))
    assert unseen == [], f"{unseen} never fired, so the detector cannot detect it: {report.summary}"
    # The raw count, not the deduplicated list. The body injects each of the four socket names once
    # and the `file://` open raises `urllib.Request` once, so a sixth event would mean this control
    # reached the socket stack for real — what the brief forbids, and what the two assertions above
    # cannot see: a genuine `socket.__new__` beside the injected one still leaves the reported set
    # complete. Measured, not argued: adding a real `socket.socket()` to this body takes the count
    # to 6 while `unseen` stays empty.
    assert report.counts[0] == len(_NETWORK_EVENTS), (
        f"{report.counts[0]} network events for {len(_NETWORK_EVENTS)} names, "
        f"so one fired twice: {report.summary}"
    )


def test_a_request_that_never_reached_an_opener_is_invisible_to_the_detector(
    tmp_path: Path,
) -> None:
    """The known limit of the network counter, recorded as a test.

    Constructing a `Request` raises no audit event at all, so this detector cannot see a module that
    builds requests at import time — it can only see a module that hands one to an opener. That is
    the right trade for this stage, because the promise under test is "importing performs no
    network operation", and a constructed-but-unsent request performs none. Saying so here is what
    stops a future reader from reading the zeros above as proof that no request object was ever
    made.
    """
    report = _run_child(_REQUEST_CONSTRUCTION_BODY, tmp_path)

    assert report.counts == _ZERO, (
        f"constructing a request should be silent, and the negative must stay silent to mean "
        f"anything: {report.summary}"
    )
