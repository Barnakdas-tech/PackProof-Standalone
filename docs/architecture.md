# PackProof architecture

How the pieces fit, what each layer owns, and which rules every future module must
follow. Written for someone who has just cloned the repository.

## The invariant everything else depends on

PackProof produces **evidence**, and evidence must be reproducible.

Deterministic code — hashing, file inventory, comparison, archive validation,
provenance checks, risk scoring — generates every finding. The AI layer only explains
findings that already exist. A model cannot create, downgrade, or dismiss a finding.

If a future feature cannot state which byte, hash, or attestation it is based on, it
does not belong in the engine.

## Layer model

Dependencies point downward only. A lower layer must never import a higher one.

```
                    cli/            parsing + presentation, no policy
                      |
        reporting/  ai/  provenance/  detectors/  analyzer/     the engine
                      |
                  registry/          npm reference parsing, registry metadata retrieval
                                     and artifact download; source-host communication
                                     lands here later
                      |
              schemas/  config/  utils/     shared, dependency-free foundations
```

### Implemented today

| Module | Owns | Must not |
|---|---|---|
| `src/packproof/__init__.py` | The authoritative `__version__` literal | Import anything (keeps `import packproof` side-effect free) |
| `src/packproof/__main__.py` | `python -m packproof` | Contain logic; it calls `cli.main:main` |
| `src/packproof/cli/main.py` | Typer app, global flags, help text, exit codes | Analysis, network, filesystem writes |
| `src/packproof/config/settings.py` | Validated limits: HTTP timeout, the metadata-response ceiling (`max_metadata_bytes`) and the archive ceiling (`max_download_bytes`), workspace dir and the workspace security policy | Secrets in plain attributes; implicit env binding; creating the workspace |
| `src/packproof/utils/logging.py` | One `configure_logging()` entry point | `basicConfig()`, per-module handler setup |
| `src/packproof/utils/__init__.py` | A home for genuinely shared helpers | Business logic in disguise |
| `src/packproof/schemas/models.py` | `PackageRef`: the shared reference model — name, optional scope, one exact version — with its grammar enforced on the fields so every entry route is bound by it. Those rules are npm's today because npm is the only registry implemented, and the grammar is a subset of node-semver's: exact and unprefixed, with no normalization. `PackageVersionMetadata`, `PackageDistribution` and `PackageRepository` are the shape of a registry's answer: text-only fields, an SRI integrity value, one HTTPS tarball URL, and no field that means "trusted" | Import from `registry/` or any higher layer; widen a field beyond its allowlist |
| `src/packproof/registry/npm.py` | Turning a typed reference into a validated `PackageRef`; npm's spelling of a reference — `@scope/name@version`, dist-tags, ranges — is decided here | Network, filesystem or code execution; restating the grammar the model owns |
| `src/packproof/registry/npm_fetcher.py` | One bounded HTTPS `GET` of a package document per reference, the version selected from that document locally, and the answer projected into validated `schemas/` models or refused as one of the named `registry/errors.py` categories. It also owns `registry_opener()`, the transport chain both registry clients run through: one approved origin, no redirect following, no environment proxy | Downloading or opening a tarball; following a URL that came from inside a response; logging a response body; network or filesystem work at import time |
| `src/packproof/registry/download_policy.py` | Turning the registry's untrusted `dist.tarball` text into a target this process is willing to connect to, or refusing it. The authority is approved by exact string equality against the configured host — not by the parsed `hostname`, which lowercases and strips userinfo — and the request URL is then rebuilt from the approved components, so the value that was checked is the value that gets dialled. Every refusal is a named code, not a sentence | Opening a socket; approving a URL because the model could parse it; putting a refused or signed URL into the refusal it produces |
| `src/packproof/registry/artifact_store.py` | The filesystem half. Staging under a name this process invented (`.partial.<pid>.<random hex>`, created `O_EXCL`, private mode), hashing while writing, publishing by `os.link` of the staged file to `<sha256>.tgz`, syncing the directory, and one ledger per operation whose state is derived from its fields rather than stored. `settle()` recovers a publication the ledger never finished recording | Renaming over a destination — `os.replace` swaps a symlink's name and takes no `dir_fd`; inspecting the destination for anything but `EEXIST`; taking a name from a URL or an archive member; extracting |
| `src/packproof/registry/npm_download.py` | Asking an approved target for the artifact bytes over the shared `registry_opener()` chain, bounding the body by the bytes that actually arrive, checking **every** integrity claim the metadata supplied rather than only the strongest, and returning a `PackageArtifact` only for bytes that hold. A destination that already carries these bytes is reused, and a destination carrying other bytes is refused and left alone | Publishing bytes it could not check; naming the signed URL in a message, a `repr()`, a log line or a returned object; decompressing, extracting or executing anything |
| `src/packproof/schemas/artifacts.py` | `PackageArtifact` and `ArtifactVerification`: what PackProof now has on disk, deliberately not a copy of what the registry claimed. `VERIFIED` cannot be constructed with an empty `checked_claims` and `UNAVAILABLE` cannot be constructed with a non-empty one, `file_name` must be `<64 lowercase hex>.tgz`, the directory must be absolute, and no field can hold a URL | Representing a mismatch — mismatched bytes raise, so no artifact exists for them; importing from `registry/` |
| `src/packproof/registry/errors.py` | Two taxonomies, because a registry call has two ways to be wrong: `RegistryError` for what the registry *said* and `ArtifactError` for what PackProof could or could not *do* with that answer. ASCII messages that name fields, codes, byte counts and paths under this process's control | Carrying a response object, a reason phrase, a response body or a URL taken from either |

### Planned, deliberately not created yet

Creating a package with only an empty `__init__.py` invites dead code and makes the
tree look further along than it is. These land with their first real module and tests
(`schemas/` and `registry/` arrived that way in Stage 2.1, the metadata client joined them
in Stage 2.2 and the artifact download in Stage 2.3B, so what is listed below for them is
the module or model that has not been written yet, not the package):

| Future package or module | Responsibility | First expected module |
|---|---|---|
| `registry/` — the source-host half | The source host's answer at the resolved revision, written under `config.workspace_path` | a host client behind the same `registry_opener()` policy |
| `schemas/` — the models after the artifact | `FileRecord`, `Finding`, `RiskAssessment`, `ScanReport` | joins `models.py` and `artifacts.py`, which already hold the reference, metadata and download models |
| `analyzer/` | Deterministic integrity analysis: inventory, SHA-256 hashing, structural comparison, diff classification | `inventory.py`, `hashing.py` |
| `detectors/` | Classify legitimate build output vs. unexplained modification | `artifact_detector.py`, `security_detector.py` |
| `provenance/` | Verify signatures, SLSA/npm attestations, and the commit-to-artifact claim | `attestations.py` |
| `ai/` | Explanation generation over already-computed findings, via a local model | `explainer.py` |
| `reporting/` | Render the validated report as JSON, HTML, PDF | `json_reporter.py` |

The npm-only assumption lives in `registry/`, not in the engine. PyPI support means a
sibling package that produces the same `schemas/` objects — `analyzer/` should not need to
know which registry a package came from.

## Planned data flow

```
"express@4.21.2"
   │
   ├─ registry/    manifest → tarball (bounded) + GitHub source at the resolved revision
   │               │            both written under config.workspace_path
   ├─ schemas/     PackageRef, FileRecord …        ← validation boundary for all input
   ├─ analyzer/    inventory + hashes + diff        ← deterministic
   ├─ detectors/   artifact vs suspicious           ← deterministic, evidence per finding
   ├─ provenance/  attestations + signatures        ← deterministic, may be "absent"
   ├─ ai/          natural-language explanation     ← explanatory only
   ├─ analyzer/    risk score from weighted findings← deterministic
   └─ reporting/   JSON (canonical) → HTML / PDF    ← rendered from validated models
```

A scan is identified by a deterministic scan id derived from package, version, resolved
source revision, and content hashes — so re-running the same inputs reproduces the same
id and the same report.

## Security rules for untrusted input

Everything PackProof reads is authored or published by someone who may be hostile.

1. **Never execute package code.** Read as data; parsing is not running.
2. **Never run npm lifecycle scripts** — `preinstall`, `install`, `postinstall`,
   `prepare`. PackProof must never invoke `npm install` on a scanned package.
3. **Validate all external metadata** before use. Registry JSON, `package.json`,
   repository URLs, version strings, and file lists are untrusted input; Pydantic
   models in `schemas/` are the only place they enter the engine.
4. **Never execute commands found in package files** — no `scripts`, no shell snippets,
   no build instructions.
5. **Bounded reads.** Size ceilings — `max_metadata_bytes` for a registry document and
   `max_download_bytes` for an archive, counted against the bytes that actually arrive
   because `Content-Length` is a claim, not a fact — entry-count ceilings,
   decompression-ratio ceilings, and timeouts on every network call.
6. **One origin, and no redirect is ever followed.** A request goes only to the configured
   registry host, over an authority approved by exact string equality and a URL rebuilt from
   the components that were approved. A `Location` value is a URL supplied *inside* a
   response, so the handler that would have followed one refuses instead.
7. **Safe filesystem handling.** Reject path traversal (`..`), absolute member paths,
   and symlinks or hardlinks pointing outside the workspace; keep untrusted content
   under `workspace_dir`, which is vetted whenever a `Settings` object is built — see
   *The scan workspace boundary* below. Both checks are required and neither substitutes
   for the other: the workspace check describes one configured directory as it was at
   configuration time, while member names are chosen by the package author later, inside
   an archive nobody has inspected yet. Neither check is used to name a file PackProof
   writes: a staged name is invented by this process and the published name is a digest
   this process computed over bytes it has already checked.
8. **No secrets in logs or reports.** Credentials live in `SecretStr` fields or the
   environment, are redacted by construction, and are never interpolated into a
   message, a rendered report, or a CI log. A registry's time-limited signed tarball
   query is a capability on the same footing, so it reaches neither a message, a `repr()`,
   a log line, nor a rendered traceback.
9. **Fail closed.** A download, extraction, or verification that cannot be completed
   safely is reported as an inconclusive scan, not silently skipped.

## The scan workspace boundary

The workspace is the one directory PackProof writes untrusted bytes into: the downloaded
tarball, the extracted package and the fetched source revision all land there before the
analyzer reads them. Nothing inside it is trusted and nothing in it is ever imported.
Keeping it away from the code PackProof runs is what turns "read as data, never execute"
from an intention into something checkable.

**Default location** — a per-user application cache, so one install always uses the same
workspace instead of a different one per working directory:

| Platform | Default |
|---|---|
| macOS | `~/Library/Caches/PackProof/scan-workspace` |
| Linux and other POSIX | `$XDG_CACHE_HOME/PackProof/scan-workspace`, else `~/.cache/PackProof/scan-workspace` |
| Windows | `%LOCALAPPDATA%\PackProof\scan-workspace`, else `%USERPROFILE%\AppData\Local\PackProof\scan-workspace` |

These are three conventions and two operating systems of evidence. The macOS location was
executed on macOS; the Linux and Windows rows are what `default_workspace_dir()` implements
and are tested by patching `sys.platform` and the cache variables, which exercises the real
branch on this machine rather than a mock's own assumption about it. CI's `ubuntu-latest`
runs the Linux branch for real once it runs at all; nothing has executed the Windows branch.

A relative or whitespace-only `XDG_CACHE_HOME` / `LOCALAPPDATA` is treated as unset: both
specifications call such a value meaningless, so falling back to the per-user location is
the safe reading of a malformed environment rather than a configuration failure. `platformdirs`
would compute the same thing and was left out deliberately — `sys.platform`, two
environment variables and `Path.home()` already express it, and a tool whose job is
auditing other people's dependencies should be conservative about adding its own.

**Override**: `PACKPROOF_WORKSPACE_DIR`, read through `Settings.from_env()`.

**Rejected values** (`_validate_workspace()` in `config/settings.py`):

- empty, whitespace-only, or "the current directory" (`""`, `"."`, `"./"`) — a scanner
  has to know which directory it is writing into;
- any path containing a `..` component, at any depth: the resolved location could be
  anywhere, so the component is refused rather than resolved and hoped safe;
- a `~user` prefix naming no account. `Path.expanduser()` answers that with `RuntimeError`,
  which pydantic does *not* convert the way it converts a `ValueError`, so the exception
  would reach the operator as a traceback naming no configuration variable. It is caught
  by name and re-raised as `ConfigurationError`, chained with `from`, never replaced by a
  fallback directory;
- anything that already exists and is not a directory — a regular file, or a symlink whose
  target was never created. The check is on the resolved location and on the filesystem
  type, never on the name, because an extension is a convention and a directory called
  `x.txt` is perfectly usable;
- anything that resolves to a protected root, or beneath one;
- anything that *contains* a protected root. This is the same boundary checked from the
  other side, and it is not decoration: a workspace of `/` never looks "inside" the
  installation, yet it would make the extraction boundary as wide as the filesystem and
  let any member name reach PackProof's own code. In a source checkout that also rules
  out the home directory itself, since the checkout sits in it;
- anything whose own location or contents cannot be inspected — an ancestor that denies
  directory search, a name longer than the filesystem allows, a working directory deleted
  underneath a relative override. Refused as `ConfigurationError`, with the operating
  system's reason kept as `__cause__`: no decision was ever made about such a path, so
  there is nothing to fall back to.

Protected roots come from the running package and the running interpreter. Neither is taken
from the working directory, and no rule is keyed to `$HOME` in either direction: a home
directory is an ordinary location, protected only when one of the three sources below
actually lands inside it, and never protected merely because an ancestor walk passed through
it.

| Root | When it applies |
|---|---|
| The `packproof` package directory | always |
| The repository root | when one ancestor directory holds a `pyproject.toml` whose `project.name` is `packproof`, *and* version-control metadata (`.git`, as a directory or as the file a worktree writes), *and* the package actually running sits at that directory's `src/packproof` |
| The interpreter's package roots | always, for every usable path `sysconfig` reports as `purelib` or `platlib` |

The last row is a property of the *interpreter*, not of where PackProof's files happen to
sit. Keying it to the package's parent directory covered a wheel, whose package really is
inside `site-packages`, and missed an editable install, whose source stays in the checkout
while the environment it imports from is somewhere else entirely — and a workspace
directory named `packproof`, or `pydantic`, inside that environment shadows a real module
just as well from either kind of install. Asking `sysconfig` rather than assuming keeps the
claim as narrow as it is true: an ordinary parent directory — a home, a working directory,
an interpreter installed unusually — is never protected as a side effect, so everything
beneath it stays available as a workspace.

The repository row requires three signals because each of the first two is a copyable
artifact. A manifest says what a project is called, and a name is a string anyone can write;
a `.git` entry says that some tree is versioned, and `mkdir somewhere/.git` is enough to say
that. Neither says the directory holds *this* code, so a rule built on those two alone is
satisfied by a home directory that contains a virtualenv — and adopting a directory that way
did measurable damage: the per-user default workspace always lives beneath the home
directory, so it then resolved *inside* a protected root and `Settings()` refused its own
default, which is a tool failing for someone who configured nothing and is indistinguishable
from a broken install.

The third signal is a path comparison rather than a probe: the candidate's `src/packproof`
must be the directory the interpreter already imported from. Asking it first also bounds the
ancestor walk, because the manifest is now opened only for the one directory that can
possibly hold this installation.

Three signals agreeing is a structural heuristic, not verification. Nothing about a
`pyproject.toml` or a `.git` entry is authenticated — the `.git` entry is not even opened,
only seen as a directory or a file — so what the rule establishes is "this directory is the
checkout this package was imported from", which is enough to keep untrusted bytes out of it
and is not a claim about who wrote the code. It also commits to one source layout, the
`src/packproof` tree `pyproject.toml` builds with `packages = ["src/packproof"]`: a
flat-layout checkout is not recognized as a repository, and the consequence is a *narrower*
protected set — the package directory and the interpreter's roots stay protected — never a
wider one.

A wheel install and an unpacked release tarball have no version-control metadata, get no
repository root, and remain fully functional on package-and-environment protection alone.

Malformed manifests are refused rather than fatal. `pyproject.toml` above the package is
not PackProof's file, and valid TOML may put anything under `project` — a string, a number,
a list — so `project` must be a table and `name` a string before either is read. Unreadable
files, non-UTF-8 content and unparseable TOML are the exceptions a manifest *read* can
produce, and those are the only ones the reader catches; unrecognized simply means "not
PackProof". A manifest that cannot be *opened* because its directory denies search is a
different case and is deliberately not absorbed there: the ancestor walk has no business
deciding that a directory it could not look inside is somebody else's project, so the error
travels outward and becomes a workspace refusal.

**Failures of inspection are refusals, not crashes.** `Path.exists()`, `is_dir()` and
`is_symlink()` re-raise `EACCES` and `ENAMETOOLONG` instead of reporting `False`, and
`Path.absolute()` asks the operating system for a working directory that may have been
deleted. Those were tracebacks: an `OSError` is not converted by pydantic the way a
`ValueError` is, so it escaped `Settings()` and `from_env()`'s handler naming no configuration
variable — reachable with no override at all, by a `HOME` the process cannot search, since the
default workspace is derived from it. Each filesystem step is now wrapped, chained with
`raise … from`, and reported as a `ConfigurationError`; nothing falls back to a second-choice
directory, because a path that cannot be checked has not been shown to be safe. Listing the
protected roots reads the filesystem as well, and a boundary that cannot be computed is a
reason to refuse rather than a reason to assume nothing is protected.

Wrapping the calls is not the whole guarantee, because a refusal is assembled after the
handlers have been left. A message that decided how to name a root by asking the filesystem
again could fail in a place no handler covers — and could report a role derived from a later
look than the one that produced the refusal. So `_protected_roots_with_roles()` attaches each
role to its path **at discovery**, where the identity of that path is already known, and
`protected_roots()` remains the path-only projection of the same list. Building a refusal is
then a comparison between values already in hand and performs no filesystem operation.

The two kinds of inspection failure also get **different guidance**, because they have
different remedies. A workspace that cannot be inspected is a property of the configured
value: the message names `workspace_dir`, quotes the operating-system reason, and says to fix
the permissions on that path or point `PACKPROOF_WORKSPACE_DIR` somewhere readable. Protected
locations that cannot be listed are a property of the installation, and no workspace value
helps while they stay unreadable — the next value would be validated against the same
boundary — so that message names the installation, the checkout, and the interpreter's package
directories as what to check, and says in terms that repointing the setting will not fix it.
Advice that points at the wrong knob is worse than none: it makes a working configuration look
like the cause of a failure it did not cause.

A refusal names the **role** of the root it reached — PackProof's own package directory, the
PackProof source checkout, or the running interpreter's package directory — because the three
need different answers from the operator, and calling somebody else's `site-packages`
"PackProof's own files" sends them looking in the wrong tree. A path discovered by more than
one source keeps the first role: the package directory is named as the package even when it is
also an interpreter root, which is the narrower and more actionable claim.

**Normalization and symlinks.** `~` is expanded, the value is made absolute, and
containment is decided on `Path.resolve()` output — not on the string that was typed.
Resolution is what puts `src`, `./src/packproof/cli/../..` and a symlink whose target is
the package directory into the same comparison. A relative override is therefore anchored
to the working directory and *then* checked, which is why `PACKPROOF_WORKSPACE_DIR=src`
is refused from a checkout and accepted from anywhere else; the default needs no such
anchoring, because it is computed from the home directory. `Path.is_relative_to()` alone
would not do any of that: fed an unnormalized path it answers a lexical question about what
is a filesystem fact.

**Two accessors, two questions.** `Settings.workspace_dir` keeps the operator's spelling —
expanded and absolute but *not* symlink-resolved — because a report or an error message has
to show what was configured rather than an unrelated directory the filesystem happened to
resolve it to. `Settings.workspace_path` answers the other question, where this actually
is, and is the value any containment check must compare against. They differ whenever a
symlink is involved, and `workspace_path` re-resolves on every access.

That last property is a limit, not a feature to lean on, so it is stated plainly rather
than implied. Validation describes the filesystem as it was at construction time. Replace
the workspace leaf with a symlink afterwards and `workspace_path` reports the new target
while no decision already taken is revisited — a test asserts this rather than arguing
about it. Nothing in configuration can close that gap, and it is not meant to: whoever
writes into the workspace resolves and checks each destination at the moment of writing.

Two further limits, same spirit. Non-existent paths are resolved lexically, because the
workspace does not exist yet and validation must not create it. And constructing `Settings`
never touches the filesystem beyond reading what is already there; the directory is created
later, by the code that writes into it.

**Configuration validation is not archive-extraction security.** The check above answers
one question about one operator-supplied value. Member names inside a tarball are chosen
by the package author at extraction time, so the extractor must independently, for every
member: reject `..` and absolute member paths, resolve the destination and confirm it is
contained in the workspace, refuse or isolate symlinks and hardlinks, honour the size,
entry-count and decompression-ratio limits, and create directories without inheriting the
archive's mode bits. A correct workspace path makes none of that unnecessary.

## Artifact retrieval and publication

Stage 2.3B is the first code in PackProof that writes registry bytes to disk, and it stops at
the file. **No archive is opened, listed or extracted, nothing inside one is executed, and `npm`
is never invoked** — extraction is Stage 3, and the rules that will govern it are already
written in *Configuration validation is not archive-extraction security* above.

One download, one owner per step:

1. `npm_fetcher.py` retrieves the version's metadata, which records `dist.tarball` as text and
   never requests it.
2. `download_policy.py` decides whether that text is a target worth connecting to, and rebuilds
   the request URL from the components it approved. A refusal names a code, not a URL: npm
   serves tarballs with time-limited signed queries, so the query is a capability, and a refused
   target is not repeated back to the caller either.
3. `npm_download.py` sends the request through `registry_opener()` — the same chain the metadata
   client uses, so one origin guard, one redirect refusal and one no-proxy rule answer for both
   — and streams the body into a file `artifact_store.py` has staged, hashing it as it arrives.
4. Every integrity claim the metadata supplied is compared over those bytes. A mismatch raises,
   the staged file is discarded, and nothing is published.
5. Only then does the digest become a name: `os.link` of the staged file to `<sha256>.tgz`, an
   `fsync` of the directory, and a `PackageArtifact` reporting which claims held.

Three choices inside that path carry their reasons because each was measured first:

- **Publication is a link, not a rename.** `os.replace` replaces the *name* a destination holds,
  including through a symlink, and takes no `dir_fd`; `os.link` never clobbers and answers
  `EEXIST` with the destination intact. So `EEXIST` is the only error in which inspecting the
  destination means anything, and every other errno is a plain failure.
- **`Content-Length` is believed zero times.** The ceiling counts the bytes that arrive, and a
  body that overshoots is abandoned within one chunk of the limit rather than after it.
- **No claim is traded for a stronger one.** Checking only `dist.integrity` would let a version's
  weaker `dist.shasum` go unverified while the artifact still reported `VERIFIED`, so every
  supplied claim is checked and every claim that held is named, in the models' own rank order.

**What this stage does not guarantee.** Four limits are open on purpose. Each is pinned by a test
that asserts the limit as it is rather than a better one nobody measured.

- **Frame locals are cleared in PackProof's frames and nowhere else.** `download` and `_attempt`
  empty their own locals in their own `finally`, so walking a failure's traceback finds `None`
  where the signed URL was — the names are present, the values are not. Helper frames in the
  same module bind the request or the response while they work and are not scrubbed: a failure
  raised inside `_send` leaves exactly one holder, `request` in the `_send` frame. Nothing
  renders frame locals — a traceback prints source lines, and the message, `repr`, rendered
  traceback and log line of every failure path are asserted target-free — so this is not a
  disclosure, and it is why **U9/S-11 stays open**. Standard-library frames above PackProof's sit
  outside this boundary entirely.
- **The commit window is not atomic, and cannot be made so.** A publication is a link followed by
  a directory sync, and an interrupt between them is a state the ledger has to describe rather
  than a state the code can prevent; arbitrary asynchronous interruption at certain bytecode
  boundaries is not fully atomic in CPython. `settle()` exists to name what such a window became,
  and the round-8 precedence rules decide which of several competing exceptions the caller sees.
- **The store publishes under the name it is handed.** `finalize` checks that a digest is
  present, not that the resulting `<name>.tgz` is 64 lowercase hex, so a caller that passed
  `"../escape"` would write outside the workspace. Nothing in the download path can produce that
  value — `npm_download.py` assigns `hashlib.sha256(...).hexdigest()` and `PackageArtifact`
  refuses a `file_name` that is not the content-addressed spelling — so the guarantee is real but
  lives one layer up, in the caller and the model. `artifact_store.py`'s test says so by name.
- **Nothing here is measured on Windows.** The publication core is exercised on macOS (local
  runs, CPython 3.11.15 / 3.12.13 / 3.13.6 / 3.14.4) and Ubuntu (CI, 3.11 / 3.12 / 3.13).
  `os.link`, `dir_fd` and `O_EXCL` behave differently on Windows, and the only Windows content
  anywhere in PackProof is `default_workspace_dir()`'s branch, which is tested by patching
  `sys.platform` — that is not the same as having run on Windows, and this stage claims no more
  than the platforms that were executed.

## Configuration and logging conventions

- Modules obtain a logger with `logging.getLogger(__name__)` and never configure one.
  Because those names begin with `packproof.`, they inherit level, format, and
  destination from `utils/logging.configure_logging()`, which the CLI is to call once
  when a command exists that does work — today's `--help` and `--version` do not, and
  neither the logging call nor `Settings.from_env()` is wired into a command yet.
- Logs go to **stderr**; report output goes to **stdout**, so that when a report command
  exists, `packproof … > report.json` stays clean. No command emits a report yet.
- Limits come from `Settings`, not from module-level constants scattered around, and
  never from anything a scanned package controls.
- New settings are added as validated, bounded `Settings` fields with an entry in
  `ENVIRONMENT_FIELDS` — one place defines the environment contract.

## Testing conventions

| Directory | Contents | Network |
|---|---|---|
| `tests/unit/` | One module per test module, `typer.testing.CliRunner` for the CLI | forbidden |
| `tests/integration/` | Process-boundary behavior (`python -m packproof`, entry points) | forbidden |
| `tests/fixtures/` | Synthetic inputs; see its README | n/a |

The npm metadata client is mocked at its transport boundary: a test injects an opener and
the client never builds a real one, so no request leaves the process. A socket guard in the
fetcher's module fails any test that reaches the network anyway, and one integration test
imports the registry layers in a fresh interpreter under `sys.addaudithook` to prove the
import itself opens no socket and writes no file. The source-host client, when it arrives,
is mocked the same way.

The artifact layers are tested against a real temporary directory instead of a mocked
filesystem, because the properties worth proving are properties of the directory: the file on
disk hashes to the name it was published under, a refused download leaves no file, foreign
bytes in a destination position are still there at their original mtime. The transport is the
part that is stubbed, and deliberately so — a chunk list that can raise partway through is what
makes the ceiling, truncation and claim-failure tests about this client rather than about
`http.client`. Rare paths are reached by deterministic fault injection (a log sink that is
down, an `fsync` that reports errno 0, an interrupt landing inside the commit window) rather
than left untested.

Two limits of that audit-hook test are written into its own file, because a control that
overstates what it can see is worse than no control at all. The four socket-stack names are
raised for the hook with `sys.audit` instead of dialled, since no test in this stage may open a
socket or ask for a name; only `urllib.Request` is proved from a real call, through urllib's own
`file://` opener. And a `Request` that never reaches an opener fires no event whatever, so "no
network-shaped event during import" means no network *operation* — not that no request object
was built.

CI must never depend on an external service's availability or on a package that someone
could unpublish. Determinism is a product requirement, so a test that reaches the
network is a bug in the test.

## CI

`.github/workflows/ci.yml` runs on pull requests to any branch and on pushes to `main`
and development branches. The matrix covers Python 3.11, 3.12, and 3.13 — the floor
declared in `requires-python` plus the two supported releases above it.

The matrix is pinned through `UV_PYTHON`, not a `--python` flag on `uv sync`: `uv run`
re-checks the environment against `.python-version` and would rebuild a 3.11 venv as
3.13 before pytest ran. A guard step asserts that the interpreter actually in use
matches the matrix entry, so a silently wrong Python version fails the job instead of
passing it three times with the same interpreter. (This was observed locally, which is
why the guard exists.)

Order matters: `ruff check` (correctness), then `ruff format --check` (style), then
`pytest` (behavior), then CLI smoke tests. `uv sync --frozen` refuses to re-resolve, so
a pull request whose `uv.lock` is out of date with `pyproject.toml` fails immediately
rather than hiding the mismatch.

Permissions are read-only (`contents: read`) and there is no deploy step: this workflow
verifies code, it does not publish it.

## Decisions worth revisiting

- **`Settings.from_env()` over `pydantic-settings`.** Three bounded values do not
  justify a dependency and a second, implicit configuration surface. Revisit if
  per-ecosystem registries or AI model options make the env parsing branch-heavy.
- **`urllib.request` for the metadata client, not an HTTP library.** The requirement is
  one bounded `GET`, a keyword-bound timeout, and a transport a test can replace — all of
  which the standard library gives, and the fetcher takes the opener as a constructor
  argument so swapping the transport later changes one call site. Revisit when the
  download stage needs retries, ranged requests, or parallel fetches.
- **Version in `__init__.py`, read at build time.** Keeps one literal and avoids
  `importlib.metadata` at runtime. Alternative: static version in `pyproject.toml` and
  a runtime metadata lookup. Current choice is better for source-tree execution.
- **Ruff selects `S` (bandit).** Expect per-file ignores in tests and possibly in
  `registry/`; add narrow, commented ignores rather than dropping the rule set.
- **Planned Pydantic models in `schemas/` as the only input boundary.** Correct for a
  CLI product; if PackProof later ships a library API, decide whether public
  dataclasses matter for downstream consumers.
- **Report format precedence (JSON canonical).** If HTML becomes the primary artifact,
  the JSON schema still stays authoritative to keep reports diffable.
