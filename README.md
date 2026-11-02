# PackProof

**Prove what's inside your packages.**

PackProof is a software supply-chain integrity analyzer. It compares a package you
*install* from a registry with the source you can *read* on GitHub, and tells you
whether the differences are ordinary build output or something that should worry you.

---

## The problem

You inspect a repository, review its code, check its tests — then run
`npm install`, and receive a different artifact. The tarball on the registry is not
the repository. Its contents were produced by a build machine, from some commit, at
some time, by whoever held the publishing credentials that day.

Some divergence is legitimate and expected:

- TypeScript compiled down to JavaScript
- Modules bundled and minified
- Files generated at build time (clients, parsers, schemas)
- Build and release tooling adding metadata or wrappers

Other divergence is how compromises actually look:

- Publishing credentials leaked, and the tarball contains code nobody reviewed
- A CI/CD build pipeline tampered with between commit and publish
- A small payload injected into an otherwise familiar file
- A dependency whose installed behavior does not match its own source
- A release tagged and published from a revision that is not the tagged one

Today, telling those two lists apart means reading minified JavaScript by hand. Most
teams do not, so the check silently never happens.

## What PackProof does

Given a package name and version, PackProof will:

1. Download the published artifact from the registry.
2. Identify the source repository it came from.
3. Resolve the exact source revision for that release.
4. Compare file inventories and SHA-256 hashes between source and artifact.
5. Classify legitimate build output (compiled, bundled, minified, generated).
6. Flag modifications that a stated build cannot explain.
7. Verify available provenance attestations and signatures.
8. Explain findings in plain language through a local AI model.
9. Produce a deterministic, evidence-backed risk assessment.
10. Emit a detailed audit report as JSON, HTML, and PDF.

The first ecosystem is **npm**; **PyPI** is planned. Adding an ecosystem must not
require rewriting the analysis engine — registry specifics stay behind fetchers.

## Why source-to-package integrity matters

A dependency's published artifact runs with your build's permissions and, through it,
your product's users' trust. Integrity checking is not about re-doing vulnerability
scanning; it answers a different question: *is the thing I installed the thing I
reviewed?* If the answer is unknowable, every other control downstream is guesswork.

## Architecture

Python `src`-layout package. Each layer owns one concern and depends downward only.

```
packproof/
├── pyproject.toml              # single source of project metadata and tool config
├── uv.lock                     # reproducible dependency resolution (tracked)
├── src/packproof/
│   ├── __init__.py             # authoritative __version__
│   ├── __main__.py             # `python -m packproof`
│   ├── cli/main.py             # Typer app: parsing + presentation only
│   ├── config/settings.py      # validated, immutable runtime limits
│   └── utils/logging.py        # one logging entry point for the whole app
├── tests/{unit,integration,fixtures}/
├── docs/architecture.md        # module responsibilities and future design
├── examples/                   # example scans and inputs
└── .github/workflows/ci.yml
```

Planned engine layers — `fetchers/`, `analyzer/`, `detectors/`, `provenance/`, `ai/`,
`reporting/`, `schemas/` — are described in
[docs/architecture.md](docs/architecture.md). They are deliberately **not** created as
empty directories ahead of their code.

The load-bearing rule: **deterministic verification produces the findings; AI only
explains them.** Hashing, file comparison, archive validation, provenance checks and
risk scoring are reproducible logic. A model that disagrees with the evidence is
wrong, and PackProof will say so.

## Technology stack

| Concern | Choice | Milestone 0.1 |
|---|---|---|
| Language | Python `>=3.11` (developed on 3.13) | in use |
| Packaging & dependencies | `uv` + `pyproject.toml` + `uv.lock` | in use |
| CLI | Typer | in use |
| Data validation | Pydantic | in use (settings) |
| Tests | Pytest | in use |
| Lint & format | Ruff | in use |
| Logging | standard `logging` | in use |
| CI | GitHub Actions | in use |
| HTTP | HTTPX | planned, not installed |
| Reports | ReportLab, Jinja2 | planned, not installed |
| AI explanations | Ollama / model API | planned, not installed |
| Web UI | React + TypeScript + Vite + Tailwind | planned, not started |

## Status: implemented vs planned

**Implemented in 0.1.0**

- Installable Python package with a reproducible lockfile
- `packproof --help`, `packproof --version`, `python -m packproof --help`
- Single-source version, read at build time (no duplicated version strings)
- Validated, immutable `Settings` with environment overrides
- A workspace boundary: the directory untrusted downloads land in is checked against the
  package directory, the interpreter's package roots and any identified PackProof checkout,
  in both directions, with a per-platform cache default that ignores the working directory
- Central logging configuration, idempotent, stderr by default
- Unit and integration tests for all of the above
- Ruff lint + format gates, GitHub Actions CI on Python 3.11/3.12/3.13
- This README, license, `.gitignore`, architecture doc, PR template

**Not implemented — planned for later milestones**

- ❌ Package downloading and registry metadata resolution
- ❌ Source repository and revision resolution
- ❌ File comparison, hashing, diffing
- ❌ Build-artifact classification and modification detection
- ❌ Provenance and signature verification
- ❌ AI-generated explanations
- ❌ Risk scoring
- ❌ JSON, HTML and PDF reports
- ❌ Web interface
- ❌ PyPI support

There is no `packproof scan` command yet. The architecture is ready for the npm
downloader; the downloader is the next milestone.

## Installation

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Barnakdas-tech/PackProof-Standalone.git
cd PackProof-Standalone
uv sync --dev          # creates .venv, installs runtime + dev dependencies
```

Verify the environment:

```bash
uv run packproof --version
uv run python -m packproof --help
```

## Development setup

```bash
uv sync --dev                    # install with dev tools
uv run pytest                    # test suite
uv run ruff check .              # lint
uv run ruff format .             # apply formatting
uv run ruff format --check .     # verify formatting (what CI runs)
```

`uv sync` installs an editable build, so edits in `src/` take effect immediately.
Python 3.13 is pinned in `.python-version` for local work; `requires-python = ">=3.11"`
is the compatibility floor and CI tests all three versions.

## CLI usage

```bash
packproof --help          # application help
packproof -h              # same, short form
packproof --version       # e.g. "packproof 0.1.0"
packproof -V              # same, short form
python -m packproof --help  # identical application, no installed script needed
```

Planned syntax, **not available yet**:

```bash
packproof scan express@4.21.2
packproof download express@4.21.2
packproof source express@4.21.2
packproof report <scan-id>
```

Exit codes follow the Click/Typer conventions PackProof is built on, verified against
the installed version: `0` for success (`--help`, `--version`), `2` for a usage error
(`packproof` with no arguments prints help and exits 2, as do an unknown command or an
unknown option), and `1` for a user abort.

### Configuration

Optional environment variables, validated whenever a `Settings` instance is built:

| Variable | Default | Meaning |
|---|---|---|
| `PACKPROOF_HTTP_TIMEOUT_SECONDS` | `30.0` | Per-request timeout (must be > 0, ≤ 300) |
| `PACKPROOF_MAX_DOWNLOAD_BYTES` | `52428800` | Download size ceiling (must be > 0, ≤ 2 GiB) |
| `PACKPROOF_WORKSPACE_DIR` | per-user cache, see below | Where untrusted downloads land |

Invalid values raise `ConfigurationError` instead of falling back silently. No API
keys are required.

No command builds `Settings` yet — the CLI exposes `--help` and `--version` only — so
validation becomes an operating reality the moment `scan` calls `Settings.from_env()`.

The workspace default is a per-user application cache, so it stays in one place instead
of moving with whatever directory the command was run from:

| Platform | Default workspace |
|---|---|
| macOS | `~/Library/Caches/PackProof/scan-workspace` |
| Linux and other POSIX | `$XDG_CACHE_HOME/PackProof/scan-workspace`, else `~/.cache/PackProof/scan-workspace` |
| Windows | `%LOCALAPPDATA%\PackProof\scan-workspace`, else `%USERPROFILE%\AppData\Local\PackProof\scan-workspace` |

Reading the configuration never creates the directory; the code that writes into the
workspace does. A configured workspace is rejected if it is empty, names the current
directory, contains a `..` component, carries a `~user` prefix that matches no account,
resolves to something that already exists and is not a directory (a plain file, or a
symlink pointing at nothing), cannot be inspected at all because an ancestor denies
directory search or the name is longer than the filesystem allows, or resolves (after `~`
expansion and symlink resolution) into a protected location — the `packproof` package
directory, the interpreter's own package roots, or a PackProof checkout identified by its
manifest, its version-control metadata *and* the running package sitting at that
directory's `src/packproof` — or contains those locations, which is why `/` is refused as
well. Each refusal names which of the three it reached, because they call for different
fixes. `docs/architecture.md` explains how those locations are identified, why the
checkout rule is a structural heuristic rather than verification, and what this check
deliberately does not cover.

**How much of this was executed where.** The macOS default and every rejection above were
run on macOS. The Linux and Windows locations are the conventions `default_workspace_dir()`
implements and are tested by simulating `sys.platform` and the cache environment variables
in place of a second operating system; the Linux branch additionally runs for real on
`ubuntu-latest` once CI executes, and Windows has not been run anywhere. Nothing here
claims otherwise.

## Testing

```bash
uv run pytest                                    # everything
uv run pytest -m unit                            # fast single-module tests
uv run pytest -m integration                     # cross-boundary tests
uv run pytest tests/unit/test_cli.py -v          # one file, verbose
```

Tests use `typer.testing.CliRunner` for the CLI, `tmp_path` for filesystem isolation,
and a real subprocess for `python -m packproof`. Every test asserts behavior; there
are no placeholder assertions.

## Security principles

PackProof handles untrusted third-party software by design. These rules are binding
on every future module:

- **Never execute downloaded package code.** Parsing is not running.
- **Never run npm lifecycle scripts** (`preinstall`, `postinstall`, `prepare`) of a
  scanned package.
- **Never trust external package metadata without validation** — registry JSON,
  `package.json`, and repository fields are attacker-influenced input.
- **Never run arbitrary commands found in package files.**
- **Never expose secrets in logs.** No credential field may reach a log record;
  secrets live in `SecretStr` fields and environment variables, never in source.
- **Safe filesystem handling** — path traversal, symlink escapes, and zip-slip must be
  rejected when archive extraction lands; sizes and entry counts get bounds.
- **Separate untrusted downloads from application code** — downloads go under
  `workspace_dir`, and that path is validated against the locations PackProof trusts and
  imports from whenever `Settings` is built, not against a spelling of them.
- **Reproducible evidence** — findings are hashes and diffs a third party can re-derive
  and disagree with, not model opinions.

Archive extraction is intentionally not implemented yet; these constraints exist so it
lands with the guards attached.

## Contributing

Branch from `main` as `feat/…`, `fix/…`, or `chore/…` and open a pull request; use
[`.github/pull_request_template.md`](.github/pull_request_template.md). CI must be
green: lint, format check, and tests on Python 3.11, 3.12, and 3.13.

Before opening a PR:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Scope discipline is expected: no dependency added without a reason, no empty
placeholders, no analysis logic in the CLI layer.

## License

MIT — see [LICENSE](LICENSE).
