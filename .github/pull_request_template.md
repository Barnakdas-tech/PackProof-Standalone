## Summary

One or two sentences on what changed. Keep it factual: this PR either adds a capability
or fixes a defect, and the sentence should say which.

## Problem addressed

What was broken, missing, or unsafe before this change? Link the issue if there is one.

If this PR is refactoring without a behavior change, say so explicitly and explain what
the previous structure made hard.

## Implementation details

The approach, in enough detail that a reviewer can follow the diff without guessing:

-
-
-

## Architecture decisions

- Which layer owns the new code, and why that layer (`fetchers/`, `analyzer/`,
  `detectors/`, `provenance/`, `ai/`, `reporting/`, `schemas/`, `config/`, `cli/`).
- Any new dependency, and why the standard library was not enough.
- Any deviation from `docs/architecture.md`, and whether that document needs updating.
- Anything deliberately *not* done here, with the reason.

## AI models and coding assistants

Required for this project. List each model or assistant that contributed code, review,
or documentation, with what it was used for.

| Model / assistant | Used for | Human verification performed |
|---|---|---|
| | | |

State plainly which parts were generated versus reviewed. AI output that was not
independently verified is not evidence that the code is correct.

## Tests performed

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run packproof --help
uv run python -m packproof --help
```

- [ ] New behavior has a test that fails without it
- [ ] No test was weakened, skipped, or deleted to make the suite pass
- [ ] Tests are deterministic: no network, no wall-clock or hash-order dependence
- [ ] Filesystem-touching tests use `tmp_path` / isolated directories

Paste the actual output of the commands you ran. Do not report a command as passing
if you did not run it.

## Security considerations

PackProof processes untrusted third-party software. Answer each:

- Does this change execute, import, or eval anything from a scanned package? It must not.
- Does it run npm lifecycle scripts? It must not.
- Is external metadata (registry JSON, `package.json`, repository fields) validated
  before use rather than trusted?
- Any new filesystem handling: are path traversal, absolute member paths, and symlink
  escapes rejected? Is content confined to `workspace_dir`?
- Any new network call: is there a timeout and a size bound?
- Can any secret, token, or credential reach a log record, a report, or CI output?
- Does a failure path fail closed rather than silently skipping a check?

## Documentation updates

- [ ] `README.md` reflects reality — implemented features separated from planned ones
- [ ] `docs/architecture.md` updated if module responsibilities or invariants changed
- [ ] Docstrings on every new public function and class
- [ ] `CHANGELOG`/version bumped if this changes released behavior

## Checklist

- [ ] Branch is `feat/…`, `fix/…`, `chore/…`, or `refactor/…` — not `main`
- [ ] `uv.lock` is committed and consistent with `pyproject.toml` (`uv sync --frozen` succeeds)
- [ ] `ruff check .` and `ruff format --check .` pass locally
- [ ] `pytest` passes locally on Python 3.11 and 3.13
- [ ] No secrets, credentials, `.env` files, or downloaded package content committed
- [ ] No unrelated changes included; diff reviewed line by line
- [ ] CI is green
- [ ] Version compatibility (`requires-python = ">=3.11"`) considered, not just the local 3.13 interpreter
