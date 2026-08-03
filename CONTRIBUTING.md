# Contributing

Thanks for your interest in DeepSight. This document is the contract between
you and the maintainers; please read it before opening a pull request.

## Development setup

Prerequisites: Python >= 3.11 and [uv](https://docs.astral.sh/uv).

```bash
uv venv
uv pip install -e '.[dev]'
```

Run the full gate before pushing:

```bash
make all   # lint + typecheck + test
```

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <imperative summary>
```

Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `ci`, `build`,
`perf`. Scope is a component name when useful, e.g.
`feat(cache): add TTL eviction`. The changelog is maintained from these
messages, so write them for a reader, not for the diff.

## Pull requests

- One logical change per PR. Split refactors from features.
- Describe the _why_, not just the _what_; link issues and design docs
  (`docs/architecture.md`).
- Keep the diff reviewable. If a PR needs more than ~400 changed lines of
  substance, it probably needs to be split.
- All CI checks must be green: lint, typecheck, tests.

## Quality gates

- **Zero-warning policy.** `make lint` runs ruff with the configured rule set
  (`E,F,I,UP,B,SIM`). A warning is a failed build, not a suggestion.
- **Types are non-negotiable.** `make typecheck` runs mypy over `src/` with
  `disallow_untyped_defs` and `warn_unused_ignores`. If you reach for an
  ignore or a cast, expect to justify it in review.
- **Tests.** New behavior ships with tests. Unit tests must not require live
  model endpoints or network access; inject fakes or mock the HTTP backends.
  Async tests run under pytest-asyncio (auto mode is configured).

## Code style

The bar is _edited, not generated_. Code should read as if a senior engineer
wrote it deliberately:

- No comments that narrate what the code plainly does. Comment the _why_, or
  nothing.
- No dead code, unused exports, or speculative abstractions. Build what the
  milestone requires.
- Consistent naming within a module; prefer the existing vocabulary over
  introducing synonyms.
- Every non-2xx API response uses the standard error envelope used elsewhere
  in the codebase; match the existing convention rather than inventing a new
  one.

## Adding a benchmark

Benchmarks live in `bench/` and stream rows from the Hugging Face
datasets-server API (no local dataset downloads). See
[docs/benchmarks.md](docs/benchmarks.md) for the methodology.

To add a benchmark:

1. Open `bench/harness.py` and add an entry to the `BENCHES` registry:
   `name: (dataset, config, split, image_field, question_field, answer_fields, answer_type_field)`.
   The dataset must expose rows via the datasets-server `/rows` API and images
   via `src` fields (a `decoded_image` object for MathVista).
2. Add a system prompt in `make_payload()` if the benchmark needs one (e.g.
   "answer with just the value").
3. Extend `score()` with any special scoring rules (float tolerance, multi
   answer lists). The default is normalized exact match.
4. Add a row for the benchmark in the README results table (TBD cells) and a
   note in `docs/benchmarks.md`.
5. Verify with a small run: `python3 bench/harness.py --bench <name>
   --limit 5 --endpoint <endpoint> --model <model> --api-key <key>`.
