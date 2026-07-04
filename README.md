# neural-program-simplification

Toy research project on simplifying neural tensor programs into sparse,
interpretable versions that preserve behavior on reference datasets.

The initial research target is conservative: first find transformations that remove
redundant computation while leaving the model mathematically unchanged on a fixed
reference dataset. Later phases can study lossy approximations such as replacing SiLU
with ReLU, pruning near-zero activations or neurons, and introducing sparse surrogate
subprograms with measured degradation.

No model transformation code exists yet. This repository currently contains the docs,
tests, project hygiene, and first experiment-infrastructure primitives needed to start
carefully.

## 1-minute quickstart

```bash
git clone https://github.com/d0rbu/neural-program-simplification.git
cd neural-program-simplification
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
```

## What this includes

| Area | Tooling |
|---|---|
| Package management | `uv`, `pyproject.toml`, `uv.lock` |
| Local commit checks | `pre-commit` |
| Linting | `ruff` |
| Type checking | `ty` |
| Tests | `pytest`, `pytest-cov`, `hypothesis` |
| Runtime contracts | `phantom-types`, `beartype` |
| Array shape/dtype checks | `jaxtyping` |
| Model runtime | `transformers`, `torch`, `accelerate` |
| Starter task datasets | packaged JSON task documents under `neural_program_simplification/data/task_datasets/` |
| Agent guidance | `AGENTS.md`, `CLAUDE.md` |

## Repo layout

```
neural_program_simplification/  reusable experiment infrastructure
tests/              pytest suite, including property tests
docs/               project documentation
.github/workflows/  CI checks
```

## Where to go next

| You want to... | Read |
|---|---|
| Understand the research direction | [`docs/research/agenda.md`](docs/research/agenda.md) |
| Start developing | [`docs/onboarding/getting-started.md`](docs/onboarding/getting-started.md) |
| Understand the correctness model | [`docs/development/correctness.md`](docs/development/correctness.md) |
| Add a new experiment | [`docs/pipelines/experiment-lifecycle.md`](docs/pipelines/experiment-lifecycle.md) |
| See tool configuration | [`docs/reference/configuration.md`](docs/reference/configuration.md) |
| Find a file's purpose | [`docs/reference/file-reference.md`](docs/reference/file-reference.md) |

## License

MIT. See [`LICENSE`](LICENSE).
