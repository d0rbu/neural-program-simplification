# File Reference

## Top Level

| File | Purpose |
|---|---|
| `README.md` | Project summary, quickstart, and doc links |
| `AGENTS.md` | Agent entry point and repo conventions |
| `CLAUDE.md` | Claude-specific pointer to agent conventions |
| `pyproject.toml` | Package metadata and tool configuration |
| `uv.lock` | Locked dependency graph |
| `.pre-commit-config.yaml` | Local commit hooks for lockfile, lint, type, and test checks |
| `.python-version` | Python version for local tooling |
| `.gitignore` | Local artifacts excluded from git |
| `LICENSE` | MIT license |

## Source

| File | Purpose |
|---|---|
| `neural_program_simplification/__init__.py` | Public exports for experiment infrastructure |
| `neural_program_simplification/task_datasets.py` | Task-dataset domain types, JSON storage, and tokenizer validation |
| `neural_program_simplification/model_execution.py` | Next-token model protocol and task-dataset scoring results |
| `neural_program_simplification/huggingface.py` | Optional Hugging Face causal-LM loader and adapter |

## Tests

| File | Purpose |
|---|---|
| `tests/test_correctness_tools.py` | Phantom type, runtime check, array contract, and property-test examples |
| `tests/test_task_datasets.py` | Task-dataset storage and tokenizer-invariant tests |
| `tests/test_model_execution.py` | Next-token scoring tests with a static model |

## Docs

| Path | Purpose |
|---|---|
| `docs/README.md` | Documentation index |
| `docs/research/` | Research agenda and project-specific planning docs |
| `docs/onboarding/` | Setup and day-to-day workflows |
| `docs/development/` | Correctness and testing guidance |
| `docs/pipelines/` | Experiment lifecycle guidance |
| `docs/pipelines/task-datasets.md` | Task-dataset format and model execution flow |
| `docs/reference/` | Architecture, configuration, and file map |

## CI

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Runs `ruff`, `ty`, and `pytest` on pushes and pull requests |
