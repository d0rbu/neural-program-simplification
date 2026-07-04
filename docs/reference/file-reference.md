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
| `neural_program_simplification/types.py` | Shared phantom types for non-empty strings, model ids, and task text |
| `neural_program_simplification/task_datasets.py` | Task-document domain types and JSON storage |
| `neural_program_simplification/task_dataset_library.py` | Built-in task dataset registry and loaders |
| `neural_program_simplification/model_execution.py` | Hugging Face tokenization, Torch tensor batching, behavior masks, and masked causal LM loss |
| `neural_program_simplification/huggingface.py` | Thin Hugging Face causal-LM loader |
| `neural_program_simplification/data/task_datasets/*.json` | Packaged starter task datasets |

## Tests

| File | Purpose |
|---|---|
| `tests/test_correctness_tools.py` | Phantom type, runtime check, array contract, and property-test examples |
| `tests/test_task_datasets.py` | Task-dataset storage tests |
| `tests/test_builtin_task_datasets.py` | Built-in dataset registry, default-tokenizer compatibility, and fake-model execution tests |
| `tests/test_model_execution.py` | Tokenization, behavior-mask, batching, and masked-loss tests |

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
