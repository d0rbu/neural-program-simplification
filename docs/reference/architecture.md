# Architecture

This repository starts as a compact correctness-first research scaffold. The first
source package contains reusable experiment infrastructure, not model transformations.

## Scaffold

The importable package exists because the first concrete experiment needs reusable code
for loading task datasets, validating answer-token placement, and scoring prompts with
causal language models.

| Module | Purpose |
|---|---|
| `neural_program_simplification.task_datasets` | Versioned task-dataset storage, prompt/answer phantom types, tokenizer validation, and JSON load/save helpers |
| `neural_program_simplification.model_execution` | Generic next-token model protocol and task-dataset scoring results |
| `neural_program_simplification.huggingface` | Lazy optional Hugging Face adapter for local causal LM loading |
| `tests/test_correctness_tools.py` | Executable examples for phantom types, runtime checks, array contracts, and property tests |

## Planned Shape

When implementation starts, likely modules include:

| Area | Purpose |
|---|---|
| reference data | load, validate, and fingerprint reference datasets used for equivalence claims |
| tracing | record activations and intervention-site statistics |
| exact transforms | propose and verify behavior-preserving tensor-program rewrites |
| lossy transforms | evaluate approximations under explicit metrics and tolerances |
| reports | summarize accepted and rejected simplification candidates |

## Correctness Boundary

Raw values should be refined near the boundary where they enter the system. Core code
should receive domain types such as `Probability`, not broad primitive values.

Array-heavy code should use `jaxtyping` for shape and dtype expectations and ordinary
runtime checks for semantic constraints such as non-negativity or finite values.

Reference-dataset equivalence should be represented as a first-class result with enough
metadata to reproduce the claim.

## Tests

`tests/` contains example tests and property tests. The default suite is intentionally
fast enough to run before every handoff.
