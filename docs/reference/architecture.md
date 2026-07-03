# Architecture

This repository currently starts as a compact, package-free research scaffold.

## Scaffold

The base repository intentionally starts without an importable source package. Add one
only when the first concrete experiment needs real reusable code.

| Module | Purpose |
|---|---|
| `tests/test_correctness_tools.py` | Executable examples for phantom types, runtime checks, array contracts, and property tests |

## Planned Shape

When implementation starts, likely modules include:

| Area | Purpose |
|---|---|
| reference data | load and fingerprint reference datasets used for equivalence claims |
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
