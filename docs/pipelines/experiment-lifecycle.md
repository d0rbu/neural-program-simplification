# Experiment Lifecycle

Use this as the default lifecycle for neural program simplification experiments.

## 1. Define the Question

Write down the hypothesis, metric, and expected failure modes before adding code.

For this repo, also state whether the experiment is exact-preserving or lossy.

## 2. Make State Explicit

Represent raw config as validated dataclasses. Use phantom types for values that have
domain bounds such as probabilities, positive counts, feature IDs, seeds, and split
fractions.

Always make the reference dataset and model identity explicit. A simplification claim
without those two anchors is underspecified.

## 3. Build Small Reusable Units

Keep reusable logic in a real module once the project has source code. Keep one-off
orchestration in scripts or notebooks that call reusable code.

## 4. Test Invariants

Add example tests for known cases and property tests for broad invariants.

Exact-preserving experiments should include a reference-dataset equivalence check.
Lossy experiments should include tolerance-bound checks and examples of known failures.

## 5. Record Outputs

Keep generated artifacts out of git by default. Put durable notes in docs or experiment
reports, and make artifact paths explicit.

Record both successful simplifications and rejected candidates. Negative evidence is
part of the research result.
