# Research Agenda

This project explores whether full neural networks, especially large language models,
can be simplified into sparser and more interpretable tensor programs by using a
reference dataset as the behavioral anchor.

## Motivation

Modern neural networks contain dense tensor programs with many interacting parameters,
activations, and nonlinearities. Some of that computation may be redundant for a fixed
reference distribution. The first question is not whether a smaller model is globally
equivalent, but whether parts of the original computation can be removed or rewritten
without changing behavior on a well-defined dataset.

## Core Idea

Work at a fine scale inside the tensor program. Candidate interventions may replace,
remove, or constrain individual weights, neurons, subspaces, activations, or nonlinear
pieces. Every intervention should be tied to a precise reference dataset and an
explicit equivalence or approximation criterion.

## Phase 1: Exact Reference-Dataset Simplification

Start with transformations that preserve the model's behavior on the reference dataset.

Examples to investigate:

- prune subspaces that are unused across the entire dataset
- remove weights or directions that provably do not affect recorded activations
- collapse redundant linear maps where the reference activations live in a lower-rank subspace
- identify activation patterns that make parts of the tensor program locally inactive

The output of this phase should be an equivalent model or tensor program for the
reference dataset, along with a proof or exhaustive empirical check over that dataset.

## Phase 2: Controlled Lossy Approximation

After exact simplification is understood, study approximations with explicit metrics.

Possible directions:

- replace SiLU with ReLU in regions where the reference activations make the change small
- prune neurons or activation paths whose values stay very close to zero
- sparsify weights or low-rank factors under bounded output drift
- replace dense subcomputations with interpretable sparse rules

Lossy work should report the reference dataset, metric, tolerance, and failure cases.

## Non-Goals For Now

- no implementation before the first concrete experiment is specified
- no claims about global model equivalence from reference-dataset evidence alone
- no benchmark chasing without a clear simplification hypothesis
- no lossy approximation until the exact-preserving baseline is mature enough to compare against

## Open Questions

- What is the right unit of intervention: weight, neuron, direction, matrix block, or graph edge?
- Which reference datasets expose enough behavior to make simplification meaningful?
- Can exact reference-dataset equivalence be certified cheaply for transformer blocks?
- Which simplifications improve interpretability rather than merely compressing parameters?
- How should failures be surfaced when a simplification holds on most but not all reference examples?
