# Glossary

| Term | Meaning |
|---|---|
| Phantom type | A runtime primitive narrowed by a predicate and represented as a richer static type after validation. |
| Refinement function | A function such as `parse_probability` that validates raw input and returns a phantom type. |
| Runtime boundary | A place where untrusted values enter the system, such as CLI args, config files, data files, model outputs, or public APIs. |
| Property test | A Hypothesis test that checks an invariant across many generated examples. |
| Array contract | A dtype and shape expectation expressed with `jaxtyping` and enforced with `beartype`. |
| Reference dataset | The fixed dataset used as the behavioral anchor for equivalence and approximation claims. |
| Tensor program | The concrete computation graph induced by model weights, activations, and nonlinearities. |
| Intervention site | A specific weight, neuron, subspace, activation, nonlinearity, or tensor-program edge being changed. |
| Exact simplification | A transformation that preserves model behavior on the reference dataset under the stated criterion. |
| Lossy approximation | A transformation that changes behavior but stays within an explicit metric and tolerance. |
| Unused subspace | A direction or subspace that the reference activations never use, making it a candidate for exact pruning. |
