# Correctness

The project bias is to make bad state unrepresentable. This matters especially for
equivalence claims: a transformation should carry the reference dataset, metric, and
tolerance in typed state rather than loose comments.

## Research Contracts

Exact simplification work should make these states explicit:

- reference dataset identity
- model or checkpoint identity
- intervention site
- equivalence criterion
- numerical tolerance, if any
- artifact path for measured evidence

Lossy approximation work should additionally record the degradation metric and the
accepted tolerance before any behavior-changing transformation is trusted.

## Phantom Types

Use `phantom-types` when a primitive type is too broad for a domain concept.

Examples in this repo:

- `Probability`: `float` in `[0, 1]`, demonstrated in `tests/test_correctness_tools.py`

Future project-specific phantom types may include non-empty dataset IDs, finite
tolerances, positive feature counts, layer indices, and validated intervention IDs.

Pattern:

1. Define the phantom type near the code that owns the domain concept.
2. Add a `parse_*` function that refines raw values.
3. Store only refined values in dataclasses and core APIs.
4. Use `st.from_type(YourType)` in property tests when a strategy exists.

## Runtime Checks

Use `beartype` at runtime boundaries and on small public functions where type violations
would otherwise become confusing downstream failures.

Do not decorate every private helper reflexively. Prefer validation at boundaries and
around domain invariants.

## Array Contracts

Use `jaxtyping` for NumPy, JAX, PyTorch, or other array-like values when shape and dtype
matter. Pair it with `beartype`:

```python
from beartype import beartype
from jaxtyping import Float64, jaxtyped

Vector = Float64[np.ndarray, "n"]

@jaxtyped(typechecker=beartype)
def normalize_weights(weights: Vector) -> Vector:
    ...
```

## Property Tests

Use Hypothesis for:

- normalization and conservation laws
- parser and serializer round trips
- shape-preserving transformations
- monotonicity and ordering invariants
- edge cases that are easy to miss with example tests

Keep generated examples bounded so the default test suite stays fast.
