# DR3 advanced numerical stack

The `dr3_advanced` C++17 library contains independent AAD/curve and 2D PDE
stacks. AVX2 is the portable CI baseline. Reusable code is silent; only the
optional benchmark executable prints status.

## Reverse AAD and curves

`ReverseTape<Value>` is explicitly caller-owned. An active value records its
tape identity, stable node index, unique generation, and primal. Constants are
node-free. A tape rejects cross-tape values, stale handles, and an output from
another tape. `rewind` retains handles before its mark and invalidates removed
nodes. Capacity changes do not matter because edges contain indices, not
pointers.

`reverse(output, seed)` resets adjoints before the sweep, making repeated
sweeps deterministic. `AccumulateExistingAdjoints` instead propagates the
complete existing state plus the new seed; call `zeroAdjoints()` before an
independent accumulated sweep. Recording follows normal floating-point domain
semantics (`log(-1)` and similar expressions produce non-finite primals rather
than a special AAD exception). A reserved reverse sweep performs no allocation.

The traits specializations cover `double` and one AVX2 `Vec4d` register per
node. SIMD operations and adjoints are lane-local. The valid-lane reverse
overload masks its seed, preventing padded tail lanes from contributing.

`CurveEvaluationPlan` copies and validates its pillar/query inputs, then stores
fixed classifications, indices, weights, extrapolation policy, and a grid
signature. Evaluation also compares the exact pillar grid, so signature
collisions cannot accept a different grid. Plans have no mutating API or shared
cache; callers own values and outputs. The templated contract works for scalar,
SIMD, forward-dual, and reverse-active values.

## Two-dimensional PDEs

`Grid2D` owns two finite, strictly increasing axes with at least three points.
The first coordinate is contiguous. Its surfaces expose only the exact logical
cell count. All stencil functions take pre-sized caller output, leave boundary
cells unchanged, reject aliasing/shape mismatch, and allocate nothing.

`SplitOperator2D` represents `A=A0+A1+A2`. `A0` is the explicit nine-point
term, while `A1` and `A2` are directional tridiagonal terms. The Heston builder
places the reaction `-r` exactly once in `A1`. `AdiWorkspace::initialize`
validates `0 < theta <= 1` and positive time steps, allocates all stage/line
storage, and factors both directions. Douglas and Modified Craig-Sneyd steps
reuse those factors and allocate nothing. The source comments state the exact
MCS equations. Rannacher startup uses two caller-workspaced half Douglas steps
with `theta=1`.

The Heston solver accepts both Feller and non-Feller models and reports the
condition. At variance zero it uses nonnegative linear extrapolation from the
first two interior rows, the limiting closure for the degenerate PDE; the far
variance boundary uses zero normal gradient. Zero correlation leaves the mixed
operator bitwise zero. Numerical failure returns `success=false`, a NaN price,
and an error string. Returned surfaces clamp only stencil undershoots no larger
than the configured and documented default tolerance `2e-4`.

The independent semi-analytic reference uses the characteristic-function
probability integrals with Simpson refinement. It shares no grid, stencil, ADI,
or boundary code with the PDE solver and validates every tolerance control.

## PCR and telemetry

PCR preserves input coefficients and right-hand sides and double-buffers every
stage. Arbitrary sizes are extended to the next power of two with independent
identity rows, which cannot change the logical solution. Interior stage blocks
use one AVX2 register; boundary/tail rows use the identical scalar recurrence.
Thomas remains the default lattice solver. PCR rejects input/output aliasing
and non-finite or near-zero stage denominators.

Configure telemetry explicitly:

```sh
cmake -S . -B build-bench -DDR3_BUILD_TESTS=OFF \
  -DDR3_BUILD_EXAMPLES=OFF -DDR3_BUILD_BENCHMARKS=ON -DDR3_ISA=AVX2
cmake --build build-bench --config Release --target dr3_numerical_benchmarks
./build-bench/advanced/dr3_numerical_benchmarks --self-test
./build-bench/advanced/dr3_numerical_benchmarks \
  --warmups 2 --samples 7 --output dr3-numerical-benchmarks.json
```

Correctness is checked before timing. JSON schema version 1 records the kernel,
compiler/version, build, ISA, OS, hardware threads, dimensions, counts,
nanosecond units, separate setup and solve distributions, checksum, reference
error, and git commit when available. Checksums and fixtures are deterministic.
There is no fixed universal speedup claim: results depend on compiler, CPU,
problem shape, and memory behavior. Hosted-runner timing is an artifact for
comparison, never a merge gate, and the full timing mode is not a CTest.

Expensive convergence-order cases require
`DR3_ENABLE_SLOW_NUMERICAL_TESTS=ON`. CUDA remains absent by design; CPU-only
builds have no CUDA dependency. The external repeatable-CUDA-runner and
end-to-end transfer/setup gates must be demonstrated before adding a CUDA
target.
