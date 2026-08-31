# DR3

[![Build and tests](https://github.com/andyD123/DR3/actions/workflows/ci.yml/badge.svg)](https://github.com/andyD123/DR3/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

DR3 is a C++17 SIMD numerical-computing library and example suite for
vectorized data processing, automatic differentiation, financial pricing and
risk, and focused CPU analytics workloads.

The repository combines reusable libraries with deterministic examples,
reference checks, sanitizer coverage, and cross-platform CI. AVX2/FMA is the
default full-suite baseline; SSE4.2 has a focused compatibility path, and
AVX512 is available as an explicit opt-in build.

## What is included

| Area | Capabilities |
| --- | --- |
| SIMD core | Owning vectors, non-owning views, masks, transforms, filters, reductions, scans, and binned or compensated accumulation |
| Differentiation and curves | Forward sensitivities, caller-owned scalar/SIMD reverse AAD tapes, interpolation, and reusable curve evaluation plans |
| Financial numerics | Binomial, trinomial, and barrier pricing; European, American, and batched PDE solvers; sensitivities; and deterministic parallel execution |
| Advanced solvers | Two-dimensional ADI/Heston pricing, Rannacher startup, Thomas and PCR tridiagonal solvers, and versioned benchmark telemetry |
| Data and inference | Preprocessing, int8 quantization, exact embedding search, deterministic k-means, neural kernels, and small-batch MLP inference |
| Verification | CTest reference checks plus MSVC/GCC/Clang, SSE4.2, ASan/UBSan, TSan, and repeated concurrency CI paths |

The data and inference components are focused CPU kernels and examples, not a
tensor, BLAS, ANN, training, or ONNX framework.

## Requirements and ISA selection

- CMake 3.17 or newer
- A C++17 compiler; CI exercises MSVC on Windows and GCC/Clang on Ubuntu
- An x86 or x86-64 processor appropriate for the selected instruction set

`DR3_ISA` accepts `SSE2`, `SSE4.2`, `AVX2`, or `AVX512`. The full CI baseline
uses `AVX2`; the separate `VectorTestSSE42` target covers the 128-bit
`DRC::VecD2D` path. CMake selects an ISA for each target, so only run a binary
on hardware that supports the selected instructions. In particular, do not run
an AVX512 build on a processor without AVX512 support.

The current tree is CPU-only and has no CUDA build dependency.

## Build and test

From the repository root:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDR3_BUILD_TESTS=ON \
  -DDR3_BUILD_EXAMPLES=ON \
  -DDR3_ISA=AVX2
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

`DR3_BUILD_TESTS` defaults to `OFF`. Enabling it fetches GoogleTest v1.14.0
and registers the test and self-test executables with CTest. Examples default
to `ON`.

For a reusable-library build with no test dependency or example executables:

```sh
cmake -S . -B build-consumer \
  -DDR3_BUILD_TESTS=OFF \
  -DDR3_BUILD_EXAMPLES=OFF \
  -DDR3_ISA=AVX2
cmake --build build-consumer --config Release
```

See the [build guide](docs/Build.md) for the focused SSE4.2 path, sanitizer
builds, slow numerical tests, optional fixtures, benchmark targets, and every
CMake option.

## Use DR3 from CMake

DR3 currently supports build-tree integration through `add_subdirectory`:

```cmake
add_subdirectory(path/to/DR3)
target_link_libraries(my_target PRIVATE DR3::Vectorisation)
```

```cpp
#include "VecX/dr3.h"
```

Link `dr3_lattice` for the tree and one-dimensional PDE library, or
`dr3_advanced` for the advanced numerical stack. The checked-in
[`tests/cmake_consumer`](tests/cmake_consumer/) project verifies the vector and
advanced-library integration path.

## Project guide

- [`Vectorisation/`](Vectorisation/) contains the SIMD containers and
  algorithms, curves, forward AD, similarity, preprocessing, quantization, and
  neural kernels.
- [`lattice/`](lattice/) contains tree pricers, one-dimensional PDE solvers,
  sensitivities, tridiagonal solvers, and deterministic parallel execution.
- [`advanced/`](advanced/) contains reverse AAD, reusable curve plans,
  two-dimensional ADI/Heston, PCR, and deterministic benchmark telemetry.
- [`GettingStarted/`](GettingStarted/) and
  [`accumulateExample/`](accumulateExample/) introduce vector operations,
  filters, views, and accumulation.
- [`portfolioGreeksExample/`](portfolioGreeksExample/),
  [`portfolioStressExample/`](portfolioStressExample/),
  [`monteCarloExample/`](monteCarloExample/), and
  [`ModelSensitivity/`](ModelSensitivity/) provide deterministic finance and
  sensitivity examples.
- [`KMeansExample/`](KMeansExample/),
  [`EmbeddingSearch/`](EmbeddingSearch/), and
  [`SmallBatchInference/`](SmallBatchInference/) provide focused data and
  inference examples with committed self-test fixtures.

Detailed design references:

- [Building DR3](docs/Build.md)
- [Concurrency contract](CONCURRENCY.md)
- [Advanced numerical stack](advanced/docs/AdvancedNumerics.md)

## Benchmarking

Use documented `--self-test` modes when validating correctness; they do not run
timing loops. Timing output depends on the CPU, compiler, ISA, build type,
problem shape, and workload, so DR3 does not claim a universal speedup.
Advanced benchmark telemetry validates deterministic checksums before timing
and can emit versioned JSON as described in the
[advanced numerical guide](advanced/docs/AdvancedNumerics.md#pcr-and-telemetry).

## License

DR3 is available under the [Apache License 2.0](LICENSE).
