# Building DR3

## Supported platforms and ISA

DR3's required CMake paths are exercised with Visual Studio, GCC, and Clang on
Windows and Ubuntu. `DR3_ISA=AVX2` is the default. The main VectorTest uses
`DRC::VecD4D` and therefore needs AVX2 at runtime. The separate
`VectorTestSSE42` target compiles the existing 128-bit `DRC::VecD2D` path with
`DR3_ISA=SSE4.2` (`-msse4.2` on GCC/Clang). AVX512 may be compiled with
`-DDR3_ISA=AVX512`, but do not execute that binary without runtime feature
detection. GCC/Clang AVX2 builds use the portable `-mavx2 -mfma` baseline;
they do not use `-march=native`.

For single-configuration generators, an omitted `CMAKE_BUILD_TYPE` defaults to
`Release` (or `Debug` when a sanitizer option is enabled). SIMD timings are
meaningless in an unoptimized Debug build, so pass an explicit build type when
comparing local measurements. Multi-configuration generators continue to use
their normal `--config` selection.

## Consumer build (no test dependency)

Tests default to off. This configures and builds the reusable Vectorisation,
lattice, and advanced libraries without fetching GoogleTest or building test,
example, or benchmark executables:

```sh
cmake -S . -B build-consumer \
  -DDR3_BUILD_TESTS=OFF \
  -DDR3_BUILD_EXAMPLES=OFF
cmake --build build-consumer --config Release
```

Projects that use `add_subdirectory(DR3)` can link `DR3::Vectorisation` and
include headers without a source-relative path:

```cmake
add_subdirectory(path/to/DR3)
target_link_libraries(my_target PRIVATE DR3::Vectorisation)
```

```cpp
#include "VecX/dr3.h"
```

The checked-in consumer verifies that path without Python or an installed
package. It also calls an advanced-library function so the transitive
`dr3_advanced -> dr3_lattice` link is checked rather than hidden inside an
unreferenced static archive:

```sh
cmake -S tests/cmake_consumer -B build-cmake-consumer -DDR3_ROOT="$PWD"
cmake --build build-cmake-consumer --config Release
```

## Tests

Request tests explicitly. This is the only path that fetches the pinned
GoogleTest dependency:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDR3_BUILD_TESTS=ON \
  -DDR3_BUILD_EXAMPLES=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The focused SSE4.2 path can be verified independently:

```sh
cmake -S . -B build-sse42 \
  -DCMAKE_BUILD_TYPE=Release \
  -DDR3_BUILD_TESTS=ON \
  -DDR3_BUILD_EXAMPLES=OFF \
  -DDR3_ISA=SSE4.2
cmake --build build-sse42 --target VectorTestSSE42
ctest --test-dir build-sse42 -R SSE42 --output-on-failure
```

CTest registrations use `add_test(COMMAND $<TARGET_FILE:...>)`, avoiding
post-build GoogleTest discovery failures on Windows multi-config generators.

## Options

- `DR3_BUILD_EXAMPLES=ON|OFF` (default `ON`)
- `DR3_BUILD_TESTS=ON|OFF` (default `OFF`)
- `DR3_BENCHMARKS=ON|OFF` (default `OFF`; reproducible numerical benchmark telemetry)
- `DR3_FASTFLOAT_FIXTURES=ON|OFF` (default `OFF`; pinned CSV parser test)
- `DR3_ENABLE_SLOW_NUMERICAL_TESTS=ON|OFF` (default `OFF`)
- `DR3_BUILD_BENCHMARKS=ON|OFF` (default `OFF`; advanced telemetry)
- `DR3_ISA=SSE2|SSE4.2|AVX2|AVX512` (default `AVX2`)
- `DR3_ENABLE_SANITIZERS=ON` (GCC/Clang AddressSanitizer and UBSan)
- `DR3_ENABLE_TSAN=ON` (GCC/Clang ThreadSanitizer)

The Greeks, stress, and Monte Carlo `--self-test` paths do not execute timing
loops and do not depend on `DR3_BENCHMARKS`. Invoking those programs without
`--self-test` is an explicit request for local timing output. Measurements
depend on CPU, compiler, ISA, and build type.

The optional numeric CSV fixture is the only path that fetches fast_float. It
pins release `v8.2.10` and its archive SHA256:

```sh
cmake -S . -B build-fastfloat \
  -DDR3_BUILD_TESTS=ON \
  -DDR3_BUILD_EXAMPLES=OFF \
  -DDR3_FASTFLOAT_FIXTURES=ON
cmake --build build-fastfloat --target FastFloatFixtureTest
ctest --test-dir build-fastfloat -R FastFloatFixture --output-on-failure
```

## Sanitizer status

ASan/UBSan is a required green CI gate, including leak detection. Allocator
registry storage remains valid through translation-unit teardown, cleanup
rejects live blocks, and pool-backed objects return storage to the same padded
size class from which it was allocated. Sanitizer, bounds, lifetime, undefined
behavior, or leak findings fail the job.

The separate TSan job remains focused on the allocator and concurrency suites;
ASan/UBSan and TSan are never combined in one executable.

On Linux hosts where TSan exits before `main` with `unexpected memory mapping`,
run the focused CTest command under `setarch x86_64 -R`. This disables address
randomization for the sanitizer process; it does not suppress race reports.
