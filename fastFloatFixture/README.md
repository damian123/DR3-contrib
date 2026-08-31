# Optional fast_float CSV fixture

This test-only path is enabled with `DR3_FASTFLOAT_FIXTURES=ON`. It downloads
upstream `fastfloat/fast_float` release `v8.2.10`, pinned both by tag and SHA256
archive hash. The default is off, so ordinary DR3 consumers fetch nothing.

`options.csv` contains one option per line. Every scalar is parsed with
`fast_float::from_chars` and `strtod`; their IEEE-754 bit patterns must match
(zero ULP). The parsed arrays are then priced through DR3 `VecD4D`, including a
tail row, and must exactly match the committed in-memory fixture path.

```sh
cmake -S . -B build-fastfloat \
  -DDR3_BUILD_TESTS=ON \
  -DDR3_BUILD_EXAMPLES=OFF \
  -DDR3_FASTFLOAT_FIXTURES=ON
cmake --build build-fastfloat --target FastFloatFixtureTest
ctest --test-dir build-fastfloat -R FastFloatFixture --output-on-failure
```
