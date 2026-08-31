# AI kernel primitives

These headers provide focused CPU inference utilities rather than a tensor or
BLAS framework:

- `VecX/quantization.h`: signed/unsigned 8-bit quantization, dequantization,
  AVX2 quantized dot products, and scale/zero-point correction.
- `VecX/preprocessing.h`: reductions, scaling, missing-value handling,
  column standardization, online/rolling statistics, and outlier filtering.
- `VecX/kmeans.h`: exact squared-L2 assignment and deterministic k-means.

Quantization uses one parameter pair per tensor (per-channel parameters are a
deliberate later extension), explicit round-half-away-from-zero semantics, and
saturation to the destination range. Dot products return `int32_t` and throw
`std::overflow_error` if the exact accumulated value is outside that range;
for worst-case signed inputs (`127 * 127`), at most 133,144 terms fit. The
implementation does not require VNNI and retains scalar tail/fallback paths.

Preprocessing calculations use double-precision references for aggregate
statistics. Constant columns standardize to zero. A missing value is a NaN;
infinities remain data and are rejected naturally by operations that require
finite bounds. `preprocessing_dr3.h` provides Vec overloads whose outlier
scores and threshold selection reuse DR3's existing `transform`, `reduce`,
and `filter` facilities.
