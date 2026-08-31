# Small-batch MLP inference

This example implements float32 matrix-vector dense layers for batches of
1–16 predictions. The network is:

```
dense -> RMSNorm -> GELU -> dense -> softmax
```

`DenseLayer.h` provides scalar-double accumulation and DR3 AVX2 float32
paths for `output = weights × input + bias`. The SIMD path scores each
row through a non-owning span and handles dimensions that are not multiples
of eight. This is a deliberately small inference example, not a GEMM library,
tensor framework, training system, or ONNX runtime.

The self-test uses committed model weights, inputs, and expected outputs. It
checks scalar/SIMD agreement, the stored fixture, softmax sums, rejected
dimensions, batch limits, tails, and deterministic results for every batch
size from 1 through 16. The C++ build does not require Python.

## Build and self-test

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDR3_BUILD_TESTS=ON -DDR3_BUILD_EXAMPLES=ON
cmake --build build --config Release --target SmallBatchInference
./build/SmallBatchInference/SmallBatchInference --self-test
ctest --test-dir build -C Release -R SmallBatchInference --output-on-failure
```

The committed expected values can be displayed from the scalar reference with
`--emit-expected`.

## Local measurements

Running the binary without arguments measures `32->64->2`, `64->128->8`,
`128->256->16`, and `768->256->2`. It prints scalar and SIMD latency,
predictions per second, maximum absolute/relative error, compiler, instruction
set, and Debug/Release mode.

The output is a local measurement, not a universal speedup claim. Results
depend on CPU, compiler, instruction set, build type, batch size, and workload.
