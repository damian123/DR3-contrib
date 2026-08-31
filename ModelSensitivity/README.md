# Forward model sensitivity

`ModelSensitivity` evaluates a smooth logistic-regression score and reports
`dscore/dx[i]` for every input. Each input is activated separately through
DR3's existing forward `VecxD` / `D()` path and checked against a central
finite difference with step `1e-6` and absolute tolerance `2e-8`.

The fixture deliberately uses a smooth sigmoid. Non-differentiable functions
need an explicit policy: for example, a finite difference at a ReLU kink does
not define a unique derivative and must not be silently used as an AAD oracle.

Forward AAD is useful for model sensitivity when the number of active inputs
is small. It is not a replacement for reverse-mode differentiation or model
training.

```sh
./build/ModelSensitivity/ModelSensitivity --self-test
```
