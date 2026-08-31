# Portfolio Greeks reference check

The self-test pins the canonical European call fixture `S=K=100`, `T=1`,
`r=0.05`, and `sigma=0.20` to `10.450583572185565`. This fixture is the
worked Black-Scholes example published in Espen Gaarder Haug, *The Complete
Guide to Option Pricing Formulas*, 2nd edition, Black-Scholes chapter. The
scalar absolute tolerance is `1e-12`; the DR3 `VecD4D` AVX2/tail tolerance is
`2e-11`.

`--self-test` performs no timing. A normal invocation prints measurements from
the current machine; results depend on CPU, compiler, ISA, and build type, and
the README makes no universal speedup claim.
