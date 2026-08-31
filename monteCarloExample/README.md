# Deterministic Monte Carlo example

The self-test uses a fixed xorshift seed, repeats the calculation to prove
determinism, reports its sampling error as `standard_error`, and checks the
result against the analytic Black-Scholes price within four reported standard
errors. Self-test mode contains no timing loop. A normal invocation prints a
local timing only, with no universal speedup claim.
