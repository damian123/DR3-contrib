# Concurrency contract

DR3 supports concurrent calls only under the ownership rules below. A clean
stress or sanitizer run is supporting evidence, not a guarantee that arbitrary
sharing is safe.

| API or state | Sharing class | Mutation and ownership rule | Teardown rule |
|---|---|---|---|
| `Vec`, `VecD`, `VecBool`, and `VecView` values | Independent instances | Different threads may own different values. Concurrent mutation of one value is unsupported. Pool allocation/free is synchronized. | A value may be destroyed on another joined worker because pool return is synchronized. |
| Allocator registry and pools | Internally synchronized | Allocation, return, policy creation, and cleanup hold the registry mutex. Callers must not retain raw pool pointers after return. | `freeAll()` is permitted only after all pool-backed values are destroyed and workers are joined. |
| Immutable curves | Concurrent read-only | Concurrent evaluation is allowed only while no thread resets or mutates curve data. | The curve owner outlives all readers. |
| Tree pricers | Independent calls | Inputs and outputs are caller-owned; temporary `Vec` allocations use the synchronized pool. | All calls finish before allocator-global cleanup. |
| European PDE solver | Independent calls | Each call owns its grid, result, and `PdeWorkspace`. Do not share mutable result vectors. | Results may outlive the call and use standard container ownership. |
| American PDE solver | Independent calls | Each call owns its grid, obstacle, result, and PSOR workspace. Input option/config objects may be shared read-only. | Results use standard container ownership and may outlive the call. |
| Batched PDE solver | Independent calls | Each call owns its SIMD coefficients, factorization workspace, and output. Input batches/config objects may be shared read-only. | Results use standard container ownership and may outlive the call. |
| Parallel executor | Internally synchronized coordination | Workers own disjoint output ranges and per-call workspaces. The caller must keep read-only inputs alive until return. | Every worker is joined before the executor returns or rethrows a captured worker exception. |

The allocator mutex is acquired only when storage is obtained or returned. It
is not acquired by SIMD arithmetic, tree rollback, or PDE time stepping.

## Unsupported use

- Concurrent reads and writes to the same `Vec`, `VecD`, result vector, or
  workspace are unsupported.
- Curve mutation while another thread reads the curve is unsupported.
- Calling allocator-global cleanup while workers or pool-backed values are
  alive is unsupported.
- Sharing one mutable solver output or workspace between concurrent calls is
  unsupported; use independent calls and caller-owned results.

Debug assertions may be added for ownership violations that can be detected
without adding state to numerical hot loops. Tests use independent values,
fixed work partitions, synchronized starts, and joined workers.
