# Deterministic k-means and anomaly scoring

This example assigns row-major float32 observations to their nearest centroid
using exact squared L2 distance, accumulates sums/counts, recomputes centroids,
and stops when assignments or centroids converge (or after the configured
iteration limit). Initialization is a deterministic shuffle controlled by the
provided seed. Equal-distance centroid ties retain the smaller centroid index.

The self-test compares scalar-double and AVX2/tail assignments, repeats the
seeded fit, and flags a query whose nearest-centroid squared distance exceeds
an explicit threshold.

```sh
./build/KMeansExample/KMeansExample --self-test
```
