#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define DR3_KMEANS_HAS_AVX2 1
#else
#define DR3_KMEANS_HAS_AVX2 0
#endif

namespace dr3::ai {

inline double squared_l2_scalar_reference(const float* lhs,
                                          const float* rhs,
                                          std::size_t dimensions)
{
    if ((dimensions != 0U) && (lhs == nullptr || rhs == nullptr)) {
        throw std::invalid_argument("distance received a null buffer");
    }
    double result = 0.0;
    for (std::size_t i = 0; i < dimensions; ++i) {
        const double delta = static_cast<double>(lhs[i]) - rhs[i];
        result += delta * delta;
    }
    return result;
}

inline float squared_l2_simd(const float* lhs,
                             const float* rhs,
                             std::size_t dimensions)
{
    if ((dimensions != 0U) && (lhs == nullptr || rhs == nullptr)) {
        throw std::invalid_argument("distance received a null buffer");
    }
    float result = 0.0F;
    std::size_t i = 0;
#if DR3_KMEANS_HAS_AVX2
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8U <= dimensions; i += 8U) {
        const __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(lhs + i), _mm256_loadu_ps(rhs + i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(delta, delta));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    for (float lane : lanes) result += lane;
#endif
    for (; i < dimensions; ++i) {
        const float delta = lhs[i] - rhs[i];
        result += delta * delta;
    }
    return result;
}

struct KMeansOptions {
    std::size_t clusters{2};
    std::size_t max_iterations{100};
    float convergence_tolerance{1.0e-5F};
    std::uint32_t seed{5489U};
};

struct KMeansResult {
    std::vector<float> centroids;
    std::vector<std::size_t> assignments;
    std::vector<float> squared_distances;
    std::vector<std::size_t> counts;
    std::size_t iterations{0};
    bool converged{false};
};

inline void validate_kmeans_input(const float* rows,
                                  std::size_t row_count,
                                  std::size_t dimensions,
                                  const KMeansOptions& options)
{
    if (rows == nullptr) throw std::invalid_argument("k-means row buffer is null");
    if (row_count == 0U || dimensions == 0U) throw std::invalid_argument("k-means dimensions must be non-zero");
    if (options.clusters == 0U || options.clusters > row_count) {
        throw std::invalid_argument("cluster count must be between one and the row count");
    }
    if (options.max_iterations == 0U) throw std::invalid_argument("max_iterations must be non-zero");
    if (!(options.convergence_tolerance >= 0.0F) || !std::isfinite(options.convergence_tolerance)) {
        throw std::invalid_argument("invalid convergence tolerance");
    }
}

inline std::vector<std::size_t> assign_nearest_centroids(const float* rows,
                                                         std::size_t row_count,
                                                         std::size_t dimensions,
                                                         const float* centroids,
                                                         std::size_t cluster_count,
                                                         std::vector<float>* distances = nullptr)
{
    if ((row_count != 0U) && (rows == nullptr || centroids == nullptr)) {
        throw std::invalid_argument("assignment received a null buffer");
    }
    std::vector<std::size_t> assignments(row_count, 0U);
    if (distances != nullptr) distances->assign(row_count, 0.0F);
    for (std::size_t row = 0; row < row_count; ++row) {
        const float* point = rows + row * dimensions;
        std::size_t best_cluster = 0U;
        float best_distance = squared_l2_simd(point, centroids, dimensions);
        for (std::size_t cluster = 1; cluster < cluster_count; ++cluster) {
            const float distance = squared_l2_simd(
                point, centroids + cluster * dimensions, dimensions);
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster;
            }
        }
        assignments[row] = best_cluster;
        if (distances != nullptr) (*distances)[row] = best_distance;
    }
    return assignments;
}

inline std::vector<std::size_t> assign_nearest_centroids_scalar(const float* rows,
                                                                std::size_t row_count,
                                                                std::size_t dimensions,
                                                                const float* centroids,
                                                                std::size_t cluster_count)
{
    if ((row_count != 0U) && (rows == nullptr || centroids == nullptr)) {
        throw std::invalid_argument("assignment received a null buffer");
    }
    std::vector<std::size_t> assignments(row_count, 0U);
    for (std::size_t row = 0; row < row_count; ++row) {
        const float* point = rows + row * dimensions;
        double best_distance = squared_l2_scalar_reference(point, centroids, dimensions);
        for (std::size_t cluster = 1; cluster < cluster_count; ++cluster) {
            const double distance = squared_l2_scalar_reference(
                point, centroids + cluster * dimensions, dimensions);
            if (distance < best_distance) {
                best_distance = distance;
                assignments[row] = cluster;
            }
        }
    }
    return assignments;
}

inline KMeansResult kmeans(const float* rows,
                           std::size_t row_count,
                           std::size_t dimensions,
                           const KMeansOptions& options = {})
{
    validate_kmeans_input(rows, row_count, dimensions, options);
    std::vector<std::size_t> initial_indices(row_count);
    std::iota(initial_indices.begin(), initial_indices.end(), 0U);
    std::mt19937 generator(options.seed);
    // Use a specified Fisher-Yates mapping rather than std::shuffle so the
    // committed seed produces the same initialization across standard-library
    // implementations.
    for (std::size_t remaining = row_count; remaining > 1U; --remaining) {
        const std::size_t selected = static_cast<std::size_t>(generator()) % remaining;
        std::swap(initial_indices[remaining - 1U], initial_indices[selected]);
    }

    KMeansResult result;
    result.centroids.resize(options.clusters * dimensions);
    result.assignments.assign(row_count, std::numeric_limits<std::size_t>::max());
    result.squared_distances.resize(row_count);
    result.counts.resize(options.clusters);
    for (std::size_t cluster = 0; cluster < options.clusters; ++cluster) {
        const float* source = rows + initial_indices[cluster] * dimensions;
        std::copy(source, source + dimensions,
                  result.centroids.begin() + static_cast<std::ptrdiff_t>(cluster * dimensions));
    }

    std::vector<float> next_centroids(result.centroids.size());
    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        const auto assignments = assign_nearest_centroids(
            rows, row_count, dimensions, result.centroids.data(), options.clusters,
            &result.squared_distances);
        const bool unchanged = assignments == result.assignments;
        result.assignments = assignments;
        std::fill(next_centroids.begin(), next_centroids.end(), 0.0F);
        std::fill(result.counts.begin(), result.counts.end(), 0U);
        std::vector<double> sums(options.clusters * dimensions, 0.0);
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::size_t cluster = result.assignments[row];
            ++result.counts[cluster];
            for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
                sums[cluster * dimensions + dimension] += rows[row * dimensions + dimension];
            }
        }
        for (std::size_t cluster = 0; cluster < options.clusters; ++cluster) {
            for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
                const std::size_t offset = cluster * dimensions + dimension;
                next_centroids[offset] = result.counts[cluster] == 0U
                    ? result.centroids[offset]
                    : static_cast<float>(sums[offset] / static_cast<double>(result.counts[cluster]));
            }
        }
        float largest_shift = 0.0F;
        for (std::size_t cluster = 0; cluster < options.clusters; ++cluster) {
            largest_shift = std::max(largest_shift, squared_l2_simd(
                result.centroids.data() + cluster * dimensions,
                next_centroids.data() + cluster * dimensions,
                dimensions));
        }
        result.centroids.swap(next_centroids);
        result.iterations = iteration + 1U;
        if (unchanged || largest_shift <= options.convergence_tolerance * options.convergence_tolerance) {
            result.converged = true;
            break;
        }
    }
    result.assignments = assign_nearest_centroids(
        rows, row_count, dimensions, result.centroids.data(), options.clusters,
        &result.squared_distances);
    std::fill(result.counts.begin(), result.counts.end(), 0U);
    for (std::size_t assignment : result.assignments) ++result.counts[assignment];
    return result;
}

inline std::vector<std::uint8_t> anomaly_flags(const std::vector<float>& squared_distances,
                                               float threshold)
{
    if (!(threshold >= 0.0F) || !std::isfinite(threshold)) {
        throw std::invalid_argument("anomaly threshold must be finite and non-negative");
    }
    std::vector<std::uint8_t> flags(squared_distances.size());
    for (std::size_t i = 0; i < squared_distances.size(); ++i) {
        flags[i] = squared_distances[i] > threshold ? 1U : 0U;
    }
    return flags;
}

} // namespace dr3::ai
