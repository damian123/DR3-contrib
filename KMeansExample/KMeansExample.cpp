#include "VecX/kmeans.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

int run(bool self_test)
{
    using namespace dr3::ai;
    constexpr std::size_t dimensions = 3U; // exercises the SIMD tail path
    const std::vector<float> fixture{
        -0.2F,  0.1F,  0.0F,
         0.0F, -0.1F,  0.2F,
         0.3F,  0.2F, -0.1F,
         9.8F, 10.1F, 10.0F,
        10.2F,  9.9F, 10.1F,
        10.0F, 10.3F,  9.8F};
    const KMeansOptions options{2U, 50U, 1.0e-6F, 20220831U};
    const auto result = kmeans(fixture.data(), fixture.size() / dimensions,
                               dimensions, options);
    const auto repeated = kmeans(fixture.data(), fixture.size() / dimensions,
                                 dimensions, options);
    require(result.converged, "k-means did not converge");
    require(result.assignments == repeated.assignments
            && result.centroids == repeated.centroids,
            "seeded k-means is not deterministic");
    const auto scalar_assignments = assign_nearest_centroids_scalar(
        fixture.data(), fixture.size() / dimensions, dimensions,
        result.centroids.data(), options.clusters);
    require(result.assignments == scalar_assignments,
            "scalar and SIMD assignments differ");
    require(std::all_of(result.counts.begin(), result.counts.end(),
                        [](std::size_t count) { return count == 3U; }),
            "unexpected cluster counts");

    const std::vector<float> queries{
        0.0F, 0.0F, 0.0F,
        10.0F, 10.0F, 10.0F,
        30.0F, 30.0F, 30.0F};
    std::vector<float> query_distances;
    const auto query_assignments = assign_nearest_centroids(
        queries.data(), queries.size() / dimensions, dimensions,
        result.centroids.data(), options.clusters, &query_distances);
    (void)query_assignments;
    const auto flags = anomaly_flags(query_distances, 25.0F);
    require(flags == std::vector<std::uint8_t>({0U, 0U, 1U}),
            "distance anomaly threshold failed");

    std::cout << "iterations=" << result.iterations
              << " converged=" << std::boolalpha << result.converged
              << " anomaly_flags=" << static_cast<int>(flags[0]) << ','
              << static_cast<int>(flags[1]) << ',' << static_cast<int>(flags[2])
              << '\n';
    if (self_test) std::cout << "KMeans self-test passed\n";
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const bool self_test = argc > 1 && std::string(argv[1]) == "--self-test";
        if (argc > 1 && !self_test) throw std::invalid_argument("expected --self-test");
        return run(self_test);
    } catch (const std::exception& error) {
        std::cerr << "KMeansExample failed: " << error.what() << '\n';
        return 1;
    }
}
