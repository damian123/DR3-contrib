#include "VecX/dr3.h"

#include "../ExampleSupport/finance_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace DRC::VecD4D;

constexpr double kPublishedCall = 10.450583572185565;
constexpr double kScalarTolerance = 1.0e-12;
constexpr double kSimdTolerance = 2.0e-11;

VecXX black_scholes_call_simd(const VecXX& spot,
                              const VecXX& strike,
                              const VecXX& time,
                              const VecXX& rate,
                              const VecXX& volatility)
{
    const auto root_time = sqrt(time);
    const auto d1 = (log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * time)
        / (volatility * root_time);
    const auto d2 = d1 - volatility * root_time;
    return spot * cdfnorm(d1) - strike * exp(-rate * time) * cdfnorm(d2);
}

int self_test()
{
    const double scalar = dr3::examples::black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20);
    const double scalar_error = std::fabs(scalar - kPublishedCall);
    const std::size_t size = 9U; // deliberately includes a Vec4d tail
    VecXX spots(std::vector<double>(size, 100.0));
    VecXX strikes(std::vector<double>(size, 100.0));
    VecXX times(std::vector<double>(size, 1.0));
    VecXX rates(std::vector<double>(size, 0.05));
    VecXX volatilities(std::vector<double>(size, 0.20));
    const VecXX simd = black_scholes_call_simd(spots, strikes, times, rates, volatilities);
    double max_simd_error = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        max_simd_error = std::max(max_simd_error, std::fabs(simd[i] - kPublishedCall));
    }
    std::cout << std::setprecision(16)
              << "published_call=" << kPublishedCall
              << " scalar_abs_error=" << scalar_error
              << " SIMD_abs_error=" << max_simd_error
              << " scalar_tolerance=" << kScalarTolerance
              << " SIMD_tolerance=" << kSimdTolerance << '\n';
    return scalar_error <= kScalarTolerance && max_simd_error <= kSimdTolerance ? 0 : 1;
}

int benchmark()
{
    constexpr std::size_t size = 10001U;
    constexpr std::size_t iterations = 200U;
    VecXX spots(std::vector<double>(size, 100.0));
    VecXX strikes(std::vector<double>(size, 100.0));
    VecXX times(std::vector<double>(size, 1.0));
    VecXX rates(std::vector<double>(size, 0.05));
    VecXX volatilities(std::vector<double>(size, 0.20));
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        checksum += black_scholes_call_simd(
            spots, strikes, times, rates, volatilities)[iteration % size];
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "local SIMD run: " << elapsed << " ms, checksum=" << checksum
              << ". Results depend on CPU, compiler, ISA, and build type."
              << " DR3_BENCHMARKS=" << DR3_BENCHMARKS << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        AllAllocatorsGuard<typename VecXX::SCALA_TYPE> allocator_guard;
        const bool is_self_test = argc > 1 && std::string(argv[1]) == "--self-test";
        if (argc > 1 && !is_self_test) throw std::invalid_argument("expected --self-test");
        // Self-test is deliberately free of timing loops, independent of the
        // DR3_BENCHMARKS configuration. A normal invocation is an explicit
        // request for local timings.
        return is_self_test ? self_test() : benchmark();
    } catch (const std::exception& error) {
        std::cerr << "portfolioGreeksExample failed: " << error.what() << '\n';
        return 1;
    }
}
