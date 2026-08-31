#include "../ExampleSupport/finance_reference.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> stress_prices()
{
    std::vector<double> prices;
    prices.reserve(17U);
    for (int scenario = -8; scenario <= 8; ++scenario) {
        const double spot = 100.0 * (1.0 + 0.025 * scenario);
        prices.push_back(dr3::examples::black_scholes_call(spot, 100.0, 1.0, 0.05, 0.20));
    }
    return prices;
}

int self_test()
{
    const auto prices = stress_prices();
    if (prices.size() != 17U) return 1;
    for (std::size_t i = 1; i < prices.size(); ++i) {
        if (!(prices[i] > prices[i - 1]) || !std::isfinite(prices[i])) return 1;
    }
    const double expected_base = 10.450583572185565;
    const double error = std::fabs(prices[8] - expected_base);
    std::cout << "stress self-test base_abs_error=" << error << " tolerance=1e-12\n";
    return error <= 1.0e-12 ? 0 : 1;
}

int benchmark()
{
    constexpr int iterations = 50000;
    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto prices = stress_prices();
        checksum += prices[static_cast<std::size_t>(iteration) % prices.size()];
    }
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "local stress run: " << elapsed << " ms, checksum=" << checksum
              << ". Results depend on CPU, compiler, ISA, and build type."
              << " DR3_BENCHMARKS=" << DR3_BENCHMARKS << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const bool is_self_test = argc > 1 && std::string(argv[1]) == "--self-test";
        if (argc > 1 && !is_self_test) throw std::invalid_argument("expected --self-test");
        return is_self_test ? self_test() : benchmark();
    } catch (const std::exception& error) {
        std::cerr << "portfolioStressExample failed: " << error.what() << '\n';
        return 1;
    }
}
