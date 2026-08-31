#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct McResult {
    double price;
    double standard_error;
};

class DeterministicGenerator {
public:
    explicit DeterministicGenerator(std::uint64_t state) : state_(state) {}

    double uniform_open()
    {
        state_ ^= state_ >> 12U;
        state_ ^= state_ << 25U;
        state_ ^= state_ >> 27U;
        const std::uint64_t bits = state_ * 2685821657736338717ULL;
        return (static_cast<double>(bits >> 11U) + 0.5)
            / static_cast<double>(std::uint64_t{1} << 53U);
    }

    double normal()
    {
        constexpr double two_pi = 6.283185307179586476925286766559;
        return std::sqrt(-2.0 * std::log(uniform_open()))
            * std::cos(two_pi * uniform_open());
    }

private:
    std::uint64_t state_;
};

McResult price_call(std::size_t paths)
{
    if (paths < 2U) throw std::invalid_argument("at least two paths are required");
    constexpr double spot = 100.0;
    constexpr double strike = 100.0;
    constexpr double rate = 0.05;
    constexpr double volatility = 0.20;
    DeterministicGenerator generator(0x4d595df4d0f33173ULL);
    double sum = 0.0;
    double square_sum = 0.0;
    for (std::size_t path = 0; path < paths; ++path) {
        const double terminal = spot * std::exp(
            (rate - 0.5 * volatility * volatility) + volatility * generator.normal());
        const double discounted = std::exp(-rate) * std::max(terminal - strike, 0.0);
        sum += discounted;
        square_sum += discounted * discounted;
    }
    const double mean = sum / static_cast<double>(paths);
    const double sample_variance = (square_sum - sum * sum / static_cast<double>(paths))
        / static_cast<double>(paths - 1U);
    return {mean, std::sqrt(std::max(0.0, sample_variance) / static_cast<double>(paths))};
}

int self_test()
{
    constexpr std::size_t paths = 200000U;
    const McResult first = price_call(paths);
    const McResult second = price_call(paths);
    const double analytic = 10.450583572185565;
    const double error = std::fabs(first.price - analytic);
    const bool deterministic = first.price == second.price
        && first.standard_error == second.standard_error;
    const bool statistically_consistent = error <= 4.0 * first.standard_error;
    std::cout << "MC price=" << first.price
              << " standard_error=" << first.standard_error
              << " analytic_abs_error=" << error << '\n';
    return deterministic && statistically_consistent
        && std::isfinite(first.standard_error) ? 0 : 1;
}

int benchmark()
{
    const auto start = std::chrono::steady_clock::now();
    const McResult result = price_call(1000000U);
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "local MC run: " << elapsed << " ms, price=" << result.price
              << ", standard_error=" << result.standard_error
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
        std::cerr << "monteCarloExample failed: " << error.what() << '\n';
        return 1;
    }
}
