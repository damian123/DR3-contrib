#include "batched_pde.h"
#include "parallel_executor.h"
#include "tree_pricers.h"

#include "../Vectorisation/VecX/vcl_latest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef DR3_BENCHMARK_ISA
#define DR3_BENCHMARK_ISA "unspecified"
#endif

namespace
{

using dr3::lattice::ForwardPdeParameter;
using dr3::lattice::OptionType;
using dr3::lattice::PdeConfig;
using dr3::lattice::PdeScheme;
using dr3::lattice::VanillaOption;
using Clock = std::chrono::steady_clock;

constexpr std::size_t width = 4;

struct BenchmarkFixture
{
    std::vector<VanillaOption> options;
    PdeConfig pde;
    std::size_t treeSteps;
    std::size_t threadCount;
};

struct Mode
{
    std::string name;
    std::vector<double> expected;
    std::vector<double> (*run)(const BenchmarkFixture&);
    double tolerance;
};

BenchmarkFixture makeFixture(bool quick)
{
    BenchmarkFixture fixture;
    const std::size_t count = quick ? 12 : 48;
    fixture.options.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        fixture.options.push_back({70.0 + 4.0 * static_cast<double>(index % 16),
                                   100.0,
                                   0.12 + 0.02 * static_cast<double>(index % 7),
                                   -0.01 + 0.01 * static_cast<double>(index % 6),
                                   0.005 * static_cast<double>(index % 4),
                                   0.5 + 0.25 * static_cast<double>(index % 5),
                                   index % 2 == 0 ? OptionType::Call : OptionType::Put});
    fixture.pde = {PdeScheme::CrankNicolsonRannacher,
                   quick ? 101u : 201u, quick ? 100u : 200u, 0.0, 400.0};
    fixture.treeSteps = quick ? 128 : 512;
    fixture.threadCount = std::max<std::size_t>(1,
        std::min<std::size_t>(4, std::thread::hardware_concurrency()));
    return fixture;
}

Vec4d pack(const std::array<double, width>& values)
{
    return {values[0], values[1], values[2], values[3]};
}

template <typename Getter>
Vec4d packOptions(const std::vector<VanillaOption>& options,
                  std::size_t begin,
                  std::size_t active,
                  Getter&& getter)
{
    std::array<double, width> values{};
    for (std::size_t lane = 0; lane < width; ++lane)
        values[lane] = getter(options[begin + std::min(lane, active - 1)]);
    return pack(values);
}

Vec4db callMask(const std::vector<VanillaOption>& options,
                std::size_t begin,
                std::size_t active)
{
    std::array<bool, width> calls{};
    for (std::size_t lane = 0; lane < width; ++lane)
        calls[lane] = options[begin + std::min(lane, active - 1)].type == OptionType::Call;
    return {calls[0], calls[1], calls[2], calls[3]};
}

std::vector<double> scalarTree(const BenchmarkFixture& fixture)
{
    std::vector<double> result;
    result.reserve(fixture.options.size());
    for (const auto& option : fixture.options)
        result.push_back(dr3::lattice::europeanTrinomial(
            option, {fixture.treeSteps}));
    return result;
}

std::vector<double> simdTree(const BenchmarkFixture& fixture)
{
    std::vector<double> result(fixture.options.size());
    const int steps = static_cast<int>(fixture.treeSteps);
    const int nodeCount = 2 * steps + 1;
    for (std::size_t begin = 0; begin < fixture.options.size(); begin += width)
    {
        const std::size_t active = std::min(width, fixture.options.size() - begin);
        const auto spot = packOptions(fixture.options, begin, active,
            [](const auto& option) { return option.spot; });
        const auto strike = packOptions(fixture.options, begin, active,
            [](const auto& option) { return option.strike; });
        const auto volatility = packOptions(fixture.options, begin, active,
            [](const auto& option) { return option.volatility; });
        const auto rate = packOptions(fixture.options, begin, active,
            [](const auto& option) { return option.rate; });
        const auto dividend = packOptions(fixture.options, begin, active,
            [](const auto& option) { return option.dividendYield; });
        const auto maturity = packOptions(fixture.options, begin, active,
            [](const auto& option) { return option.maturity; });
        const auto calls = callMask(fixture.options, begin, active);
        const Vec4d dt = maturity / static_cast<double>(steps);
        const Vec4d dx = volatility * sqrt(3.0 * dt);
        const Vec4d drift = rate - dividend - 0.5 * volatility * volatility;
        const Vec4d varianceTerm = (dt * volatility * volatility
            + drift * drift * dt * dt) / (dx * dx);
        const Vec4d driftTerm = drift * dt / dx;
        const Vec4d pu = 0.5 * (varianceTerm + driftTerm);
        const Vec4d pd = 0.5 * (varianceTerm - driftTerm);
        const Vec4d pm = 1.0 - varianceTerm;
        const Vec4d discount = exp(-rate * dt);
        std::vector<Vec4d> prices(static_cast<std::size_t>(nodeCount));
        std::vector<Vec4d> scratch(static_cast<std::size_t>(nodeCount));
        for (int node = 0; node < nodeCount; ++node)
        {
            const Vec4d stock = spot * exp(static_cast<double>(node - steps) * dx);
            const Vec4d call = select(stock > strike, stock - strike, Vec4d(0.0));
            const Vec4d put = select(strike > stock, strike - stock, Vec4d(0.0));
            prices[static_cast<std::size_t>(node)] = select(calls, call, put);
        }
        for (int level = 0; level < steps; ++level)
        {
            const int first = level + 1;
            const int final = nodeCount - level - 1;
            for (int node = first; node < final; ++node)
                scratch[static_cast<std::size_t>(node)] = discount
                    * (prices[static_cast<std::size_t>(node + 1)] * pu
                       + prices[static_cast<std::size_t>(node)] * pm
                       + prices[static_cast<std::size_t>(node - 1)] * pd);
            prices.swap(scratch);
        }
        for (std::size_t lane = 0; lane < active; ++lane)
            result[begin + lane] = prices[static_cast<std::size_t>(steps)][static_cast<int>(lane)];
    }
    return result;
}

std::vector<double> scalarPde(const BenchmarkFixture& fixture)
{
    std::vector<double> result;
    result.reserve(fixture.options.size());
    for (const auto& option : fixture.options)
        result.push_back(dr3::lattice::europeanPdePrice(option, fixture.pde).price);
    return result;
}

std::vector<double> simdPde(const BenchmarkFixture& fixture)
{
    return dr3::lattice::batchedEuropeanPdePrices(
        fixture.options, fixture.pde).prices;
}

std::vector<double> simdPdeSensitivity(const BenchmarkFixture& fixture)
{
    const auto result = dr3::lattice::batchedEuropeanPdeForwardSensitivities(
        fixture.options, fixture.pde, ForwardPdeParameter::Rate);
    auto combined = result.prices;
    combined.insert(combined.end(), result.sensitivities.begin(), result.sensitivities.end());
    return combined;
}

std::vector<double> scalarPdeAndRho(const BenchmarkFixture& fixture)
{
    auto result = scalarPde(fixture);
    for (const auto& option : fixture.options)
    {
        auto up = option;
        auto down = option;
        constexpr double bump = 1.0e-5;
        up.rate += bump;
        down.rate -= bump;
        result.push_back((dr3::lattice::europeanPdePrice(up, fixture.pde).price
                          - dr3::lattice::europeanPdePrice(down, fixture.pde).price)
                         / (2.0 * bump));
    }
    return result;
}

std::vector<double> serialSimdPortfolio(const BenchmarkFixture& fixture)
{
    return simdPde(fixture);
}

std::vector<double> parallelSimdPortfolio(const BenchmarkFixture& fixture)
{
    std::vector<double> result(fixture.options.size());
    const std::size_t jobCount = std::min(fixture.threadCount,
        (fixture.options.size() + width - 1) / width);
    dr3::lattice::parallelFor(jobCount, fixture.threadCount, [&](std::size_t job)
    {
        const std::size_t firstBatch = job
            * ((fixture.options.size() + width - 1) / width) / jobCount;
        const std::size_t finalBatch = (job + 1)
            * ((fixture.options.size() + width - 1) / width) / jobCount;
        const std::size_t begin = std::min(firstBatch * width, fixture.options.size());
        const std::size_t end = std::min(finalBatch * width, fixture.options.size());
        std::vector<VanillaOption> local(fixture.options.begin() + static_cast<std::ptrdiff_t>(begin),
                                         fixture.options.begin() + static_cast<std::ptrdiff_t>(end));
        const auto localPrices = dr3::lattice::batchedEuropeanPdePrices(local, fixture.pde).prices;
        std::copy(localPrices.begin(), localPrices.end(), result.begin() + static_cast<std::ptrdiff_t>(begin));
    });
    return result;
}

double checksum(const std::vector<double>& values)
{
    std::vector<double> weighted(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
        weighted[index] = values[index] * static_cast<double>(index + 1);
    return dr3::lattice::deterministicCompensatedSum(weighted);
}

void verify(const Mode& mode, const BenchmarkFixture& fixture)
{
    const auto first = mode.run(fixture);
    const auto second = mode.run(fixture);
    if (first != second)
        throw std::runtime_error(mode.name + " is not deterministic");
    if (first.size() != mode.expected.size())
        throw std::runtime_error(mode.name + " result size differs from its scalar reference");
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        const double scaledTolerance = mode.tolerance
            * std::max(1.0, std::abs(mode.expected[index]));
        if (!std::isfinite(first[index])
            || std::abs(first[index] - mode.expected[index]) > scaledTolerance)
            throw std::runtime_error(mode.name + " differs from its scalar reference at result "
                                     + std::to_string(index));
    }
    std::cout << "checksum " << std::left << std::setw(34) << mode.name
              << std::setprecision(17) << checksum(first) << '\n';
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

void measure(const Mode& mode,
             const BenchmarkFixture& fixture,
             std::size_t repetitions)
{
    // Warm the code and allocator pools before collecting samples.
    volatile double warmup = checksum(mode.run(fixture));
    (void)warmup;
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition)
    {
        const auto start = Clock::now();
        volatile double sampleChecksum = checksum(mode.run(fixture));
        const auto stop = Clock::now();
        (void)sampleChecksum;
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    const double middle = median(samples);
    std::vector<double> absoluteDeviations(samples.size());
    std::transform(samples.begin(), samples.end(), absoluteDeviations.begin(),
                   [middle](double sample) { return std::abs(sample - middle); });
    std::cout << "timing   " << std::left << std::setw(34) << mode.name
              << std::right << std::fixed << std::setprecision(3)
              << " median_ms=" << middle
              << " mad_ms=" << median(std::move(absoluteDeviations)) << '\n';
}

const char* compilerName()
{
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "unknown";
#endif
}

const char* buildType()
{
#ifdef NDEBUG
    return "Release";
#else
    return "Debug";
#endif
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        bool selfTest = false;
        bool quick = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument(argv[index]);
            if (argument == "--self-test") selfTest = true;
            else if (argument == "--quick") quick = true;
            else throw std::invalid_argument("unknown benchmark argument: " + argument);
        }
        const auto fixture = makeFixture(quick || selfTest);
        const auto treeReference = scalarTree(fixture);
        const auto pdeReference = scalarPde(fixture);
        const auto sensitivityReference = scalarPdeAndRho(fixture);
        const std::vector<Mode> modes{
            {"scalar tree", treeReference, scalarTree, 0.0},
            {"SIMD tree", treeReference, simdTree, 1.0e-11},
            {"scalar European PDE", pdeReference, scalarPde, 0.0},
            {"SIMD-batched European PDE", pdeReference, simdPde, 1.0e-11},
            {"SIMD PDE with forward sensitivity", sensitivityReference,
             simdPdeSensitivity, 1.0e-4},
            {"serial SIMD portfolio", pdeReference, serialSimdPortfolio, 1.0e-11},
            {"multithreaded SIMD portfolio", pdeReference, parallelSimdPortfolio, 1.0e-11}};

        std::cout << "DR3 numerical benchmark\n"
                  << "compiler=" << compilerName() << '\n'
                  << "build_type=" << buildType() << '\n'
                  << "isa=" << DR3_BENCHMARK_ISA << '\n'
                  << "cpu_threads=" << std::thread::hardware_concurrency() << '\n'
                  << "worker_threads=" << fixture.threadCount << '\n'
                  << "grid=" << fixture.pde.spaceSteps << 'x' << fixture.pde.timeSteps << '\n'
                  << "portfolio_size=" << fixture.options.size() << '\n'
                  << "tree_steps=" << fixture.treeSteps << '\n';
        for (const auto& mode : modes) verify(mode, fixture);
        if (!selfTest)
            for (const auto& mode : modes) measure(mode, fixture, quick ? 5 : 11);
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "NumericsBenchmark error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
