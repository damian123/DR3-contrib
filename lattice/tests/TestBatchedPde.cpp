#include "batched_pde.h"

#include "../../Vectorisation/Curves/curve.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{

using dr3::lattice::ForwardPdeParameter;
using dr3::lattice::OptionType;
using dr3::lattice::PdeConfig;
using dr3::lattice::PdeScheme;
using dr3::lattice::VanillaOption;

PdeConfig config()
{
    return {PdeScheme::CrankNicolsonRannacher, 101, 100, 0.0, 400.0};
}

std::vector<VanillaOption> options(std::size_t count)
{
    std::vector<VanillaOption> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        result.push_back({80.0 + 8.0 * static_cast<double>(index % 6),
                          95.0 + 2.0 * static_cast<double>(index % 3),
                          0.15 + 0.025 * static_cast<double>(index % 5),
                          -0.005 + 0.0125 * static_cast<double>(index % 4),
                          0.005 * static_cast<double>(index % 3),
                          0.4 + 0.2 * static_cast<double>(index % 5),
                          index % 2 == 0 ? OptionType::Call : OptionType::Put});
    return result;
}

std::vector<double> scalarPrices(const std::vector<VanillaOption>& instruments)
{
    std::vector<double> result;
    for (const auto& option : instruments)
        result.push_back(dr3::lattice::europeanPdePrice(option, config()).price);
    return result;
}

void expectMatchesScalar(const std::vector<VanillaOption>& instruments)
{
    const auto scalar = scalarPrices(instruments);
    const auto batched = dr3::lattice::batchedEuropeanPdePrices(instruments, config());
    ASSERT_EQ(batched.prices.size(), scalar.size());
    for (std::size_t lane = 0; lane < scalar.size(); ++lane)
    {
        SCOPED_TRACE(lane);
        EXPECT_NEAR(batched.prices[lane], scalar[lane], 2.0e-11);
    }
}

double centralBump(const VanillaOption& option, ForwardPdeParameter parameter)
{
    auto up = option;
    auto down = option;
    const double base = parameter == ForwardPdeParameter::Rate
        ? option.rate : option.volatility;
    const double bump = 1.0e-5 * std::max(1.0, std::abs(base));
    if (parameter == ForwardPdeParameter::Rate)
    {
        up.rate += bump;
        down.rate -= bump;
    }
    else
    {
        up.volatility += bump;
        down.volatility -= bump;
    }
    return (dr3::lattice::europeanPdePrice(up, config()).price
            - dr3::lattice::europeanPdePrice(down, config()).price)
        / (2.0 * bump);
}

} // namespace

TEST(BatchedPde, OneLaneMatchesScalar)
{
    expectMatchesScalar(options(1));
}

TEST(BatchedPde, EveryActiveLaneMatchesScalar)
{
    expectMatchesScalar(options(8));
}

TEST(BatchedPde, NonMultipleOfSimdWidth)
{
    const auto instruments = options(2 * 4 + 1);
    const auto result = dr3::lattice::batchedEuropeanPdePrices(instruments, config());
    EXPECT_NE(instruments.size() % result.simdWidth, 0u);
    expectMatchesScalar(instruments);
}

TEST(BatchedPde, InactiveTailLanesIgnored)
{
    const auto instruments = options(5);
    const auto result = dr3::lattice::batchedEuropeanPdePrices(instruments, config());
    EXPECT_EQ(result.prices.size(), instruments.size());
    EXPECT_EQ(result.simdWidth, 4u);
}

TEST(BatchedPde, DifferentRatesPerLane)
{
    auto instruments = options(4);
    for (std::size_t lane = 0; lane < instruments.size(); ++lane)
        instruments[lane].rate = -0.02 + 0.03 * static_cast<double>(lane);
    expectMatchesScalar(instruments);
}

TEST(BatchedPde, DifferentVolatilitiesPerLane)
{
    auto instruments = options(4);
    for (std::size_t lane = 0; lane < instruments.size(); ++lane)
        instruments[lane].volatility = 0.1 + 0.15 * static_cast<double>(lane);
    expectMatchesScalar(instruments);
}

TEST(BatchedPde, DifferentSpotsPerLane)
{
    auto instruments = options(4);
    for (std::size_t lane = 0; lane < instruments.size(); ++lane)
        instruments[lane].spot = 60.0 + 35.0 * static_cast<double>(lane);
    expectMatchesScalar(instruments);
}

TEST(BatchedPde, LaneFailureReportsIndex)
{
    auto instruments = options(5);
    instruments[3].volatility = 0.0;
    try
    {
        (void)dr3::lattice::batchedEuropeanPdePrices(instruments, config());
        FAIL() << "expected invalid lane";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("lane 3"), std::string::npos);
    }
}

TEST(PdeSensitivity, VegaMatchesCentralBump)
{
    const auto instruments = options(5);
    const auto result = dr3::lattice::batchedEuropeanPdeForwardSensitivities(
        instruments, config(), ForwardPdeParameter::Volatility);
    for (std::size_t lane = 0; lane < instruments.size(); ++lane)
    {
        const double expected = centralBump(instruments[lane], ForwardPdeParameter::Volatility);
        EXPECT_NEAR(result.sensitivities[lane], expected,
                    1.0e-4 * std::max(1.0, std::abs(expected)));
    }
}

TEST(PdeSensitivity, RhoMatchesCentralBump)
{
    const auto instruments = options(5);
    const auto result = dr3::lattice::batchedEuropeanPdeForwardSensitivities(
        instruments, config(), ForwardPdeParameter::Rate);
    for (std::size_t lane = 0; lane < instruments.size(); ++lane)
    {
        const double expected = centralBump(instruments[lane], ForwardPdeParameter::Rate);
        EXPECT_NEAR(result.sensitivities[lane], expected,
                    1.0e-4 * std::max(1.0, std::abs(expected)));
    }
}

TEST(PdeSensitivity, CurveNodeSensitivityMatchesCentralBump)
{
    const auto instruments = options(3);
    const std::vector<double> pillars{0.0, 0.5, 1.0, 2.0};
    const std::vector<double> rates{0.01, 0.015, 0.02, 0.03};
    const auto result = dr3::lattice::batchedEuropeanPdeCurveNodeSensitivities(
        instruments, config(), pillars, rates);
    for (std::size_t node = 0; node < rates.size(); ++node)
    {
        const double bump = 1.0e-5;
        auto upRates = rates;
        auto downRates = rates;
        upRates[node] += bump;
        downRates[node] -= bump;
        auto up = instruments;
        auto down = instruments;
        const dr3::numerics::Curve<double> upCurve(pillars, upRates);
        const dr3::numerics::Curve<double> downCurve(pillars, downRates);
        for (std::size_t option = 0; option < instruments.size(); ++option)
        {
            up[option].rate = upCurve.evaluate(up[option].maturity);
            down[option].rate = downCurve.evaluate(down[option].maturity);
        }
        const auto upPrices = dr3::lattice::batchedEuropeanPdePrices(up, config()).prices;
        const auto downPrices = dr3::lattice::batchedEuropeanPdePrices(down, config()).prices;
        for (std::size_t option = 0; option < instruments.size(); ++option)
        {
            const double expected = (upPrices[option] - downPrices[option]) / (2.0 * bump);
            EXPECT_NEAR(result.nodeSensitivities[option][node], expected,
                        1.0e-4 * std::max(1.0, std::abs(expected)));
        }
    }
}

TEST(PdeSensitivity, PrimalResultUnchangedBySeeding)
{
    const auto instruments = options(7);
    const auto primal = dr3::lattice::batchedEuropeanPdePrices(instruments, config());
    const auto rate = dr3::lattice::batchedEuropeanPdeForwardSensitivities(
        instruments, config(), ForwardPdeParameter::Rate);
    const auto volatility = dr3::lattice::batchedEuropeanPdeForwardSensitivities(
        instruments, config(), ForwardPdeParameter::Volatility);
    EXPECT_EQ(rate.prices, primal.prices);
    EXPECT_EQ(volatility.prices, primal.prices);
}
