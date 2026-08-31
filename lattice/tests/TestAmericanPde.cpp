#include "american_pde.h"
#include "tree_pricers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace
{

using dr3::lattice::OptionType;
using dr3::lattice::PdeConfig;
using dr3::lattice::PdeScheme;
using dr3::lattice::PsorConfig;
using dr3::lattice::VanillaOption;

VanillaOption standardPut()
{
    return {100.0, 100.0, 0.20, 0.05, 0.0, 1.0, OptionType::Put};
}

PdeConfig config(std::size_t spaceSteps = 401, std::size_t timeSteps = 400)
{
    return {PdeScheme::CrankNicolsonRannacher,
            spaceSteps, timeSteps, 0.0, 400.0};
}

PsorConfig psor()
{
    return {1.2, 1.0e-9, 10'000};
}

} // namespace

TEST(AmericanPde, RejectsInvalidOmega)
{
    auto invalid = psor();
    for (const double omega : {-0.1, 0.0, 2.0, 2.1})
    {
        invalid.omega = omega;
        EXPECT_THROW(dr3::lattice::americanPdePrice(standardPut(), config(), invalid),
                     std::invalid_argument);
    }
    invalid = psor();
    invalid.tolerance = 0.0;
    EXPECT_THROW(dr3::lattice::americanPdePrice(standardPut(), config(), invalid),
                 std::invalid_argument);
    invalid = psor();
    invalid.maximumIterations = 0;
    EXPECT_THROW(dr3::lattice::americanPdePrice(standardPut(), config(), invalid),
                 std::invalid_argument);
}

TEST(AmericanPde, ReportsNonConvergence)
{
    const PsorConfig forcedFailure{1.2, 1.0e-15, 1};
    const auto result = dr3::lattice::americanPdePrice(
        standardPut(), config(101, 20), forcedFailure);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 1u);
    EXPECT_GT(result.residual, forcedFailure.tolerance);
}

TEST(AmericanPde, ValueNeverBelowIntrinsic)
{
    const auto option = standardPut();
    const auto result = dr3::lattice::americanPdePrice(option, config(), psor());
    ASSERT_TRUE(result.converged);
    for (std::size_t index = 0; index < result.values.size(); ++index)
    {
        const double intrinsic = std::max(option.strike - result.grid[index], 0.0);
        EXPECT_GE(result.values[index] + 1.0e-11, intrinsic);
    }
}

TEST(AmericanPde, ValueNeverBelowEuropeanValue)
{
    const auto option = standardPut();
    const auto american = dr3::lattice::americanPdePrice(option, config(), psor());
    const auto european = dr3::lattice::europeanPdePrice(option, config());
    ASSERT_TRUE(american.converged);
    ASSERT_EQ(american.values.size(), european.values.size());
    for (std::size_t index = 0; index < american.values.size(); ++index)
        EXPECT_GE(american.values[index] + 2.0e-8, european.values[index]);
}

TEST(AmericanPde, ComplementarityResidualIsSmall)
{
    const auto settings = psor();
    const auto result = dr3::lattice::americanPdePrice(standardPut(), config(), settings);
    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.residual, settings.tolerance);
}

TEST(AmericanPde, PutMatchesHighResolutionTrinomial)
{
    const auto option = standardPut();
    const auto result = dr3::lattice::americanPdePrice(option, config(), psor());
    ASSERT_TRUE(result.converged);
    const double tree = dr3::lattice::americanTrinomial(option, {4096});
    EXPECT_NEAR(result.price, tree, 0.03);
}

TEST(AmericanPde, CallWithoutDividendMatchesEuropeanCall)
{
    auto option = standardPut();
    option.type = OptionType::Call;
    const auto american = dr3::lattice::americanPdePrice(option, config(), psor());
    const auto european = dr3::lattice::europeanPdePrice(option, config());
    ASSERT_TRUE(american.converged);
    EXPECT_NEAR(american.price, european.price, 0.02);
}

TEST(AmericanPde, PutIncreasesWithVolatility)
{
    auto low = standardPut();
    auto high = low;
    low.volatility = 0.15;
    high.volatility = 0.35;
    const auto lowResult = dr3::lattice::americanPdePrice(low, config(201, 200), psor());
    const auto highResult = dr3::lattice::americanPdePrice(high, config(201, 200), psor());
    ASSERT_TRUE(lowResult.converged);
    ASSERT_TRUE(highResult.converged);
    EXPECT_GT(highResult.price, lowResult.price);
}

TEST(AmericanPde, PutDecreasesWithSpot)
{
    auto lowSpot = standardPut();
    auto highSpot = lowSpot;
    lowSpot.spot = 90.0;
    highSpot.spot = 110.0;
    const auto lowResult = dr3::lattice::americanPdePrice(lowSpot, config(201, 200), psor());
    const auto highResult = dr3::lattice::americanPdePrice(highSpot, config(201, 200), psor());
    ASSERT_TRUE(lowResult.converged);
    ASSERT_TRUE(highResult.converged);
    EXPECT_GT(lowResult.price, highResult.price);
}

TEST(AmericanPde, RefinementStabilizesPrice)
{
    const auto option = standardPut();
    const auto coarse = dr3::lattice::americanPdePrice(option, config(51, 50), psor());
    const auto medium = dr3::lattice::americanPdePrice(option, config(101, 100), psor());
    const auto fine = dr3::lattice::americanPdePrice(option, config(201, 200), psor());
    ASSERT_TRUE(coarse.converged);
    ASSERT_TRUE(medium.converged);
    ASSERT_TRUE(fine.converged);
    EXPECT_LT(std::abs(fine.price - medium.price),
              std::abs(medium.price - coarse.price));
}
