#include "dr3/advanced/heston.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
using dr3::advanced::EuropeanOption;
using dr3::advanced::Grid2D;
using dr3::advanced::HestonParameters;
using dr3::advanced::HestonPdeResult;
using dr3::advanced::HestonPdeSettings;
using dr3::advanced::HestonReferenceSettings;

HestonParameters baseParameters(double correlation = -0.5,
                                 EuropeanOption option = EuropeanOption::Call)
{
    return {100.0, 100.0, 1.0, 0.03, 0.01, 0.04, 0.04,
            1.5, 0.3, correlation, option};
}

double parityRight(const HestonParameters& parameters)
{
    return parameters.spot * std::exp(-parameters.dividendYield * parameters.maturity)
        - parameters.strike * std::exp(-parameters.rate * parameters.maturity);
}

TEST(HestonReference, PutCallParity)
{
    auto call = baseParameters();
    auto put = call;
    put.option = EuropeanOption::Put;
    EXPECT_NEAR(dr3::advanced::hestonReferencePrice(call)
                    - dr3::advanced::hestonReferencePrice(put),
                parityRight(call), 1.0e-9);
}

TEST(HestonReference, BlackScholesLimit)
{
    auto parameters = baseParameters();
    parameters.volatilityOfVariance = 0.0;
    parameters.initialVariance = parameters.longRunVariance = 0.04;
    EXPECT_NEAR(dr3::advanced::hestonReferencePrice(parameters),
        dr3::advanced::blackScholesPrice(EuropeanOption::Call, 100, 100, 1,
                                         0.03, 0.01, 0.2), 1.0e-12);
}

TEST(HestonReference, IntegrationRefinementStabilizes)
{
    HestonReferenceSettings coarse;
    coarse.initialIntervals = 128;
    coarse.maxRefinements = 1;
    HestonReferenceSettings medium = coarse;
    medium.initialIntervals = 256;
    HestonReferenceSettings fine = coarse;
    fine.initialIntervals = 512;
    const double coarsePrice = dr3::advanced::hestonReferencePrice(baseParameters(), coarse);
    const double mediumPrice = dr3::advanced::hestonReferencePrice(baseParameters(), medium);
    const double finePrice = dr3::advanced::hestonReferencePrice(baseParameters(), fine);
    EXPECT_LT(std::abs(finePrice - mediumPrice), std::abs(mediumPrice - coarsePrice));
}

TEST(HestonReference, FixedRegressionFixtures)
{
    EXPECT_NEAR(dr3::advanced::hestonReferencePrice(baseParameters()), 8.55995, 2.0e-5);
}

TEST(HestonReference, RejectsInvalidParameters)
{
    auto parameters = baseParameters();
    parameters.correlation = 1.1;
    EXPECT_THROW(dr3::advanced::hestonReferencePrice(parameters), std::invalid_argument);
    parameters = baseParameters();
    HestonReferenceSettings settings;
    settings.absoluteTolerance = 0.0;
    EXPECT_THROW(dr3::advanced::hestonReferencePrice(parameters, settings), std::invalid_argument);
}

TEST(HestonPde, ZeroCorrelationRemovesMixedDerivative)
{
    const auto parameters = baseParameters(0.0);
    HestonPdeSettings settings;
    const Grid2D grid(settings.spotAxis, settings.varianceAxis);
    const auto split = dr3::advanced::makeHestonOperator(parameters, grid);
    for (std::size_t index = 0; index < grid.size(); ++index)
        for (int second = -1; second <= 1; ++second)
            for (int first = -1; first <= 1; ++first)
                EXPECT_DOUBLE_EQ(split.explicitCoefficient(first, second, index), 0.0);
}

TEST(HestonPde, ZeroVolOfVolConstantVarianceMatchesBlackScholes)
{
    auto parameters = baseParameters();
    parameters.volatilityOfVariance = 0.0;
    const auto result = dr3::advanced::hestonPdePrice(parameters);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_NEAR(result.price,
        dr3::advanced::blackScholesPrice(EuropeanOption::Call, 100, 100, 1,
                                         0.03, 0.01, 0.2), 1.0e-12);
}

void expectMatchesReference(double correlation)
{
    const auto parameters = baseParameters(correlation);
    const double reference = dr3::advanced::hestonReferencePrice(parameters);
    const auto result = dr3::advanced::hestonPdePrice(parameters);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_NEAR(result.price, reference, 0.10);
}

TEST(HestonPde, MatchesReferenceWithPositiveCorrelation) { expectMatchesReference(0.5); }
TEST(HestonPde, MatchesReferenceWithNegativeCorrelation) { expectMatchesReference(-0.5); }
TEST(HestonPde, MatchesReferenceWithZeroCorrelation) { expectMatchesReference(0.0); }

TEST(HestonPde, FellerConditionSatisfiedCase)
{
    auto parameters = baseParameters();
    parameters.volatilityOfVariance = 0.2;
    const auto result = dr3::advanced::hestonPdePrice(parameters);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_TRUE(result.fellerConditionSatisfied);
}

TEST(HestonPde, FellerConditionViolatedCase)
{
    auto parameters = baseParameters();
    parameters.volatilityOfVariance = 0.5;
    const auto result = dr3::advanced::hestonPdePrice(parameters);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_FALSE(result.fellerConditionSatisfied);
}

TEST(HestonPde, ShortMaturity)
{
    auto parameters = baseParameters();
    parameters.maturity = 0.05;
    HestonPdeSettings settings;
    settings.spotAxis = dr3::advanced::Axis1D::uniform(0.0, 400.0, 161);
    settings.varianceAxis = dr3::advanced::Axis1D::uniform(0.0, 1.0, 61);
    const auto result = dr3::advanced::hestonPdePrice(parameters, settings);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_NEAR(result.price, dr3::advanced::hestonReferencePrice(parameters), 0.10);
}

TEST(HestonPde, LongMaturity)
{
    auto parameters = baseParameters();
    parameters.maturity = 5.0;
    HestonPdeSettings settings;
    settings.timeSteps = 200;
    const auto result = dr3::advanced::hestonPdePrice(parameters, settings);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_NEAR(result.price, dr3::advanced::hestonReferencePrice(parameters), 0.20);
}

TEST(HestonPde, CallIsIncreasingInSpot)
{
    auto lower = baseParameters();
    lower.spot = 90.0;
    auto upper = lower;
    upper.spot = 110.0;
    const auto lowerResult = dr3::advanced::hestonPdePrice(lower);
    const auto upperResult = dr3::advanced::hestonPdePrice(upper);
    ASSERT_TRUE(lowerResult.success && upperResult.success);
    EXPECT_LT(lowerResult.price, upperResult.price);
}

TEST(HestonPde, PutIsDecreasingInSpot)
{
    auto lower = baseParameters(-0.5, EuropeanOption::Put);
    lower.spot = 90.0;
    auto upper = lower;
    upper.spot = 110.0;
    const auto lowerResult = dr3::advanced::hestonPdePrice(lower);
    const auto upperResult = dr3::advanced::hestonPdePrice(upper);
    ASSERT_TRUE(lowerResult.success && upperResult.success);
    EXPECT_GT(lowerResult.price, upperResult.price);
}

TEST(HestonPde, PutCallParity)
{
    auto call = baseParameters();
    auto put = call;
    put.option = EuropeanOption::Put;
    const auto callResult = dr3::advanced::hestonPdePrice(call);
    const auto putResult = dr3::advanced::hestonPdePrice(put);
    ASSERT_TRUE(callResult.success && putResult.success);
    EXPECT_NEAR(callResult.price - putResult.price, parityRight(call), 0.08);
}

TEST(HestonPde, GridRefinementReducesError)
{
#ifdef DR3_ENABLE_SLOW_NUMERICAL_TESTS
    const auto parameters = baseParameters();
    const double reference = dr3::advanced::hestonReferencePrice(parameters);
    HestonPdeSettings coarse;
    coarse.spotAxis = dr3::advanced::Axis1D::uniform(0.0, 400.0, 41);
    coarse.varianceAxis = dr3::advanced::Axis1D::uniform(0.0, 1.0, 21);
    coarse.timeSteps = 40;
    coarse.nonnegativeTolerance = 1.0e-2;
    HestonPdeSettings fine;
    const auto coarseResult = dr3::advanced::hestonPdePrice(parameters, coarse);
    const auto fineResult = dr3::advanced::hestonPdePrice(parameters, fine);
    ASSERT_TRUE(coarseResult.success && fineResult.success)
        << "coarse=" << coarseResult.error << " fine=" << fineResult.error;
    EXPECT_LT(std::abs(fineResult.price - reference), std::abs(coarseResult.price - reference));
#else
    GTEST_SKIP() << "enable DR3_ENABLE_SLOW_NUMERICAL_TESTS for grid convergence";
#endif
}

TEST(HestonPde, TimeRefinementReducesError)
{
    const auto parameters = baseParameters();
    const double reference = dr3::advanced::hestonReferencePrice(parameters);
    HestonPdeSettings coarse;
    coarse.timeSteps = 20;
    HestonPdeSettings fine;
    const auto coarseResult = dr3::advanced::hestonPdePrice(parameters, coarse);
    const auto fineResult = dr3::advanced::hestonPdePrice(parameters, fine);
    ASSERT_TRUE(coarseResult.success && fineResult.success);
    EXPECT_LE(std::abs(fineResult.price - reference), std::abs(coarseResult.price - reference) + 1e-4);
}

TEST(HestonPde, RannacherReducesPayoffOscillation)
{
    auto parameters = baseParameters();
    parameters.maturity = 0.1;
    const double reference = dr3::advanced::hestonReferencePrice(parameters);
    HestonPdeSettings unsmoothed;
    unsmoothed.timeSteps = 12;
    unsmoothed.rannacherSteps = 0;
    HestonPdeSettings smoothed = unsmoothed;
    smoothed.rannacherSteps = 2;
    const auto first = dr3::advanced::hestonPdePrice(parameters, unsmoothed);
    const auto second = dr3::advanced::hestonPdePrice(parameters, smoothed);
    ASSERT_TRUE(first.success && second.success);
    EXPECT_LE(std::abs(second.price - reference), std::abs(first.price - reference) + 0.02);
}

TEST(HestonPde, AllValuesAreFinite)
{
    const HestonPdeResult result = dr3::advanced::hestonPdePrice(baseParameters());
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_FALSE(result.finalSurface.empty());
    EXPECT_TRUE(std::all_of(result.finalSurface.begin(), result.finalSurface.end(),
                            [](double value) { return std::isfinite(value) && value >= 0.0; }));
}

} // namespace
