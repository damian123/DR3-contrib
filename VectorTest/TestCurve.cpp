#include "pch.h"

#include "../Vectorisation/Curves/curve.h"
#include "../Vectorisation/VecX/dr3.h"
#include "NumericalTestUtils.h"
#include "testNamespace.h"

#include <cmath>
#include <limits>
#include <vector>

namespace
{

using dr3::numerics::Curve;
using dr3::numerics::ExtrapolationPolicy;
using dr3::numerics::InterpolationPolicy;
using Forward = VecxD;

Curve<double> standardCurve()
{
    return {{0.0, 1.0, 2.0, 4.0}, {0.01, 0.02, 0.04, 0.08}};
}

} // namespace

TEST(Curve, RejectsEmptyPillars)
{
    EXPECT_THROW((Curve<double>({}, {})), std::invalid_argument);
}

TEST(Curve, RejectsMismatchedPillarAndValueCounts)
{
    EXPECT_THROW((Curve<double>({0.0, 1.0}, {0.01})), std::invalid_argument);
}

TEST(Curve, RejectsUnsortedPillars)
{
    EXPECT_THROW((Curve<double>({0.0, 2.0, 1.0}, {0.01, 0.02, 0.03})),
                 std::invalid_argument);
}

TEST(Curve, RejectsDuplicatePillars)
{
    EXPECT_THROW((Curve<double>({0.0, 1.0, 1.0}, {0.01, 0.02, 0.03})),
                 std::invalid_argument);
}

TEST(Curve, RejectsNonFinitePillarsAndValues)
{
    EXPECT_THROW((Curve<double>({0.0, std::numeric_limits<double>::infinity()},
                                {0.01, 0.02})), std::invalid_argument);
    EXPECT_THROW((Curve<double>({0.0, 1.0},
                                {0.01, std::numeric_limits<double>::quiet_NaN()})),
                 std::invalid_argument);
}

TEST(Curve, ExactFirstPillar)
{
    EXPECT_DOUBLE_EQ(standardCurve().evaluate(0.0), 0.01);
}

TEST(Curve, ExactInteriorPillar)
{
    EXPECT_DOUBLE_EQ(standardCurve().evaluate(2.0), 0.04);
}

TEST(Curve, ExactFinalPillar)
{
    EXPECT_DOUBLE_EQ(standardCurve().evaluate(4.0), 0.08);
}

TEST(Curve, LinearMidpoint)
{
    EXPECT_DOUBLE_EQ(standardCurve().evaluate(1.5), 0.03);
}

TEST(Curve, LinearInterpolationAtKnownFraction)
{
    EXPECT_DOUBLE_EQ(standardCurve().evaluate(2.5), 0.05);
}

TEST(Curve, FlatLeftExtrapolation)
{
    EXPECT_DOUBLE_EQ(standardCurve().evaluate(-3.0), 0.01);
}

TEST(Curve, FlatRightExtrapolation)
{
    EXPECT_DOUBLE_EQ(standardCurve().evaluate(10.0), 0.08);
}

TEST(Curve, LinearAndFlatInterpolationPoliciesAreDistinct)
{
    const Curve<double> flat({0.0, 1.0, 2.0}, {1.0, 2.0, 4.0},
                             InterpolationPolicy::Flat,
                             ExtrapolationPolicy::Linear,
                             ExtrapolationPolicy::Linear);
    EXPECT_DOUBLE_EQ(flat.evaluate(0.5), 1.0);
    EXPECT_DOUBLE_EQ(flat.evaluate(-0.5), 0.5);
    EXPECT_DOUBLE_EQ(flat.evaluate(2.5), 5.0);
}

TEST(Curve, ResetInvalidatesDerivedState)
{
    auto curve = standardCurve();
    const auto revision = curve.revision();
    curve.reset({0.0, 2.0}, {0.03, 0.07});
    EXPECT_GT(curve.revision(), revision);
    EXPECT_DOUBLE_EQ(curve.evaluate(1.0), 0.05);
}

TEST(Curve, VectorValuesMatchScalarLaneByLane)
{
    const int length = static_cast<int>(VecXX::INS::size()) + 1;
    std::vector<VecXX> vectorValues;
    for (double base : {0.01, 0.02, 0.04})
    {
        std::vector<Numeric> lanes(static_cast<std::size_t>(length));
        for (int lane = 0; lane < length; ++lane)
            lanes[static_cast<std::size_t>(lane)] = asNumber(base + 0.001 * lane);
        vectorValues.emplace_back(lanes);
    }
    const Curve<VecXX> vectorCurve({0.0, 1.0, 2.0}, std::move(vectorValues));
    const auto result = vectorCurve.evaluate(1.25);
    ASSERT_EQ(result.size(), length);
    for (int lane = 0; lane < length; ++lane)
    {
        const Curve<double> scalar({0.0, 1.0, 2.0},
            {0.01 + 0.001 * lane, 0.02 + 0.001 * lane, 0.04 + 0.001 * lane});
        EXPECT_DOUBLE_EQ(result[static_cast<std::size_t>(lane)], scalar.evaluate(1.25));
    }
}

TEST(Curve, BulkSortedEvaluationMatchesScalarEvaluation)
{
    const auto curve = standardCurve();
    const std::vector<double> queries{-1.0, 0.0, 0.25, 1.0, 1.75, 4.0, 6.0};
    std::vector<double> output(queries.size());
    curve.evaluateSorted(queries, output);
    for (std::size_t index = 0; index < queries.size(); ++index)
        EXPECT_DOUBLE_EQ(output[index], curve.evaluate(queries[index]));
    std::vector<double> tooSmall(queries.size() - 1);
    EXPECT_THROW(curve.evaluateSorted(queries, tooSmall), std::invalid_argument);
    auto unsorted = queries;
    std::swap(unsorted[2], unsorted[3]);
    EXPECT_THROW(curve.evaluateSorted(unsorted, output), std::invalid_argument);
}

TEST(Curve, NonMultipleOfSimdWidth)
{
    const int length = 2 * static_cast<int>(VecXX::INS::size()) + 1;
    std::vector<Numeric> values0(static_cast<std::size_t>(length), asNumber(1.0));
    std::vector<Numeric> values1(static_cast<std::size_t>(length), asNumber(3.0));
    const Curve<VecXX> curve({0.0, 1.0}, {VecXX(values0), VecXX(values1)});
    const auto result = curve.evaluate(0.5);
    EXPECT_EQ(result.size(), length);
    EXPECT_NE(length % static_cast<int>(VecXX::INS::size()), 0);
    for (int lane = 0; lane < length; ++lane)
        EXPECT_DOUBLE_EQ(result[static_cast<std::size_t>(lane)], 2.0);
}

TEST(CurveSensitivity, LowerNodeWeightIsOneMinusFraction)
{
    const auto weights = standardCurve().nodeSensitivities(1.25);
    EXPECT_DOUBLE_EQ(weights[1], 0.75);
}

TEST(CurveSensitivity, UpperNodeWeightIsFraction)
{
    const auto weights = standardCurve().nodeSensitivities(1.25);
    EXPECT_DOUBLE_EQ(weights[2], 0.25);
}

TEST(CurveSensitivity, NonAdjacentNodeSensitivityIsZero)
{
    const auto weights = standardCurve().nodeSensitivities(1.25);
    EXPECT_DOUBLE_EQ(weights[0], 0.0);
    EXPECT_DOUBLE_EQ(weights[3], 0.0);
}

TEST(CurveSensitivity, AdjacentSensitivityWeightsSumToOne)
{
    const auto weights = standardCurve().nodeSensitivities(1.25);
    EXPECT_DOUBLE_EQ(weights[1] + weights[2], 1.0);
}

TEST(CurveSensitivity, DiscountFactorSensitivityMatchesFormula)
{
    const int width = static_cast<int>(VecXX::INS::size());
    std::vector<Numeric> lowerPrimal(static_cast<std::size_t>(width), asNumber(0.02));
    std::vector<Numeric> upperPrimal(static_cast<std::size_t>(width), asNumber(0.04));
    std::vector<Numeric> lowerDerivative(static_cast<std::size_t>(width), asNumber(1.0));
    std::vector<Numeric> upperDerivative(static_cast<std::size_t>(width), asNumber(0.0));
    const Curve<Forward> curve({1.0, 2.0},
        {Forward(VecXX(lowerPrimal), VecXX(lowerDerivative)),
         Forward(VecXX(upperPrimal), VecXX(upperDerivative))});
    constexpr double time = 1.25;
    const auto result = dr3::numerics::discountFactor(curve, time);
    const double zeroRate = 0.02 * 0.75 + 0.04 * 0.25;
    const double expected = -time * std::exp(-zeroRate * time) * 0.75;
    for (int lane = 0; lane < width; ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)], expected, 1.0e-12);
}

TEST(CurveSensitivity, MatchesCentralBump)
{
    const std::vector<double> pillars{0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
    const std::vector<double> zeroRates{0.01, 0.012, 0.015, 0.02, 0.025, 0.03};
    constexpr double query = 1.4;
    const auto forward = dr3::numerics::linearCurveNodeSensitivities<VecXX::INS>(
        pillars, zeroRates, query);
    ASSERT_EQ(forward.size(), zeroRates.size());
    for (std::size_t node = 0; node < zeroRates.size(); ++node)
    {
        const auto bumpedCurveValue = [&](double bumpedNode)
        {
            auto bumped = zeroRates;
            bumped[node] = bumpedNode;
            return Curve<double>(pillars, bumped).evaluate(query);
        };
        EXPECT_NEAR(forward[node],
                    dr3::test::centralDifference(bumpedCurveValue, zeroRates[node]),
                    1.0e-10);
    }
}
