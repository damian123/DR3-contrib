#include "pch.h"

#include "../Vectorisation/VecX/dr3.h"
#include "../Vectorisation/VecX/forward_ad_black_scholes.h"
#include "NumericalTestUtils.h"
#include "testNamespace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace
{

using Forward = VecxD;

std::vector<int> testLengths()
{
    const int width = static_cast<int>(VecXX::INS::size());
    return {1, width - 1, width, width + 1, 2 * width + 1};
}

std::vector<Numeric> values(int length, double first = 0.8, double step = 0.07)
{
    std::vector<Numeric> result(static_cast<std::size_t>(length));
    for (int index = 0; index < length; ++index)
    {
        result[static_cast<std::size_t>(index)] = asNumber(first + step * index);
    }
    return result;
}

double normalCdf(double value)
{
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double normalDensity(double value)
{
    constexpr double inverseRootTwoPi = 0.39894228040143267794;
    return inverseRootTwoPi * std::exp(-0.5 * value * value);
}

struct BlackScholesFixture
{
    std::vector<Numeric> spot;
    std::vector<Numeric> strike;
    std::vector<Numeric> volatility;
    std::vector<Numeric> rate;
    std::vector<Numeric> dividend;
    std::vector<Numeric> maturity;
};

BlackScholesFixture blackScholesFixture(int length)
{
    BlackScholesFixture fixture;
    for (int index = 0; index < length; ++index)
    {
        fixture.spot.push_back(asNumber(80.0 + 5.0 * index));
        fixture.strike.push_back(asNumber(100.0));
        fixture.volatility.push_back(asNumber(0.15 + 0.01 * index));
        fixture.rate.push_back(asNumber(0.01 + 0.002 * index));
        fixture.dividend.push_back(asNumber(0.005 + 0.001 * index));
        fixture.maturity.push_back(asNumber(0.5 + 0.1 * index));
    }
    return fixture;
}

enum class Seed
{
    Spot,
    Volatility,
    Rate,
    Maturity
};

Forward blackScholesAd(const BlackScholesFixture& fixture, Seed seed)
{
    const VecXX spot(fixture.spot);
    const VecXX strike(fixture.strike);
    const VecXX volatility(fixture.volatility);
    const VecXX rate(fixture.rate);
    const VecXX dividend(fixture.dividend);
    const VecXX maturity(fixture.maturity);
    return dr3::numerics::blackScholesCallForwardAd(
        seed == Seed::Spot ? D(spot) : C(spot),
        C(strike),
        seed == Seed::Volatility ? D(volatility) : C(volatility),
        seed == Seed::Rate ? D(rate) : C(rate),
        C(dividend),
        seed == Seed::Maturity ? D(maturity) : C(maturity));
}

template <typename Expected>
void expectBlackScholesSensitivity(Seed seed, Expected&& expected)
{
    const int length = static_cast<int>(VecXX::INS::size()) + 1;
    const auto fixture = blackScholesFixture(length);
    const auto result = blackScholesAd(fixture, seed);
    ASSERT_EQ(result.size(), static_cast<std::size_t>(length));
    for (int lane = 0; lane < length; ++lane)
    {
        SCOPED_TRACE(lane);
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)],
                    expected(fixture, static_cast<std::size_t>(lane)), 1.0e-7);
    }
}

} // namespace

TEST(ForwardAD, SeedVariableWithOnes)
{
    for (const int length : testLengths())
    {
        const auto variable = D(VecXX(values(length)));
        ASSERT_EQ(variable.size(), static_cast<std::size_t>(length));
        for (int lane = 0; lane < length; ++lane)
            EXPECT_DOUBLE_EQ(variable.derivative()[static_cast<std::size_t>(lane)], 1.0);
    }
}

TEST(ForwardAD, SeedConstantWithZeros)
{
    for (const int length : testLengths())
    {
        const auto constant = C(VecXX(values(length)));
        for (int lane = 0; lane < length; ++lane)
            EXPECT_DOUBLE_EQ(constant.derivative()[static_cast<std::size_t>(lane)], 0.0);
    }
}

TEST(ForwardAD, AllSupportedFactoriesCompileAndPreserveShape)
{
    const int length = static_cast<int>(VecXX::INS::size()) + 1;
    const VecXX input(values(length));
    const auto dVector = D(input);
    const auto cVector = C(input);
    const auto dBroadcast = D<VecXX::INS>(asNumber(2.0), length);
    const auto cBroadcast = C<VecXX::INS>(asNumber(2.0), length);
    EXPECT_EQ(dVector.size(), cVector.size());
    EXPECT_EQ(dBroadcast.size(), cBroadcast.size());
    EXPECT_FALSE(dVector.isScalar());
    EXPECT_FALSE(dBroadcast.isScalar());
}

TEST(ForwardAD, ScalarMinusVariableHasNegativeDerivative)
{
    const auto result = asNumber(5.0) - D(VecXX(values(5)));
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        EXPECT_DOUBLE_EQ(result.derivative()[static_cast<std::size_t>(lane)], -1.0);
}

TEST(ForwardAD, AdditionRule)
{
    const auto xValues = values(5);
    const auto result = D(VecXX(xValues)) + D(VecXX(xValues));
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)], 2.0, 1.0e-12);
}

TEST(ForwardAD, SubtractionRule)
{
    const auto xValues = values(5);
    const auto result = D(VecXX(xValues)) - C(VecXX(xValues));
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)], 1.0, 1.0e-12);
}

TEST(ForwardAD, ProductRule)
{
    const auto xValues = values(5);
    const auto result = D(VecXX(xValues)) * D(VecXX(xValues));
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)],
                    2.0 * xValues[static_cast<std::size_t>(lane)], 1.0e-12);
}

TEST(ForwardAD, QuotientRule)
{
    const auto xValues = values(5);
    const auto result = D(VecXX(xValues)) / (D(VecXX(xValues)) + 2.0);
    for (std::size_t lane = 0; lane < result.size(); ++lane)
    {
        const double denominator = xValues[static_cast<std::size_t>(lane)] + 2.0;
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)],
                    2.0 / (denominator * denominator), 1.0e-12);
    }
}

TEST(ForwardAD, ExpLogChainRule)
{
    const auto xValues = values(5);
    const auto result = exp(log(D(VecXX(xValues))));
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)], 1.0, 1.0e-12);
}

TEST(ForwardAD, SqrtRuleForPositiveInputs)
{
    const auto xValues = values(5);
    const auto result = sqrt(D(VecXX(xValues)));
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)],
                    0.5 / std::sqrt(xValues[static_cast<std::size_t>(lane)]), 1.0e-12);
}

TEST(ForwardAD, CompositeExpressionMatchesCentralDifference)
{
    const auto xValues = values(7, 1.1, 0.09);
    const auto x = D(VecXX(xValues));
    const auto result = exp(x) / sqrt(x + 0.5) + log(x * x + 1.0);
    for (std::size_t lane = 0; lane < result.size(); ++lane)
    {
        const double input = xValues[static_cast<std::size_t>(lane)];
        const auto scalar = [](double value)
        {
            return std::exp(value) / std::sqrt(value + 0.5)
                + std::log(value * value + 1.0);
        };
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)],
                    dr3::test::centralDifference(scalar, input), 1.0e-5);
    }
}

TEST(ForwardAD, BlackScholesDeltaMatchesAnalytic)
{
    expectBlackScholesSensitivity(Seed::Spot, [](const auto& f, std::size_t lane)
    {
        const double rootTime = std::sqrt(f.maturity[lane]);
        const double d1 = (std::log(f.spot[lane] / f.strike[lane])
            + (f.rate[lane] - f.dividend[lane]
               + 0.5 * f.volatility[lane] * f.volatility[lane]) * f.maturity[lane])
            / (f.volatility[lane] * rootTime);
        return std::exp(-f.dividend[lane] * f.maturity[lane]) * normalCdf(d1);
    });
}

TEST(ForwardAD, BlackScholesVegaMatchesAnalytic)
{
    expectBlackScholesSensitivity(Seed::Volatility, [](const auto& f, std::size_t lane)
    {
        const double rootTime = std::sqrt(f.maturity[lane]);
        const double d1 = (std::log(f.spot[lane] / f.strike[lane])
            + (f.rate[lane] - f.dividend[lane]
               + 0.5 * f.volatility[lane] * f.volatility[lane]) * f.maturity[lane])
            / (f.volatility[lane] * rootTime);
        return f.spot[lane] * std::exp(-f.dividend[lane] * f.maturity[lane])
            * normalDensity(d1) * rootTime;
    });
}

TEST(ForwardAD, BlackScholesRateSensitivityMatchesAnalytic)
{
    expectBlackScholesSensitivity(Seed::Rate, [](const auto& f, std::size_t lane)
    {
        const double rootTime = std::sqrt(f.maturity[lane]);
        const double d1 = (std::log(f.spot[lane] / f.strike[lane])
            + (f.rate[lane] - f.dividend[lane]
               + 0.5 * f.volatility[lane] * f.volatility[lane]) * f.maturity[lane])
            / (f.volatility[lane] * rootTime);
        const double d2 = d1 - f.volatility[lane] * rootTime;
        return f.strike[lane] * f.maturity[lane]
            * std::exp(-f.rate[lane] * f.maturity[lane]) * normalCdf(d2);
    });
}

TEST(ForwardAD, BlackScholesMaturitySensitivityMatchesAnalytic)
{
    expectBlackScholesSensitivity(Seed::Maturity, [](const auto& f, std::size_t lane)
    {
        const double rootTime = std::sqrt(f.maturity[lane]);
        const double d1 = (std::log(f.spot[lane] / f.strike[lane])
            + (f.rate[lane] - f.dividend[lane]
               + 0.5 * f.volatility[lane] * f.volatility[lane]) * f.maturity[lane])
            / (f.volatility[lane] * rootTime);
        const double d2 = d1 - f.volatility[lane] * rootTime;
        return f.spot[lane] * std::exp(-f.dividend[lane] * f.maturity[lane])
                * normalDensity(d1) * f.volatility[lane] / (2.0 * rootTime)
            + f.rate[lane] * f.strike[lane]
                * std::exp(-f.rate[lane] * f.maturity[lane]) * normalCdf(d2)
            - f.dividend[lane] * f.spot[lane]
                * std::exp(-f.dividend[lane] * f.maturity[lane]) * normalCdf(d1);
    });
}

TEST(ForwardAD, NonMultipleOfSimdWidth)
{
    const int length = 2 * static_cast<int>(VecXX::INS::size()) + 1;
    const auto result = exp(D(VecXX(values(length))));
    EXPECT_EQ(result.size(), static_cast<std::size_t>(length));
    EXPECT_NE(length % static_cast<int>(VecXX::INS::size()), 0);
    for (int lane = 0; lane < length; ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)],
                    std::exp(values(length)[static_cast<std::size_t>(lane)]), 1.0e-12);
}

TEST(ForwardAD, ScalarBroadcast)
{
    const auto result = Forward(asNumber(3.0), asNumber(0.0))
        * D(VecXX(values(5))) + asNumber(2.0);
    EXPECT_FALSE(result.isScalar());
    for (std::size_t lane = 0; lane < result.size(); ++lane)
        EXPECT_NEAR(result.derivative()[static_cast<std::size_t>(lane)], 3.0, 1.0e-12);
}

TEST(ForwardAD, RejectsMismatchedLogicalSizes)
{
    EXPECT_THROW((D(VecXX(values(3))) + D(VecXX(values(5)))), std::invalid_argument);
    EXPECT_THROW((Forward(VecXX(values(3)), VecXX(values(5)))), std::invalid_argument);
    EXPECT_THROW((Forward(VecXX(values(3)), VecXX(asNumber(0.0)))), std::invalid_argument);
}

TEST(ForwardAD, BlackScholesRejectsInvalidInputs)
{
    auto input = blackScholesFixture(3);
    input.strike[1] = 0.0;
    EXPECT_THROW(blackScholesAd(input, Seed::Spot), std::invalid_argument);
    input = blackScholesFixture(3);
    input.maturity[0] = 0.0;
    EXPECT_THROW(blackScholesAd(input, Seed::Spot), std::invalid_argument);
    input = blackScholesFixture(3);
    input.volatility[2] = -0.1;
    EXPECT_THROW(blackScholesAd(input, Seed::Spot), std::invalid_argument);
    input = blackScholesFixture(3);
    input.rate[0] = std::numeric_limits<Numeric>::infinity();
    EXPECT_THROW(blackScholesAd(input, Seed::Spot), std::invalid_argument);
}
