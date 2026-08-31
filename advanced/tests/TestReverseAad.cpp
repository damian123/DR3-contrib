#include "dr3/advanced/curve_plan.h"
#include "dr3/advanced/heston.h"
#include "dr3/advanced/reverse_aad.h"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{
using dr3::advanced::aad::ReverseTraits;
using dr3::advanced::aad::ScalarTape;
using dr3::advanced::aad::SimdTape;

double lane(Vec4d value, std::size_t index)
{
    return ReverseTraits<Vec4d>::lane(value, index);
}

double central(const std::function<double(double)>& function, double value)
{
    constexpr double bump = 1.0e-8;
    return (function(value + bump) - function(value - bump)) / (2.0 * bump);
}

template <class Active>
Active callPrice(const Active& spot, double strike, const Active& rate,
                 const Active& volatility, const Active& maturity)
{
    const Active d1 = (log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity)
        / (volatility * sqrt(maturity));
    const Active d2 = d1 - volatility * sqrt(maturity);
    return spot * normalCdf(d1) - strike * exp(-rate * maturity) * normalCdf(d2);
}

struct ForwardDual
{
    double value{};
    double derivative{};
    ForwardDual(double primal = 0.0, double tangent = 0.0)
        : value(primal), derivative(tangent) {}
};

ForwardDual operator+(ForwardDual left, ForwardDual right)
{
    return {left.value + right.value, left.derivative + right.derivative};
}
ForwardDual operator-(ForwardDual left, ForwardDual right)
{
    return {left.value - right.value, left.derivative - right.derivative};
}
ForwardDual operator-(ForwardDual value) { return {-value.value, -value.derivative}; }
ForwardDual operator*(ForwardDual left, ForwardDual right)
{
    return {left.value * right.value,
            left.derivative * right.value + left.value * right.derivative};
}
ForwardDual operator/(ForwardDual left, ForwardDual right)
{
    return {left.value / right.value,
            (left.derivative * right.value - left.value * right.derivative)
                / (right.value * right.value)};
}
ForwardDual log(ForwardDual value) { return {std::log(value.value), value.derivative / value.value}; }
ForwardDual sqrt(ForwardDual value)
{
    const double primal = std::sqrt(value.value);
    return {primal, 0.5 * value.derivative / primal};
}
ForwardDual exp(ForwardDual value)
{
    const double primal = std::exp(value.value);
    return {primal, primal * value.derivative};
}
ForwardDual normalCdf(ForwardDual value)
{
    return {ReverseTraits<double>::normalCdf(value.value),
            ReverseTraits<double>::normalPdf(value.value) * value.derivative};
}

TEST(ReverseAAD, VariableSeedIsOne)
{
    ScalarTape tape;
    const auto variable = tape.variable(2.0);
    tape.reverse(variable);
    EXPECT_DOUBLE_EQ(tape.adjoint(variable), 1.0);
}

TEST(ReverseAAD, ConfigurableOutputSeed)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    const auto y = x * x;
    tape.reverse(y, 3.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), 12.0);
}

TEST(ReverseAAD, ConstantDoesNotCreateNode)
{
    ScalarTape tape;
    const auto before = tape.size();
    const auto constant = tape.constant(4.0);
    EXPECT_TRUE(constant.isConstant());
    EXPECT_EQ(tape.size(), before);
}

TEST(ReverseAAD, AdditionDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    const auto y = tape.variable(3.0);
    const auto result = x + y;
    tape.reverse(result);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y), 1.0);
}

TEST(ReverseAAD, SubtractionDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    const auto y = tape.variable(3.0);
    tape.reverse(x - y);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y), -1.0);
}

TEST(ReverseAAD, ProductRule)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    const auto y = tape.variable(3.0);
    tape.reverse(x * y);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), 3.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y), 2.0);
}

TEST(ReverseAAD, QuotientRule)
{
    ScalarTape tape;
    const auto x = tape.variable(6.0);
    const auto y = tape.variable(3.0);
    tape.reverse(x / y);
    EXPECT_NEAR(tape.adjoint(x), 1.0 / 3.0, 1.0e-12);
    EXPECT_NEAR(tape.adjoint(y), -2.0 / 3.0, 1.0e-12);
}

TEST(ReverseAAD, UnaryMinus)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    tape.reverse(-x);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), -1.0);
}

TEST(ReverseAAD, ExpDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(0.7);
    tape.reverse(exp(x));
    EXPECT_NEAR(tape.adjoint(x), std::exp(0.7), 1.0e-12);
}

TEST(ReverseAAD, LogDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(1.7);
    tape.reverse(log(x));
    EXPECT_NEAR(tape.adjoint(x), 1.0 / 1.7, 1.0e-12);
}

TEST(ReverseAAD, SqrtDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(1.7);
    tape.reverse(sqrt(x));
    EXPECT_NEAR(tape.adjoint(x), 0.5 / std::sqrt(1.7), 1.0e-12);
}

TEST(ReverseAAD, SinDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(0.7);
    tape.reverse(sin(x));
    EXPECT_NEAR(tape.adjoint(x), std::cos(0.7), 1.0e-12);
}

TEST(ReverseAAD, CosDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(0.7);
    tape.reverse(cos(x));
    EXPECT_NEAR(tape.adjoint(x), -std::sin(0.7), 1.0e-12);
}

TEST(ReverseAAD, PowerDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(1.7);
    tape.reverse(pow(x, 2.5));
    EXPECT_NEAR(tape.adjoint(x), 2.5 * std::pow(1.7, 1.5), 1.0e-12);
}

TEST(ReverseAAD, NormalCdfDerivative)
{
    ScalarTape tape;
    const auto x = tape.variable(0.7);
    tape.reverse(normalCdf(x));
    EXPECT_NEAR(tape.adjoint(x), ReverseTraits<double>::normalPdf(0.7), 1.0e-12);
}

TEST(ReverseAAD, CompositeExpressionMatchesCentralDifference)
{
    const auto function = [](double x)
    {
        return std::exp(std::sin(x)) * std::log(x + 2.0) / std::sqrt(x + 3.0);
    };
    ScalarTape tape;
    const auto x = tape.variable(0.4);
    const auto y = exp(sin(x)) * log(x + 2.0) / sqrt(x + 3.0);
    tape.reverse(y);
    EXPECT_NEAR(tape.adjoint(x), central(function, 0.4), 1.0e-8);
}

TEST(ReverseAAD, OneOutputSixtyFourInputs)
{
    ScalarTape tape;
    tape.reserve(130);
    std::vector<ScalarTape::Active> inputs;
    inputs.reserve(64);
    auto output = tape.constant(0.0);
    for (std::size_t index = 0; index < 64; ++index)
    {
        inputs.push_back(tape.variable(static_cast<double>(index + 1)));
        output = output + inputs.back() * static_cast<double>(index + 2);
    }
    tape.reverse(output);
    for (std::size_t index = 0; index < inputs.size(); ++index)
    {
        EXPECT_DOUBLE_EQ(tape.adjoint(inputs[index]), static_cast<double>(index + 2));
    }
}

TEST(ReverseAAD, RepeatedReverseIsDeterministic)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    const auto y = x * x * x;
    tape.reverse(y);
    const double first = tape.adjoint(x);
    tape.reverse(y);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), first);
}

TEST(ReverseAAD, ZeroAdjointsAllowsSecondSweep)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    const auto y = x * x;
    tape.reverse(y);
    tape.zeroAdjoints();
    tape.reverse(y, 2.0, dr3::advanced::aad::SweepMode::AccumulateExistingAdjoints);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), 8.0);
}

TEST(ReverseAAD, CrossTapeOperandsAreRejected)
{
    ScalarTape first;
    ScalarTape second;
    const auto x = first.variable(1.0);
    const auto y = second.variable(2.0);
    EXPECT_THROW((void)(x + y), std::invalid_argument);
}

TEST(ReverseAAD, StaleHandleAfterRewindIsRejected)
{
    ScalarTape tape;
    (void)tape.variable(1.0);
    const auto mark = tape.mark();
    const auto stale = tape.variable(2.0);
    tape.rewind(mark);
    EXPECT_THROW((void)(stale + 1.0), std::logic_error);
}

TEST(ReverseAAD, StaleHandleAfterClearIsRejected)
{
    ScalarTape tape;
    const auto stale = tape.variable(2.0);
    tape.clear();
    EXPECT_THROW((void)(stale + 1.0), std::logic_error);
}

TEST(ReverseAAD, CapacityGrowthDoesNotInvalidateNodes)
{
    ScalarTape tape;
    const auto x = tape.variable(2.0);
    auto output = x;
    for (int index = 0; index < 1000; ++index)
    {
        output = output + 1.0;
    }
    tape.reverse(output);
    EXPECT_DOUBLE_EQ(tape.adjoint(x), 1.0);
}

TEST(ReverseAAD, NoAllocationDuringReservedReverseSweep)
{
    ScalarTape tape;
    tape.reserve(16);
    const auto x = tape.variable(2.0);
    const auto y = exp(x * x);
    const auto capacity = tape.capacity();
    tape.reverse(y);
    EXPECT_EQ(tape.capacity(), capacity);
}

TEST(SimdReverseAAD, AdditionMatchesScalarByLane)
{
    SimdTape tape;
    const auto x = tape.variable(Vec4d(1.0, 2.0, 3.0, 4.0));
    const auto y = tape.variable(Vec4d(4.0, 3.0, 2.0, 1.0));
    const auto result = x + y;
    tape.reverse(result);
    for (std::size_t index = 0; index < 4; ++index)
    {
        EXPECT_DOUBLE_EQ(lane(result.primal(), index), 5.0);
        EXPECT_DOUBLE_EQ(lane(tape.adjoint(x), index), 1.0);
    }
}

TEST(SimdReverseAAD, CompositeExpressionMatchesScalarByLane)
{
    SimdTape tape;
    const Vec4d values(0.2, 0.4, 0.6, 0.8);
    const auto x = tape.variable(values);
    const auto y = exp(sin(x)) / sqrt(x + 2.0);
    tape.reverse(y);
    for (std::size_t index = 0; index < 4; ++index)
    {
        const double value = lane(values, index);
        const auto function = [](double z) { return std::exp(std::sin(z)) / std::sqrt(z + 2.0); };
        EXPECT_NEAR(lane(tape.adjoint(x), index), central(function, value), 3.0e-8);
    }
}

TEST(SimdReverseAAD, NoCrossLaneContamination)
{
    SimdTape tape;
    const auto x = tape.variable(Vec4d(1.0, 2.0, 3.0, 4.0));
    const auto y = x * x;
    tape.reverse(y, Vec4d(1.0, 0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(lane(tape.adjoint(x), 0), 2.0);
    for (std::size_t index = 1; index < 4; ++index)
    {
        EXPECT_DOUBLE_EQ(lane(tape.adjoint(x), index), 0.0);
    }
}

TEST(SimdReverseAAD, NonMultipleOfSimdWidth)
{
    SimdTape tape;
    const auto x = tape.variable(Vec4d(1.0, 2.0, 3.0, 999.0));
    const auto y = x * x;
    tape.reverse(y, Vec4d(1.0), 3);
    EXPECT_DOUBLE_EQ(lane(tape.adjoint(x), 2), 6.0);
    EXPECT_DOUBLE_EQ(lane(tape.adjoint(x), 3), 0.0);
}

TEST(SimdReverseAAD, BlackScholesPriceMatchesScalar)
{
    SimdTape tape;
    const auto spot = tape.variable(Vec4d(80.0, 100.0, 120.0, 140.0));
    const auto rate = tape.variable(Vec4d(0.03));
    const auto volatility = tape.variable(Vec4d(0.2));
    const auto maturity = tape.variable(Vec4d(1.25));
    const auto price = callPrice(spot, 100.0, rate, volatility, maturity);
    for (std::size_t index = 0; index < 4; ++index)
    {
        EXPECT_NEAR(lane(price.primal(), index),
            dr3::advanced::blackScholesPrice(dr3::advanced::EuropeanOption::Call,
                lane(spot.primal(), index), 100.0, 1.25, 0.03, 0.0, 0.2), 1.0e-12);
        const ForwardDual forwardPrice = callPrice(ForwardDual(lane(spot.primal(), index), 1.0),
            100.0, ForwardDual(0.03), ForwardDual(0.2), ForwardDual(1.25));
        EXPECT_NEAR(lane(price.primal(), index), forwardPrice.value, 1.0e-12);
    }
}

TEST(SimdReverseAAD, BlackScholesDeltaMatchesAnalytic)
{
    SimdTape tape;
    const auto spot = tape.variable(Vec4d(80.0, 100.0, 120.0, 140.0));
    const auto rate = tape.variable(Vec4d(0.03));
    const auto volatility = tape.variable(Vec4d(0.2));
    const auto maturity = tape.variable(Vec4d(1.25));
    const auto price = callPrice(spot, 100.0, rate, volatility, maturity);
    tape.reverse(price);
    for (std::size_t index = 0; index < 4; ++index)
    {
        const double s = lane(spot.primal(), index);
        const double d1 = (std::log(s / 100.0) + (0.03 + 0.5 * 0.04) * 1.25)
            / (0.2 * std::sqrt(1.25));
        EXPECT_NEAR(lane(tape.adjoint(spot), index), ReverseTraits<double>::normalCdf(d1), 1.0e-12);
    }
}

TEST(SimdReverseAAD, BlackScholesVegaMatchesAnalytic)
{
    SimdTape tape;
    const auto spot = tape.variable(Vec4d(80.0, 100.0, 120.0, 140.0));
    const auto rate = tape.variable(Vec4d(0.03));
    const auto volatility = tape.variable(Vec4d(0.2));
    const auto maturity = tape.variable(Vec4d(1.25));
    const auto price = callPrice(spot, 100.0, rate, volatility, maturity);
    tape.reverse(price);
    for (std::size_t index = 0; index < 4; ++index)
    {
        const double s = lane(spot.primal(), index);
        const double d1 = (std::log(s / 100.0) + (0.03 + 0.5 * 0.04) * 1.25)
            / (0.2 * std::sqrt(1.25));
        EXPECT_NEAR(lane(tape.adjoint(volatility), index),
                    s * ReverseTraits<double>::normalPdf(d1) * std::sqrt(1.25), 1.0e-10);
    }
}

TEST(SimdReverseAAD, BlackScholesRhoMatchesAnalytic)
{
    SimdTape tape;
    const auto spot = tape.variable(Vec4d(80.0, 100.0, 120.0, 140.0));
    const auto rate = tape.variable(Vec4d(0.03));
    const auto volatility = tape.variable(Vec4d(0.2));
    const auto maturity = tape.variable(Vec4d(1.25));
    const auto price = callPrice(spot, 100.0, rate, volatility, maturity);
    tape.reverse(price);
    for (std::size_t index = 0; index < 4; ++index)
    {
        const double s = lane(spot.primal(), index);
        const double d1 = (std::log(s / 100.0) + (0.03 + 0.5 * 0.04) * 1.25)
            / (0.2 * std::sqrt(1.25));
        const double d2 = d1 - 0.2 * std::sqrt(1.25);
        EXPECT_NEAR(lane(tape.adjoint(rate), index),
            100.0 * 1.25 * std::exp(-0.03 * 1.25)
                * ReverseTraits<double>::normalCdf(d2), 1.0e-10);
    }
}

TEST(SimdReverseAAD, BlackScholesMaturitySensitivityMatchesAnalytic)
{
    SimdTape tape;
    const auto spot = tape.variable(Vec4d(80.0, 100.0, 120.0, 140.0));
    const auto rate = tape.variable(Vec4d(0.03));
    const auto volatility = tape.variable(Vec4d(0.2));
    const auto maturity = tape.variable(Vec4d(1.25));
    const auto price = callPrice(spot, 100.0, rate, volatility, maturity);
    tape.reverse(price);
    for (std::size_t index = 0; index < 4; ++index)
    {
        const double s = lane(spot.primal(), index);
        const double d1 = (std::log(s / 100.0) + (0.03 + 0.5 * 0.04) * 1.25)
            / (0.2 * std::sqrt(1.25));
        const double d2 = d1 - 0.2 * std::sqrt(1.25);
        const double expected = s * ReverseTraits<double>::normalPdf(d1) * 0.2
                / (2.0 * std::sqrt(1.25))
            + 0.03 * 100.0 * std::exp(-0.03 * 1.25)
                * ReverseTraits<double>::normalCdf(d2);
        EXPECT_NEAR(lane(tape.adjoint(maturity), index), expected, 1.0e-10);
    }
}

struct CurvePortfolio
{
    ScalarTape tape;
    std::vector<double> pillars;
    std::vector<ScalarTape::Active> rates;
    ScalarTape::Active value;

    CurvePortfolio() : value(tape.constant(0.0))
    {
        tape.reserve(1200);
        pillars.resize(257);
        rates.reserve(257);
        for (std::size_t index = 0; index < pillars.size(); ++index)
        {
            pillars[index] = 0.125 * static_cast<double>(index);
            rates.push_back(tape.variable(0.01 + 0.00002 * static_cast<double>(index)));
        }
        std::vector<double> queries;
        for (std::size_t index = 0; index < 64; ++index)
        {
            queries.push_back(4.0625 + 0.25 * static_cast<double>(index));
        }
        dr3::advanced::CurveEvaluationPlan plan(pillars, queries);
        std::vector<ScalarTape::Active> interpolated(queries.size());
        plan.evaluate(pillars, rates, interpolated);
        for (std::size_t index = 0; index < queries.size(); ++index)
        {
            value = value + (100.0 + static_cast<double>(index))
                * exp(-interpolated[index] * queries[index]);
        }
        tape.reverse(value);
    }
};

TEST(SimdReverseAAD, CurvePortfolioPrimalMatchesForwardAD)
{
    CurvePortfolio portfolio;
    std::vector<double> scalarRates(257);
    for (std::size_t index = 0; index < scalarRates.size(); ++index)
        scalarRates[index] = 0.01 + 0.00002 * static_cast<double>(index);
    std::vector<double> queries;
    for (std::size_t index = 0; index < 64; ++index)
        queries.push_back(4.0625 + 0.25 * static_cast<double>(index));
    dr3::advanced::CurveEvaluationPlan plan(portfolio.pillars, queries);
    std::vector<double> interpolated(queries.size());
    plan.evaluate(portfolio.pillars, scalarRates, interpolated);
    double forwardPrimal = 0.0;
    for (std::size_t index = 0; index < queries.size(); ++index)
        forwardPrimal += (100.0 + static_cast<double>(index))
            * std::exp(-interpolated[index] * queries[index]);
    EXPECT_NEAR(portfolio.value.primal(), forwardPrimal, 1.0e-12);
}

TEST(SimdReverseAAD, CurvePillarAdjointsMatchForwardAD)
{
    CurvePortfolio portfolio;
    const std::size_t selected = 40;
    std::vector<ForwardDual> forwardRates;
    for (std::size_t index = 0; index < 257; ++index)
        forwardRates.emplace_back(0.01 + 0.00002 * static_cast<double>(index),
                                  index == selected ? 1.0 : 0.0);
    std::vector<double> queries;
    for (std::size_t index = 0; index < 64; ++index)
        queries.push_back(4.0625 + 0.25 * static_cast<double>(index));
    dr3::advanced::CurveEvaluationPlan plan(portfolio.pillars, queries);
    std::vector<ForwardDual> interpolated(queries.size());
    plan.evaluate(portfolio.pillars, forwardRates, interpolated);
    ForwardDual forwardValue;
    for (std::size_t index = 0; index < queries.size(); ++index)
        forwardValue = forwardValue + (100.0 + static_cast<double>(index))
            * exp(-interpolated[index] * queries[index]);
    EXPECT_NEAR(portfolio.tape.adjoint(portfolio.rates[selected]),
                forwardValue.derivative, 1.0e-12);
    EXPECT_NEAR(portfolio.tape.adjoint(portfolio.rates[selected]),
                portfolio.tape.adjoint(portfolio.rates[selected + 1]), 1.0e-12);
}

TEST(SimdReverseAAD, SelectedPillarsMatchCentralDifference)
{
    CurvePortfolio portfolio;
    const std::size_t selected = 40;
    const double original = 0.01 + 0.00002 * static_cast<double>(selected);
    auto primal = [selected, &portfolio](double selectedRate)
    {
        std::vector<double> rates(257);
        for (std::size_t index = 0; index < rates.size(); ++index)
        {
            rates[index] = 0.01 + 0.00002 * static_cast<double>(index);
        }
        rates[selected] = selectedRate;
        std::vector<double> queries;
        for (std::size_t index = 0; index < 64; ++index)
        {
            queries.push_back(4.0625 + 0.25 * static_cast<double>(index));
        }
        dr3::advanced::CurveEvaluationPlan plan(portfolio.pillars, queries);
        std::vector<double> interpolated(queries.size());
        plan.evaluate(portfolio.pillars, rates, interpolated);
        double value = 0.0;
        for (std::size_t index = 0; index < queries.size(); ++index)
        {
            value += (100.0 + static_cast<double>(index))
                * std::exp(-interpolated[index] * queries[index]);
        }
        return value;
    };
    EXPECT_NEAR(portfolio.tape.adjoint(portfolio.rates[selected]),
                central(primal, original), 2.0e-3);
}

TEST(SimdReverseAAD, UnrelatedPillarsHaveZeroAdjoint)
{
    CurvePortfolio portfolio;
    EXPECT_DOUBLE_EQ(portfolio.tape.adjoint(portfolio.rates.front()), 0.0);
    EXPECT_DOUBLE_EQ(portfolio.tape.adjoint(portfolio.rates.back()), 0.0);
}

TEST(SimdReverseAAD, NoAllocationDuringReservedReverseSweep)
{
    SimdTape tape;
    tape.reserve(16);
    const auto x = tape.variable(Vec4d(1.0, 2.0, 3.0, 4.0));
    const auto y = exp(x * x);
    const std::size_t capacity = tape.capacity();
    tape.reverse(y);
    EXPECT_EQ(tape.capacity(), capacity);
}

} // namespace
