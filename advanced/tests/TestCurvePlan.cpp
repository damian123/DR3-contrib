#include "dr3/advanced/curve_plan.h"
#include "dr3/advanced/reverse_aad.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

namespace
{
using dr3::advanced::CurveEvaluationPlan;
using dr3::advanced::CurveExtrapolation;

struct Dual
{
    double value{};
    double derivative{};
};

Dual operator*(Dual value, double scalar)
{
    return {value.value * scalar, value.derivative * scalar};
}

Dual operator+(Dual left, Dual right)
{
    return {left.value + right.value, left.derivative + right.derivative};
}

const std::vector<double> pillars{0.0, 1.0, 2.0, 4.0};
const std::vector<double> values{10.0, 20.0, 30.0, 50.0};

TEST(CurvePlan, DirectAndPlannedEvaluationAgree)
{
    const std::vector<double> queries{-1.0, 0.0, 0.4, 1.0, 3.0, 4.0, 5.0};
    CurveEvaluationPlan plan(pillars, queries);
    std::vector<double> output(queries.size());
    plan.evaluate(pillars, values, output);
    for (std::size_t index = 0; index < queries.size(); ++index)
    {
        EXPECT_DOUBLE_EQ(output[index],
            dr3::advanced::directCurveValue(pillars, values, queries[index]));
    }
}

TEST(CurvePlan, ExactFirstPillar)
{
    CurveEvaluationPlan plan(pillars, {0.0});
    EXPECT_DOUBLE_EQ(plan.evaluateOne(pillars, values, 0), 10.0);
    EXPECT_EQ(plan.entry(0).left, plan.entry(0).right);
}

TEST(CurvePlan, ExactInteriorPillar)
{
    CurveEvaluationPlan plan(pillars, {2.0});
    EXPECT_DOUBLE_EQ(plan.evaluateOne(pillars, values, 0), 30.0);
    EXPECT_EQ(plan.entry(0).left, plan.entry(0).right);
}

TEST(CurvePlan, ExactFinalPillar)
{
    CurveEvaluationPlan plan(pillars, {4.0});
    EXPECT_DOUBLE_EQ(plan.evaluateOne(pillars, values, 0), 50.0);
    EXPECT_EQ(plan.entry(0).left, plan.entry(0).right);
}

TEST(CurvePlan, LeftExtrapolation)
{
    CurveEvaluationPlan flat(pillars, {-1.0});
    CurveEvaluationPlan linear(pillars, {-1.0}, CurveExtrapolation::Linear);
    EXPECT_DOUBLE_EQ(flat.evaluateOne(pillars, values, 0), 10.0);
    EXPECT_DOUBLE_EQ(linear.evaluateOne(pillars, values, 0), 0.0);
}

TEST(CurvePlan, RightExtrapolation)
{
    CurveEvaluationPlan flat(pillars, {5.0});
    CurveEvaluationPlan linear(pillars, {5.0}, CurveExtrapolation::Linear);
    EXPECT_DOUBLE_EQ(flat.evaluateOne(pillars, values, 0), 50.0);
    EXPECT_DOUBLE_EQ(linear.evaluateOne(pillars, values, 0), 60.0);
}

TEST(CurvePlan, ReusesPlanWithDifferentCurveValues)
{
    CurveEvaluationPlan plan(pillars, {1.5});
    EXPECT_DOUBLE_EQ(plan.evaluateOne(pillars, values, 0), 25.0);
    EXPECT_DOUBLE_EQ(plan.evaluateOne(pillars, std::vector<double>{1, 2, 3, 5}, 0), 2.5);
}

TEST(CurvePlan, RejectsDifferentPillarGrid)
{
    CurveEvaluationPlan plan(pillars, {1.5});
    EXPECT_THROW(plan.evaluateOne(std::vector<double>{0, 1, 2, 5}, values, 0),
                 std::invalid_argument);
}

TEST(CurvePlan, DoesNotRetainCallerStorage)
{
    std::vector<double> callerPillars = pillars;
    std::vector<double> callerQueries{1.5};
    CurveEvaluationPlan plan(callerPillars, callerQueries);
    callerPillars.assign(4, 99.0);
    callerQueries[0] = 99.0;
    EXPECT_DOUBLE_EQ(plan.evaluateOne(pillars, values, 0), 25.0);
}

TEST(CurvePlan, ScalarValues)
{
    CurveEvaluationPlan plan(pillars, {3.0});
    EXPECT_DOUBLE_EQ(plan.evaluateOne(pillars, values, 0), 40.0);
}

TEST(CurvePlan, VectorValues)
{
    CurveEvaluationPlan plan(pillars, {1.5});
    const std::vector<Vec4d> vectorValues{Vec4d(0.0), Vec4d(2.0), Vec4d(4.0), Vec4d(8.0)};
    const Vec4d result = plan.evaluateOne(pillars, vectorValues, 0);
    for (std::size_t lane = 0; lane < 4; ++lane)
    {
        EXPECT_DOUBLE_EQ(dr3::advanced::aad::ReverseTraits<Vec4d>::lane(result, lane), 3.0);
    }
}

TEST(CurvePlan, ForwardADValues)
{
    CurveEvaluationPlan plan(pillars, {1.5});
    const std::vector<Dual> dualValues{{10, 0}, {20, 1}, {30, 0}, {50, 0}};
    const Dual result = plan.evaluateOne(pillars, dualValues, 0);
    EXPECT_DOUBLE_EQ(result.value, 25.0);
    EXPECT_DOUBLE_EQ(result.derivative, 0.5);
}

TEST(CurvePlan, ReverseAADValues)
{
    dr3::advanced::aad::ScalarTape tape;
    std::vector<dr3::advanced::aad::ScalarTape::Active> activeValues;
    for (double value : values)
    {
        activeValues.push_back(tape.variable(value));
    }
    CurveEvaluationPlan plan(pillars, {1.5});
    const auto result = plan.evaluateOne(pillars, activeValues, 0);
    tape.reverse(result);
    EXPECT_DOUBLE_EQ(tape.adjoint(activeValues[1]), 0.5);
    EXPECT_DOUBLE_EQ(tape.adjoint(activeValues[2]), 0.5);
}

TEST(CurvePlan, ConcurrentReadEvaluation)
{
    CurveEvaluationPlan plan(pillars, {0.5, 1.5, 3.0});
    std::atomic<bool> valid{true};
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 8; ++worker)
    {
        threads.emplace_back([&]
        {
            std::vector<double> output(3);
            for (int repeat = 0; repeat < 100; ++repeat)
            {
                plan.evaluate(pillars, values, output);
                valid = valid && output == std::vector<double>({15.0, 25.0, 40.0});
            }
        });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    EXPECT_TRUE(valid.load());
}

TEST(CurvePlan, ConcurrentResultsAreDeterministic)
{
    CurveEvaluationPlan plan(pillars, {1.25});
    std::vector<double> results(16);
    std::vector<std::thread> threads;
    for (std::size_t worker = 0; worker < results.size(); ++worker)
    {
        threads.emplace_back([&, worker]
        {
            results[worker] = plan.evaluateOne(pillars, values, 0);
        });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    for (double result : results)
    {
        EXPECT_DOUBLE_EQ(result, 22.5);
    }
}

TEST(ConcurrencySafety, ImmutableCurveReadsMatchSerial)
{
    CurveEvaluationPlan plan(pillars, {0.25, 0.75, 1.25, 2.5, 3.5});
    std::vector<double> serial(5);
    plan.evaluate(pillars, values, serial);
    std::vector<std::vector<double>> parallel(4, std::vector<double>(5));
    std::vector<std::thread> threads;
    for (auto& output : parallel)
    {
        threads.emplace_back([&plan, &output] { plan.evaluate(pillars, values, output); });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    for (const auto& output : parallel)
    {
        EXPECT_EQ(output, serial);
    }
}

} // namespace
