#include "dr3/advanced/pcr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace
{
using dr3::advanced::PcrSolver;
using dr3::advanced::PcrSystem;
using dr3::advanced::PcrWorkspace;

std::vector<double> thomas(const PcrSystem& system, const std::vector<double>& right)
{
    std::vector<double> diagonal = system.diagonal;
    std::vector<double> rhs = right;
    std::vector<double> output(system.size());
    for (std::size_t index = 1; index < system.size(); ++index)
    {
        const double multiplier = system.lower[index - 1] / diagonal[index - 1];
        diagonal[index] -= multiplier * system.upper[index - 1];
        rhs[index] -= multiplier * rhs[index - 1];
    }
    output.back() = rhs.back() / diagonal.back();
    for (std::size_t index = system.size() - 1; index-- > 0;)
        output[index] = (rhs[index] - system.upper[index] * output[index + 1]) / diagonal[index];
    return output;
}

PcrSystem dominantSystem(std::size_t size)
{
    return PcrSystem{std::vector<double>(size - 1, -1.0),
                     std::vector<double>(size, 4.0),
                     std::vector<double>(size - 1, -1.0)};
}

std::vector<double> deterministicRight(std::size_t size)
{
    std::vector<double> result(size);
    for (std::size_t index = 0; index < size; ++index)
        result[index] = std::sin(0.1 * static_cast<double>(index + 1));
    return result;
}

void expectMatchesThomas(std::size_t size, bool simd = true)
{
    const PcrSystem system = dominantSystem(size);
    const std::vector<double> right = deterministicRight(size);
    const std::vector<double> reference = thomas(system, right);
    std::vector<double> output(size);
    PcrWorkspace workspace(size);
    PcrSolver::solve(system, right, output, workspace, simd);
    for (std::size_t index = 0; index < size; ++index)
        EXPECT_NEAR(output[index], reference[index], 2.0e-12) << "size=" << size;
}

TEST(Pcr, SolvesOneByOneSystem)
{
    PcrSystem system{{}, {2.0}, {}};
    PcrWorkspace workspace(1);
    std::vector<double> output(1);
    PcrSolver::solve(system, std::vector<double>{6.0}, output, workspace);
    EXPECT_DOUBLE_EQ(output[0], 3.0);
}

TEST(Pcr, SolvesTwoByTwoSystem) { expectMatchesThomas(2); }

TEST(Pcr, SolvesKnownThreeByThreeSystem)
{
    PcrSystem system{{-1.0, -1.0}, {4.0, 4.0, 4.0}, {-1.0, -1.0}};
    PcrWorkspace workspace(3);
    std::vector<double> output(3);
    PcrSolver::solve(system, std::vector<double>{2.0, 4.0, 10.0}, output, workspace);
    EXPECT_NEAR(output[0], 1.0, 1e-13);
    EXPECT_NEAR(output[1], 2.0, 1e-13);
    EXPECT_NEAR(output[2], 3.0, 1e-13);
}

TEST(Pcr, MatchesThomasForDiagonallyDominantSystems)
{
    for (std::size_t size : {3U, 9U, 33U, 257U})
        expectMatchesThomas(size);
}

TEST(Pcr, RandomResidualIsSmall)
{
    constexpr std::size_t size = 65;
    std::mt19937 generator(0xD3A11U);
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    PcrSystem system = dominantSystem(size);
    std::vector<double> right(size);
    for (double& value : right) value = distribution(generator);
    PcrWorkspace workspace(size);
    std::vector<double> output(size);
    PcrSolver::solve(system, right, output, workspace);
    EXPECT_LT(PcrSolver::infinityResidual(system, right, output), 2.0e-12);
}

TEST(Pcr, PowerOfTwoSizes)
{
    for (std::size_t size : {2U, 8U, 32U, 64U, 256U, 512U}) expectMatchesThomas(size);
}

TEST(Pcr, NonPowerOfTwoSizes)
{
    for (std::size_t size : {3U, 7U, 9U, 31U, 33U, 63U, 65U, 255U, 257U, 511U, 513U})
        expectMatchesThomas(size);
}

TEST(Pcr, SizesAroundPowersOfTwo)
{
    for (std::size_t size : {1U, 2U, 3U, 7U, 8U, 9U, 31U, 32U, 33U,
                             63U, 64U, 65U, 255U, 256U, 257U, 511U, 512U, 513U})
        expectMatchesThomas(size);
}

TEST(Pcr, RejectsDimensionMismatch)
{
    PcrSystem system = dominantSystem(3);
    PcrWorkspace workspace(3);
    std::vector<double> output(3);
    EXPECT_THROW(PcrSolver::solve(system, std::vector<double>{1.0, 2.0}, output, workspace),
                 std::invalid_argument);
}

TEST(Pcr, RejectsNearZeroDenominator)
{
    PcrSystem system{{-1.0}, {0.0, 2.0}, {-1.0}};
    PcrWorkspace workspace(2);
    std::vector<double> output(2);
    EXPECT_THROW(PcrSolver::solve(system, std::vector<double>{1.0, 1.0}, output, workspace),
                 std::domain_error);
}

TEST(Pcr, RejectsUnsupportedAliasing)
{
    PcrSystem system = dominantSystem(3);
    PcrWorkspace workspace(3);
    std::vector<double> right{1.0, 2.0, 3.0};
    EXPECT_THROW(PcrSolver::solve(system, right.data(), right.size(), right.data(), right.size(),
                                  workspace), std::invalid_argument);
}

TEST(Pcr, PreservesInput)
{
    PcrSystem system = dominantSystem(9);
    std::vector<double> right = deterministicRight(9);
    const PcrSystem original = system;
    const auto originalRight = right;
    PcrWorkspace workspace(9);
    std::vector<double> output(9);
    PcrSolver::solve(system, right, output, workspace);
    EXPECT_EQ(system.lower, original.lower);
    EXPECT_EQ(system.diagonal, original.diagonal);
    EXPECT_EQ(system.upper, original.upper);
    EXPECT_EQ(right, originalRight);
}

TEST(Pcr, RepeatedSolveIsDeterministic)
{
    const PcrSystem system = dominantSystem(65);
    const auto right = deterministicRight(65);
    PcrWorkspace workspace(65);
    std::vector<double> first(65), second(65);
    PcrSolver::solve(system, right, first, workspace);
    PcrSolver::solve(system, right, second, workspace);
    EXPECT_EQ(first, second);
}

TEST(Pcr, SimdStagesMatchScalarStages)
{
    const PcrSystem system = dominantSystem(257);
    const auto right = deterministicRight(257);
    PcrWorkspace scalarWorkspace(257), simdWorkspace(257);
    std::vector<double> scalar(257), simd(257);
    PcrSolver::solve(system, right, scalar, scalarWorkspace, false);
    PcrSolver::solve(system, right, simd, simdWorkspace, true);
    EXPECT_EQ(scalar, simd);
}

TEST(Pcr, FinalResidualMatchesThomasResidual)
{
    const PcrSystem system = dominantSystem(513);
    const auto right = deterministicRight(513);
    const auto reference = thomas(system, right);
    PcrWorkspace workspace(513);
    std::vector<double> output(513);
    PcrSolver::solve(system, right, output, workspace);
    EXPECT_NEAR(PcrSolver::infinityResidual(system, right, output),
                PcrSolver::infinityResidual(system, right, reference), 2.0e-12);
}

} // namespace
