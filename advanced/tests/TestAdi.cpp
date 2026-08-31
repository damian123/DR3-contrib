#include "dr3/advanced/adi.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
using dr3::advanced::AdiScheme;
using dr3::advanced::AdiSolver;
using dr3::advanced::AdiWorkspace;
using dr3::advanced::Axis1D;
using dr3::advanced::Grid2D;
using dr3::advanced::SplitOperator2D;
using dr3::advanced::Surface2D;

constexpr double pi = 3.14159265358979323846;

Grid2D adiGrid(std::size_t count = 9)
{
    return Grid2D(Axis1D::uniform(0.0, 1.0, count),
                  Axis1D::uniform(0.0, 1.0, count));
}

SplitOperator2D heatOperator(const Grid2D& grid, double firstDiffusion = 0.1,
                             double secondDiffusion = 0.15)
{
    SplitOperator2D result(grid);
    const double firstH = grid.firstAxis()[1] - grid.firstAxis()[0];
    const double secondH = grid.secondAxis()[1] - grid.secondAxis()[0];
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            const std::size_t index = grid.index(first, second);
            result.firstLower(index) = firstDiffusion / (firstH * firstH);
            result.firstDiagonal(index) = -2.0 * firstDiffusion / (firstH * firstH);
            result.firstUpper(index) = firstDiffusion / (firstH * firstH);
            result.secondLower(index) = secondDiffusion / (secondH * secondH);
            result.secondDiagonal(index) = -2.0 * secondDiffusion / (secondH * secondH);
            result.secondUpper(index) = secondDiffusion / (secondH * secondH);
        }
    }
    return result;
}

Surface2D<double> eigenmode(const Grid2D& grid)
{
    Surface2D<double> result(grid, 0.0);
    for (std::size_t second = 0; second < grid.secondSize(); ++second)
        for (std::size_t first = 0; first < grid.firstSize(); ++first)
            result(first, second) = std::sin(pi * grid.firstAxis()[first])
                * std::sin(pi * grid.secondAxis()[second]);
    return result;
}

const dr3::advanced::AdiBoundary zeroBoundary =
    [](double, const Grid2D& grid, double* values)
    {
        for (std::size_t first = 0; first < grid.firstSize(); ++first)
        {
            values[grid.index(first, 0)] = 0.0;
            values[grid.index(first, grid.secondSize() - 1)] = 0.0;
        }
        for (std::size_t second = 0; second < grid.secondSize(); ++second)
        {
            values[grid.index(0, second)] = 0.0;
            values[grid.index(grid.firstSize() - 1, second)] = 0.0;
        }
    };

TEST(Adi, OperatorSplitEqualsFullOperator)
{
    const Grid2D grid = adiGrid(7);
    SplitOperator2D split = heatOperator(grid);
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
            split.explicitCoefficient(1, 1, grid.index(first, second)) = 0.07;
    const auto input = eigenmode(grid);
    std::vector<double> explicitPart(grid.size()), firstPart(grid.size()),
                        secondPart(grid.size()), full(grid.size());
    split.applyExplicit(input.data(), explicitPart.data(), grid.size());
    split.applyFirst(input.data(), firstPart.data(), grid.size());
    split.applySecond(input.data(), secondPart.data(), grid.size());
    split.applyFull(input.data(), full.data(), grid.size());
    for (std::size_t index = 0; index < grid.size(); ++index)
        EXPECT_NEAR(full[index], explicitPart[index] + firstPart[index] + secondPart[index], 1e-13);
}

TEST(Adi, ZeroMixedTerm)
{
    const Grid2D grid = adiGrid(5);
    const SplitOperator2D split = heatOperator(grid);
    const auto input = eigenmode(grid);
    std::vector<double> output(grid.size(), 1.0);
    split.applyExplicit(input.data(), output.data(), output.size());
    EXPECT_EQ(output, std::vector<double>(grid.size(), 0.0));
}

TEST(Adi, ZeroSecondDirectionReducesToOneDimensionalSolver)
{
    const Grid2D grid = adiGrid(7);
    SplitOperator2D split = heatOperator(grid, 0.1, 0.0);
    auto input = eigenmode(grid);
    Surface2D<double> output(grid);
    AdiWorkspace workspace;
    workspace.initialize(split, 0.001);
    AdiSolver::step(split, input, output, 0.0, zeroBoundary, workspace, AdiScheme::Douglas);
    for (std::size_t second = 2; second + 2 < grid.secondSize(); ++second)
        EXPECT_NEAR(output(3, second) / input(3, second),
                    output(3, second + 1) / input(3, second + 1), 1e-12);
}

double oneStepEigenError(AdiScheme scheme, double timeStep)
{
    const Grid2D grid = adiGrid(9);
    SplitOperator2D split = heatOperator(grid);
    auto input = eigenmode(grid);
    Surface2D<double> output(grid);
    AdiWorkspace workspace;
    workspace.initialize(split, timeStep);
    AdiSolver::step(split, input, output, 0.0, zeroBoundary, workspace, scheme);
    const double h = grid.firstAxis()[1] - grid.firstAxis()[0];
    const double eigenvalue = (0.1 + 0.15) * 2.0 * (std::cos(pi * h) - 1.0) / (h * h);
    const double expected = input(4, 4) * std::exp(eigenvalue * timeStep);
    return std::abs(output(4, 4) - expected);
}

double mcsEvolutionError(std::size_t stepCount)
{
    constexpr double finalTime = 0.04;
    const Grid2D grid = adiGrid(9);
    SplitOperator2D split = heatOperator(grid);
    auto current = eigenmode(grid);
    Surface2D<double> next(grid);
    const double timeStep = finalTime / static_cast<double>(stepCount);
    AdiWorkspace workspace;
    workspace.initialize(split, timeStep);
    for (std::size_t step = 0; step < stepCount; ++step)
    {
        AdiSolver::step(split, current, next, step * timeStep, zeroBoundary,
                        workspace, AdiScheme::ModifiedCraigSneyd);
        current.values().swap(next.values());
    }
    const double h = grid.firstAxis()[1] - grid.firstAxis()[0];
    const double eigenvalue = (0.1 + 0.15) * 2.0 * (std::cos(pi * h) - 1.0) / (h * h);
    return std::abs(current(4, 4) - std::exp(eigenvalue * finalTime));
}

TEST(Adi, DouglasManufacturedSolution)
{
    EXPECT_LT(oneStepEigenError(AdiScheme::Douglas, 0.001), 2.0e-5);
}

TEST(Adi, McsManufacturedSolution)
{
    EXPECT_LT(oneStepEigenError(AdiScheme::ModifiedCraigSneyd, 0.001), 2.0e-6);
}

TEST(Adi, McsHandlesMixedDerivative)
{
    const Grid2D grid = adiGrid(7);
    SplitOperator2D split = heatOperator(grid);
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            const std::size_t index = grid.index(first, second);
            split.explicitCoefficient(-1, -1, index) = 0.05;
            split.explicitCoefficient(1, 1, index) = 0.05;
            split.explicitCoefficient(-1, 1, index) = -0.05;
            split.explicitCoefficient(1, -1, index) = -0.05;
        }
    auto input = eigenmode(grid);
    Surface2D<double> output(grid);
    AdiWorkspace workspace;
    workspace.initialize(split, 0.001);
    EXPECT_NO_THROW(AdiSolver::step(split, input, output, 0.0, zeroBoundary,
                                    workspace, AdiScheme::ModifiedCraigSneyd));
    EXPECT_TRUE(std::all_of(output.values().begin(), output.values().end(),
                            [](double value) { return std::isfinite(value); }));
}

TEST(Adi, McsTimeRefinementReducesError)
{
    EXPECT_LT(mcsEvolutionError(8), mcsEvolutionError(4));
}

TEST(Adi, McsShowsSecondOrderTimeConvergenceForSmoothSolution)
{
#ifdef DR3_ENABLE_SLOW_NUMERICAL_TESTS
    const double coarse = mcsEvolutionError(4);
    const double fine = mcsEvolutionError(8);
    EXPECT_GT(coarse / fine, 3.0);
#else
    GTEST_SKIP() << "enable DR3_ENABLE_SLOW_NUMERICAL_TESTS for convergence order";
#endif
}

TEST(Adi, RannacherHandlesNonsmoothInitialCondition)
{
    const Grid2D grid = adiGrid(9);
    SplitOperator2D split = heatOperator(grid);
    Surface2D<double> payoffSurface(grid, 0.0), scratch(grid), output(grid);
    for (std::size_t second = 0; second < grid.secondSize(); ++second)
        for (std::size_t first = 0; first < grid.firstSize(); ++first)
            payoffSurface(first, second) = std::max(grid.firstAxis()[first] - 0.5, 0.0)
                * std::sin(pi * grid.secondAxis()[second]);
    AdiWorkspace workspace;
    workspace.initialize(split, 0.0005, 1.0);
    AdiSolver::rannacherStep(split, payoffSurface, scratch, output,
                            0.0, zeroBoundary, workspace);
    EXPECT_TRUE(std::all_of(output.values().begin(), output.values().end(),
                            [](double value) { return std::isfinite(value); }));
}

TEST(Adi, BoundariesAppliedAtEveryStage)
{
    const Grid2D grid = adiGrid(5);
    SplitOperator2D split = heatOperator(grid);
    auto input = eigenmode(grid);
    Surface2D<double> output(grid);
    std::size_t applications = 0;
    const dr3::advanced::AdiBoundary boundary = [&applications](double, const Grid2D& boundaryGrid,
                                                                double* values)
    {
        ++applications;
        zeroBoundary(0.0, boundaryGrid, values);
    };
    AdiWorkspace workspace;
    workspace.initialize(split, 0.001);
    AdiSolver::step(split, input, output, 0.0, boundary, workspace,
                    AdiScheme::ModifiedCraigSneyd);
    EXPECT_GE(applications, 7U);
}

TEST(Adi, ReusesDirectionalFactorizations)
{
    const Grid2D grid = adiGrid(5);
    SplitOperator2D split = heatOperator(grid);
    auto input = eigenmode(grid);
    Surface2D<double> first(grid), second(grid);
    AdiWorkspace workspace;
    workspace.initialize(split, 0.001);
    const auto count = workspace.factorizationBuildCount();
    AdiSolver::step(split, input, first, 0.0, zeroBoundary, workspace);
    AdiSolver::step(split, first, second, 0.001, zeroBoundary, workspace);
    EXPECT_EQ(workspace.factorizationBuildCount(), count);
}

TEST(Adi, NoAllocationDuringStep)
{
    const Grid2D grid = adiGrid(5);
    SplitOperator2D split = heatOperator(grid);
    auto input = eigenmode(grid);
    Surface2D<double> output(grid);
    AdiWorkspace workspace;
    workspace.initialize(split, 0.001);
    const auto* outputAddress = output.data();
    const auto capacity = output.values().capacity();
    AdiSolver::step(split, input, output, 0.0, zeroBoundary, workspace);
    EXPECT_EQ(output.data(), outputAddress);
    EXPECT_EQ(output.values().capacity(), capacity);
}

TEST(Adi, RejectsInvalidTheta)
{
    const Grid2D grid = adiGrid(5);
    SplitOperator2D split = heatOperator(grid);
    AdiWorkspace workspace;
    EXPECT_THROW(workspace.initialize(split, 0.01, 0.0), std::invalid_argument);
    EXPECT_THROW(workspace.initialize(split, 0.01, 1.1), std::invalid_argument);
}

TEST(Adi, RejectsShapeMismatch)
{
    const Grid2D grid = adiGrid(5);
    SplitOperator2D split = heatOperator(grid);
    Surface2D<double> input(grid), output(adiGrid(6));
    AdiWorkspace workspace;
    workspace.initialize(split, 0.01);
    EXPECT_THROW(AdiSolver::step(split, input, output, 0.0, zeroBoundary, workspace),
                 std::invalid_argument);
}

TEST(Adi, OneStepMatchesTinyDenseReference)
{
    const Grid2D grid = adiGrid(3);
    SplitOperator2D split(grid);
    const std::size_t center = grid.index(1, 1);
    split.explicitCoefficient(0, 0, center) = -0.2;
    split.firstDiagonal(center) = -0.3;
    split.secondDiagonal(center) = -0.4;
    Surface2D<double> input(grid, 0.0), output(grid);
    input(1, 1) = 1.25;
    AdiWorkspace workspace;
    constexpr double dt = 0.01;
    constexpr double theta = 1.0 / 3.0;
    workspace.initialize(split, dt, theta);
    AdiSolver::step(split, input, output, 0.0, zeroBoundary, workspace,
                    AdiScheme::ModifiedCraigSneyd);

    // Independently assembled dense 1x1 MCS equations.
    constexpr double u = 1.25;
    constexpr double a0 = -0.2;
    constexpr double a1 = -0.3;
    constexpr double a2 = -0.4;
    constexpr double a = a0 + a1 + a2;
    const double y0 = u + dt * a * u;
    const double y1 = (y0 - theta * dt * a1 * u) / (1.0 - theta * dt * a1);
    const double y2 = (y1 - theta * dt * a2 * u) / (1.0 - theta * dt * a2);
    const double hat = y0 + theta * dt * a0 * (y2 - u);
    const double corrected = hat + (0.5 - theta) * dt * a * (y2 - u);
    const double corrected1 = (corrected - theta * dt * a1 * u)
        / (1.0 - theta * dt * a1);
    const double denseReference = (corrected1 - theta * dt * a2 * u)
        / (1.0 - theta * dt * a2);
    EXPECT_NEAR(output(1, 1), denseReference, 1.0e-14);
}

} // namespace
