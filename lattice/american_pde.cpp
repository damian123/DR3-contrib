#include "american_pde.h"

#include "pde_workspace.h"
#include "tree_pricer_detail.h"
#include "tridiagonal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dr3::lattice
{
namespace
{

struct SpatialOperator
{
    std::vector<double> lower;
    std::vector<double> main;
    std::vector<double> upper;
};

struct PsorStepResult
{
    bool converged;
    std::size_t iterations;
    double residual;
};

void validate(const VanillaOption& option,
              const PdeConfig& config,
              const PsorConfig& psor)
{
    detail::validate(option, {1});
    if (config.spaceSteps < 3 || config.timeSteps < 1)
        throw std::invalid_argument("American PDE grid requires at least three space nodes and one time step");
    if (!std::isfinite(config.minimumSpot) || !std::isfinite(config.maximumSpot)
        || config.maximumSpot <= config.minimumSpot
        || option.spot < config.minimumSpot || option.spot > config.maximumSpot)
        throw std::invalid_argument("American PDE spot bounds are invalid");
    if (config.scheme == PdeScheme::Explicit)
        throw std::invalid_argument("American PDE requires an implicit projected scheme");
    if (config.scheme != PdeScheme::BackwardEuler
        && config.scheme != PdeScheme::CrankNicolson
        && config.scheme != PdeScheme::CrankNicolsonRannacher)
        throw std::invalid_argument("American PDE scheme is invalid");
    if (!std::isfinite(psor.omega) || psor.omega <= 0.0 || psor.omega >= 2.0)
        throw std::invalid_argument("PSOR omega must be finite and strictly between zero and two");
    if (!std::isfinite(psor.tolerance) || psor.tolerance <= 0.0)
        throw std::invalid_argument("PSOR tolerance must be finite and positive");
    if (psor.maximumIterations < 1)
        throw std::invalid_argument("PSOR maximumIterations must be at least one");
}

SpatialOperator makeSpatialOperator(const VanillaOption& option, const Grid1D& grid)
{
    SpatialOperator op{std::vector<double>(grid.nodeCount()),
                       std::vector<double>(grid.nodeCount()),
                       std::vector<double>(grid.nodeCount())};
    const double inverseSpacing = 1.0 / grid.spacing();
    const double inverseSpacingSquared = inverseSpacing * inverseSpacing;
    const double variance = option.volatility * option.volatility;
    const double carry = option.rate - option.dividendYield;
    for (std::size_t index = grid.interiorBegin(); index < grid.interiorEnd(); ++index)
    {
        const double diffusion = 0.5 * variance * grid[index] * grid[index]
            * inverseSpacingSquared;
        const double convection = 0.5 * carry * grid[index] * inverseSpacing;
        op.lower[index] = diffusion - convection;
        op.main[index] = -2.0 * diffusion - option.rate;
        op.upper[index] = diffusion + convection;
    }
    return op;
}

TridiagonalSystem makeSystem(const SpatialOperator& op,
                             std::size_t nodeCount,
                             double theta,
                             double timeStep)
{
    const std::size_t interiorCount = nodeCount - 2;
    TridiagonalSystem system;
    system.lower.resize(interiorCount - 1);
    system.main.resize(interiorCount);
    system.upper.resize(interiorCount - 1);
    for (std::size_t row = 0; row < interiorCount; ++row)
    {
        const std::size_t index = row + 1;
        system.main[row] = 1.0 - theta * timeStep * op.main[index];
        if (row > 0) system.lower[row - 1] = -theta * timeStep * op.lower[index];
        if (row + 1 < interiorCount)
            system.upper[row] = -theta * timeStep * op.upper[index];
    }
    system.validate();
    return system;
}

void applyAmericanBoundaries(std::vector<double>& values,
                             const std::vector<double>& obstacle,
                             const VanillaOption& option,
                             const Grid1D& grid,
                             double timeToMaturity)
{
    const auto european = europeanBoundaryValues(
        option, grid.minimum(), grid.maximum(), timeToMaturity);
    values.front() = std::max(european.lower, obstacle.front());
    values.back() = std::max(european.upper, obstacle.back());
}

double complementarityResidual(const TridiagonalSystem& system,
                               const std::vector<double>& rhs,
                               const std::vector<double>& values,
                               const std::vector<double>& obstacle)
{
    double residual = 0.0;
    for (std::size_t row = 0; row < system.size(); ++row)
    {
        double applied = system.main[row] * values[row + 1];
        if (row > 0) applied += system.lower[row - 1] * values[row];
        if (row + 1 < system.size())
            applied += system.upper[row] * values[row + 2];
        const double obstacleGap = values[row + 1] - obstacle[row + 1];
        const double equationGap = applied - rhs[row + 1];
        const double rowResidual = std::max({
            std::max(-obstacleGap, 0.0),
            std::max(-equationGap, 0.0),
            std::abs(std::min(obstacleGap, equationGap))});
        residual = std::max(residual, rowResidual);
    }
    return residual;
}

PsorStepResult projectedSor(const TridiagonalSystem& system,
                            const std::vector<double>& rhs,
                            const std::vector<double>& obstacle,
                            const PsorConfig& config,
                            std::vector<double>& values)
{
    double residual = std::numeric_limits<double>::infinity();
    for (std::size_t iteration = 1; iteration <= config.maximumIterations; ++iteration)
    {
        for (std::size_t row = 0; row < system.size(); ++row)
        {
            double offDiagonal = 0.0;
            if (row > 0) offDiagonal += system.lower[row - 1] * values[row];
            if (row + 1 < system.size())
                offDiagonal += system.upper[row] * values[row + 2];
            const double unconstrained = (rhs[row + 1] - offDiagonal) / system.main[row];
            const double relaxed = values[row + 1]
                + config.omega * (unconstrained - values[row + 1]);
            values[row + 1] = std::max(relaxed, obstacle[row + 1]);
        }
        residual = complementarityResidual(system, rhs, values, obstacle);
        if (!std::isfinite(residual))
            return {false, iteration, residual};
        if (residual <= config.tolerance)
            return {true, iteration, residual};
    }
    return {false, config.maximumIterations, residual};
}

PsorStepResult thetaStep(const VanillaOption& option,
                         const Grid1D& grid,
                         const SpatialOperator& op,
                         const TridiagonalSystem& system,
                         const std::vector<double>& obstacle,
                         double theta,
                         double timeStep,
                         double newTime,
                         const PsorConfig& psor,
                         PdeWorkspace& workspace)
{
    applyAmericanBoundaries(workspace.next, obstacle, option, grid, newTime);
    const double explicitWeight = (1.0 - theta) * timeStep;
    for (std::size_t index = grid.interiorBegin(); index < grid.interiorEnd(); ++index)
    {
        workspace.rightHandSide[index] = workspace.previous[index] + explicitWeight
            * (op.lower[index] * workspace.previous[index - 1]
               + op.main[index] * workspace.previous[index]
               + op.upper[index] * workspace.previous[index + 1]);
        workspace.next[index] = std::max(workspace.previous[index], obstacle[index]);
    }
    workspace.rightHandSide[1] += theta * timeStep * op.lower[1]
        * workspace.next.front();
    const std::size_t finalInterior = grid.nodeCount() - 2;
    workspace.rightHandSide[finalInterior] += theta * timeStep
        * op.upper[finalInterior] * workspace.next.back();

    const auto result = projectedSor(system, workspace.rightHandSide,
                                     obstacle, psor, workspace.next);
    for (double value : workspace.next)
        if (!std::isfinite(value))
            return {false, result.iterations, std::numeric_limits<double>::infinity()};
    workspace.previous.swap(workspace.next);
    return result;
}

} // namespace

AmericanPdeResult americanPdePrice(const VanillaOption& option,
                                   const PdeConfig& config,
                                   const PsorConfig& psor)
{
    validate(option, config, psor);
    Grid1D grid(config.minimumSpot, config.maximumSpot, config.spaceSteps);
    PdeWorkspace workspace(grid.nodeCount());
    const auto obstacle = europeanTerminalPayoff(option, grid);
    workspace.previous = obstacle;
    if (option.maturity == 0.0)
        return {grid.interpolate(workspace.previous, option.spot), true, 0, 0.0,
                std::move(grid), std::move(workspace.previous)};

    const auto op = makeSpatialOperator(option, grid);
    const double timeStep = option.maturity / static_cast<double>(config.timeSteps);
    const auto backwardEuler = makeSystem(op, grid.nodeCount(), 1.0, timeStep);
    const auto crankNicolson = makeSystem(op, grid.nodeCount(), 0.5, timeStep);
    const double halfStep = 0.5 * timeStep;
    const auto backwardEulerHalf = makeSystem(op, grid.nodeCount(), 1.0, halfStep);
    std::size_t totalIterations = 0;
    double maximumResidual = 0.0;

    const auto runStep = [&](const TridiagonalSystem& system,
                             double theta, double stepSize, double newTime)
    {
        const auto result = thetaStep(option, grid, op, system, obstacle,
                                      theta, stepSize, newTime, psor, workspace);
        totalIterations += result.iterations;
        maximumResidual = std::max(maximumResidual, result.residual);
        return result.converged;
    };

    bool converged = true;
    std::size_t firstRegularStep = 1;
    if (config.scheme == PdeScheme::CrankNicolsonRannacher)
    {
        converged = runStep(backwardEulerHalf, 1.0, halfStep, halfStep)
            && runStep(backwardEulerHalf, 1.0, halfStep, timeStep);
        firstRegularStep = 2;
    }
    for (std::size_t step = firstRegularStep;
         converged && step <= config.timeSteps; ++step)
    {
        if (config.scheme == PdeScheme::BackwardEuler)
            converged = runStep(backwardEuler, 1.0, timeStep, step * timeStep);
        else
            converged = runStep(crankNicolson, 0.5, timeStep, step * timeStep);
    }

    const double price = grid.interpolate(workspace.previous, option.spot);
    return {price, converged, totalIterations, maximumResidual,
            std::move(grid), std::move(workspace.previous)};
}

} // namespace dr3::lattice
