#include "batched_pde.h"

#include "../Vectorisation/Curves/curve.h"
#include "../Vectorisation/VecX/vcl_latest.h"
#include "tree_pricer_detail.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dr3::lattice
{
namespace
{

constexpr std::size_t simdWidth = 4;
using Simd = Vec4d;
using Mask = Vec4db;

struct ForwardDual
{
    Simd value;
    Simd derivative;
};

ForwardDual operator+(const ForwardDual& lhs, const ForwardDual& rhs)
{
    return {lhs.value + rhs.value, lhs.derivative + rhs.derivative};
}
ForwardDual operator-(const ForwardDual& lhs, const ForwardDual& rhs)
{
    return {lhs.value - rhs.value, lhs.derivative - rhs.derivative};
}
ForwardDual operator-(const ForwardDual& value)
{
    return {-value.value, -value.derivative};
}
ForwardDual operator*(const ForwardDual& lhs, const ForwardDual& rhs)
{
    return {lhs.value * rhs.value,
            lhs.value * rhs.derivative + rhs.value * lhs.derivative};
}
ForwardDual operator/(const ForwardDual& lhs, const ForwardDual& rhs)
{
    return {lhs.value / rhs.value,
            (lhs.derivative * rhs.value - lhs.value * rhs.derivative)
                / (rhs.value * rhs.value)};
}

template <typename Value>
Value makeValue(const Simd& primal, const Simd& derivative);

template <>
Simd makeValue<Simd>(const Simd& primal, const Simd&)
{
    return primal;
}

template <>
ForwardDual makeValue<ForwardDual>(const Simd& primal, const Simd& derivative)
{
    return {primal, derivative};
}

Simd primal(const Simd& value) { return value; }
Simd primal(const ForwardDual& value) { return value.value; }
Simd tangent(const Simd&) { return Simd(0.0); }
Simd tangent(const ForwardDual& value) { return value.derivative; }

Simd exponential(const Simd& value) { return exp(value); }
ForwardDual exponential(const ForwardDual& value)
{
    const Simd result = exp(value.value);
    return {result, result * value.derivative};
}

Simd selectValue(const Mask& mask, const Simd& whenTrue, const Simd& whenFalse)
{
    return select(mask, whenTrue, whenFalse);
}
ForwardDual selectValue(const Mask& mask,
                        const ForwardDual& whenTrue,
                        const ForwardDual& whenFalse)
{
    return {select(mask, whenTrue.value, whenFalse.value),
            select(mask, whenTrue.derivative, whenFalse.derivative)};
}

template <typename Value>
Value positivePart(const Value& value)
{
    const auto mask = primal(value) > Simd(0.0);
    return selectValue(mask, value, makeValue<Value>(Simd(0.0), Simd(0.0)));
}

Simd pack(const std::array<double, simdWidth>& values)
{
    return {values[0], values[1], values[2], values[3]};
}

template <typename Getter>
Simd packOptions(const std::vector<VanillaOption>& options,
                 std::size_t begin,
                 std::size_t active,
                 Getter&& getter)
{
    std::array<double, simdWidth> values{};
    for (std::size_t lane = 0; lane < simdWidth; ++lane)
        values[lane] = getter(options[begin + std::min(lane, active - 1)]);
    return pack(values);
}

Simd packSeeds(const std::vector<double>& seeds,
               std::size_t begin,
               std::size_t active)
{
    std::array<double, simdWidth> values{};
    for (std::size_t lane = 0; lane < active; ++lane)
        values[lane] = seeds.empty() ? 0.0 : seeds[begin + lane];
    return pack(values);
}

Mask optionTypeMask(const std::vector<VanillaOption>& options,
                    std::size_t begin,
                    std::size_t active)
{
    std::array<bool, simdWidth> calls{};
    for (std::size_t lane = 0; lane < simdWidth; ++lane)
        calls[lane] = options[begin + std::min(lane, active - 1)].type == OptionType::Call;
    return {calls[0], calls[1], calls[2], calls[3]};
}

Mask activeMask(std::size_t active)
{
    return {active > 0, active > 1, active > 2, active > 3};
}

template <typename Value>
struct PackedInputs
{
    Value strike;
    Value volatility;
    Value rate;
    Value dividend;
    Value maturity;
    Mask calls;
    Mask active;
    std::size_t activeCount;
};

template <typename Value>
PackedInputs<Value> packInputs(const std::vector<VanillaOption>& options,
                               std::size_t begin,
                               std::size_t active,
                               const std::vector<double>& rateSeeds,
                               const std::vector<double>& volatilitySeeds)
{
    const Simd zero(0.0);
    return {
        makeValue<Value>(packOptions(options, begin, active,
            [](const auto& option) { return option.strike; }), zero),
        makeValue<Value>(packOptions(options, begin, active,
            [](const auto& option) { return option.volatility; }),
            packSeeds(volatilitySeeds, begin, active)),
        makeValue<Value>(packOptions(options, begin, active,
            [](const auto& option) { return option.rate; }),
            packSeeds(rateSeeds, begin, active)),
        makeValue<Value>(packOptions(options, begin, active,
            [](const auto& option) { return option.dividendYield; }), zero),
        makeValue<Value>(packOptions(options, begin, active,
            [](const auto& option) { return option.maturity; }), zero),
        optionTypeMask(options, begin, active), activeMask(active), active};
}

template <typename Value>
struct BatchedSpatialOperator
{
    std::vector<Value> lower;
    std::vector<Value> main;
    std::vector<Value> upper;
};

template <typename Value>
struct BatchedTridiagonalSystem
{
    std::vector<Value> lower;
    std::vector<Value> main;
    std::vector<Value> upper;
};

template <typename Value>
class BatchedThomasFactorization
{
public:
    BatchedThomasFactorization(const BatchedTridiagonalSystem<Value>& system,
                               std::size_t active,
                               std::size_t globalOffset)
        : multipliers_(system.lower.size()), pivots_(system.main), upper_(system.upper)
    {
        validatePivot(pivots_.front(), active, globalOffset);
        for (std::size_t row = 1; row < pivots_.size(); ++row)
        {
            multipliers_[row - 1] = system.lower[row - 1] / pivots_[row - 1];
            pivots_[row] = pivots_[row] - multipliers_[row - 1] * upper_[row - 1];
            validatePivot(pivots_[row], active, globalOffset);
        }
    }

    void solve(const std::vector<Value>& rhs,
               std::vector<Value>& output,
               std::vector<Value>& workspace) const
    {
        workspace[0] = rhs[0];
        for (std::size_t row = 1; row < pivots_.size(); ++row)
            workspace[row] = rhs[row] - multipliers_[row - 1] * workspace[row - 1];
        output.back() = workspace.back() / pivots_.back();
        for (std::size_t row = pivots_.size() - 1; row-- > 0;)
            output[row] = (workspace[row] - upper_[row] * output[row + 1]) / pivots_[row];
    }

private:
    static void validatePivot(const Value& pivot,
                              std::size_t active,
                              std::size_t globalOffset)
    {
        const auto value = primal(pivot);
        for (std::size_t lane = 0; lane < active; ++lane)
            if (!std::isfinite(value[static_cast<int>(lane)])
                || std::abs(value[static_cast<int>(lane)]) <= 1.0e-14)
                throw std::runtime_error("batched PDE lane "
                    + std::to_string(globalOffset + lane) + " has a singular pivot");
    }

    std::vector<Value> multipliers_;
    std::vector<Value> pivots_;
    std::vector<Value> upper_;
};

template <typename Value>
BatchedSpatialOperator<Value> makeSpatialOperator(const PackedInputs<Value>& input,
                                                  const Grid1D& grid)
{
    const auto zero = makeValue<Value>(Simd(0.0), Simd(0.0));
    BatchedSpatialOperator<Value> op{
        std::vector<Value>(grid.nodeCount(), zero),
        std::vector<Value>(grid.nodeCount(), zero),
        std::vector<Value>(grid.nodeCount(), zero)};
    const double inverseSpacing = 1.0 / grid.spacing();
    const double inverseSpacingSquared = inverseSpacing * inverseSpacing;
    const Value variance = input.volatility * input.volatility;
    const Value carry = input.rate - input.dividend;
    for (std::size_t index = grid.interiorBegin(); index < grid.interiorEnd(); ++index)
    {
        const auto diffusion = variance * makeValue<Value>(
            Simd(0.5 * grid[index] * grid[index] * inverseSpacingSquared), Simd(0.0));
        const auto convection = carry * makeValue<Value>(
            Simd(0.5 * grid[index] * inverseSpacing), Simd(0.0));
        op.lower[index] = diffusion - convection;
        op.main[index] = -diffusion - diffusion - input.rate;
        op.upper[index] = diffusion + convection;
    }
    return op;
}

template <typename Value>
BatchedTridiagonalSystem<Value> makeSystem(const BatchedSpatialOperator<Value>& op,
                                           std::size_t nodeCount,
                                           double theta,
                                           const Value& timeStep)
{
    const std::size_t count = nodeCount - 2;
    const auto zero = makeValue<Value>(Simd(0.0), Simd(0.0));
    const auto one = makeValue<Value>(Simd(1.0), Simd(0.0));
    const auto thetaValue = makeValue<Value>(Simd(theta), Simd(0.0));
    BatchedTridiagonalSystem<Value> system{
        std::vector<Value>(count - 1, zero), std::vector<Value>(count, zero),
        std::vector<Value>(count - 1, zero)};
    for (std::size_t row = 0; row < count; ++row)
    {
        const std::size_t index = row + 1;
        system.main[row] = one - thetaValue * timeStep * op.main[index];
        if (row > 0) system.lower[row - 1] = -thetaValue * timeStep * op.lower[index];
        if (row + 1 < count)
            system.upper[row] = -thetaValue * timeStep * op.upper[index];
    }
    return system;
}

template <typename Value>
Value payoff(const PackedInputs<Value>& input, double spot)
{
    const auto spotValue = makeValue<Value>(Simd(spot), Simd(0.0));
    const auto call = positivePart(spotValue - input.strike);
    const auto put = positivePart(input.strike - spotValue);
    return selectValue(input.calls, call, put);
}

template <typename Value>
std::pair<Value, Value> boundary(const PackedInputs<Value>& input,
                                const Grid1D& grid,
                                const Value& time)
{
    const auto minimum = makeValue<Value>(Simd(grid.minimum()), Simd(0.0));
    const auto maximum = makeValue<Value>(Simd(grid.maximum()), Simd(0.0));
    const auto discountedStrike = input.strike * exponential(-input.rate * time);
    const auto discountedMinimum = minimum * exponential(-input.dividend * time);
    const auto discountedMaximum = maximum * exponential(-input.dividend * time);
    const auto callLower = positivePart(discountedMinimum - discountedStrike);
    const auto callUpper = positivePart(discountedMaximum - discountedStrike);
    const auto putLower = positivePart(discountedStrike - discountedMinimum);
    const auto putUpper = positivePart(discountedStrike - discountedMaximum);
    return {selectValue(input.calls, callLower, putLower),
            selectValue(input.calls, callUpper, putUpper)};
}

template <typename Value>
void clampFinite(std::vector<Value>& values,
                 std::size_t active,
                 std::size_t globalOffset)
{
    for (auto& value : values)
    {
        const auto p = primal(value);
        for (std::size_t lane = 0; lane < active; ++lane)
            if (!std::isfinite(p[static_cast<int>(lane)]))
                throw std::runtime_error("batched PDE lane "
                    + std::to_string(globalOffset + lane) + " produced a non-finite value");
        value = positivePart(value);
    }
}

template <typename Value>
void thetaStep(const PackedInputs<Value>& input,
               const Grid1D& grid,
               const BatchedSpatialOperator<Value>& op,
               const BatchedThomasFactorization<Value>& factorization,
               double theta,
               const Value& timeStep,
               const Value& newTime,
               std::size_t globalOffset,
               std::vector<Value>& previous,
               std::vector<Value>& next,
               std::vector<Value>& rhs,
               std::vector<Value>& solution,
               std::vector<Value>& solver)
{
    const auto boundaries = boundary(input, grid, newTime);
    next.front() = boundaries.first;
    next.back() = boundaries.second;
    const auto explicitWeight = makeValue<Value>(Simd(1.0 - theta), Simd(0.0)) * timeStep;
    for (std::size_t index = grid.interiorBegin(); index < grid.interiorEnd(); ++index)
        rhs[index] = previous[index] + explicitWeight
            * (op.lower[index] * previous[index - 1]
               + op.main[index] * previous[index]
               + op.upper[index] * previous[index + 1]);
    const auto implicitWeight = makeValue<Value>(Simd(theta), Simd(0.0)) * timeStep;
    rhs[1] = rhs[1] + implicitWeight * op.lower[1] * next.front();
    const std::size_t finalInterior = grid.nodeCount() - 2;
    rhs[finalInterior] = rhs[finalInterior]
        + implicitWeight * op.upper[finalInterior] * next.back();

    const std::size_t count = grid.nodeCount() - 2;
    std::copy(rhs.begin() + 1, rhs.begin() + 1 + count, solution.begin());
    factorization.solve(solution, solution, solver);
    std::copy(solution.begin(), solution.end(), next.begin() + 1);
    clampFinite(next, input.activeCount, globalOffset);
    previous.swap(next);
}

template <typename Value>
void explicitStep(const PackedInputs<Value>& input,
                  const Grid1D& grid,
                  const BatchedSpatialOperator<Value>& op,
                  const Value& timeStep,
                  const Value& newTime,
                  std::size_t globalOffset,
                  std::vector<Value>& previous,
                  std::vector<Value>& next)
{
    const auto boundaries = boundary(input, grid, newTime);
    next.front() = boundaries.first;
    next.back() = boundaries.second;
    for (std::size_t index = grid.interiorBegin(); index < grid.interiorEnd(); ++index)
        next[index] = previous[index] + timeStep
            * (op.lower[index] * previous[index - 1]
               + op.main[index] * previous[index]
               + op.upper[index] * previous[index + 1]);
    clampFinite(next, input.activeCount, globalOffset);
    previous.swap(next);
}

template <typename Value>
void validateExplicit(const BatchedSpatialOperator<Value>& op,
                      const Grid1D& grid,
                      const Value& timeStep,
                      std::size_t active,
                      std::size_t globalOffset)
{
    for (std::size_t index = grid.interiorBegin(); index < grid.interiorEnd(); ++index)
    {
        const auto lower = primal(timeStep * op.lower[index]);
        const auto main = primal(makeValue<Value>(Simd(1.0), Simd(0.0))
                                 + timeStep * op.main[index]);
        const auto upper = primal(timeStep * op.upper[index]);
        for (std::size_t lane = 0; lane < active; ++lane)
            if (lower[static_cast<int>(lane)] < -1.0e-14
                || main[static_cast<int>(lane)] < -1.0e-14
                || upper[static_cast<int>(lane)] < -1.0e-14)
                throw std::invalid_argument("batched PDE lane "
                    + std::to_string(globalOffset + lane) + " has an unstable explicit scheme");
    }
}

template <typename Value>
std::pair<Simd, Simd> solveChunk(const std::vector<VanillaOption>& options,
                                std::size_t begin,
                                std::size_t active,
                                const PdeConfig& config,
                                const std::vector<double>& rateSeeds,
                                const std::vector<double>& volatilitySeeds)
{
    const auto input = packInputs<Value>(options, begin, active, rateSeeds, volatilitySeeds);
    Grid1D grid(config.minimumSpot, config.maximumSpot, config.spaceSteps);
    const auto zero = makeValue<Value>(Simd(0.0), Simd(0.0));
    std::vector<Value> previous(grid.nodeCount(), zero), next(grid.nodeCount(), zero),
        rhs(grid.nodeCount(), zero), solution(grid.nodeCount() - 2, zero),
        solver(grid.nodeCount() - 2, zero);
    for (std::size_t index = 0; index < grid.nodeCount(); ++index)
        previous[index] = payoff(input, grid[index]);

    const auto op = makeSpatialOperator(input, grid);
    const auto timeSteps = makeValue<Value>(Simd(static_cast<double>(config.timeSteps)), Simd(0.0));
    const auto timeStep = input.maturity / timeSteps;
    if (config.scheme == PdeScheme::Explicit)
    {
        validateExplicit(op, grid, timeStep, active, begin);
        for (std::size_t step = 1; step <= config.timeSteps; ++step)
            explicitStep(input, grid, op, timeStep,
                         timeStep * makeValue<Value>(Simd(static_cast<double>(step)), Simd(0.0)),
                         begin, previous, next);
    }
    else
    {
        const auto backwardSystem = makeSystem(op, grid.nodeCount(), 1.0, timeStep);
        const BatchedThomasFactorization<Value> backward(backwardSystem, active, begin);
        const auto crankSystem = makeSystem(op, grid.nodeCount(), 0.5, timeStep);
        const BatchedThomasFactorization<Value> crank(crankSystem, active, begin);
        const auto halfStep = timeStep * makeValue<Value>(Simd(0.5), Simd(0.0));
        const auto halfSystem = makeSystem(op, grid.nodeCount(), 1.0, halfStep);
        const BatchedThomasFactorization<Value> halfBackward(halfSystem, active, begin);
        std::size_t firstStep = 1;
        if (config.scheme == PdeScheme::CrankNicolsonRannacher)
        {
            thetaStep(input, grid, op, halfBackward, 1.0, halfStep, halfStep,
                      begin, previous, next, rhs, solution, solver);
            thetaStep(input, grid, op, halfBackward, 1.0, halfStep, timeStep,
                      begin, previous, next, rhs, solution, solver);
            firstStep = 2;
        }
        for (std::size_t step = firstStep; step <= config.timeSteps; ++step)
        {
            const auto newTime = timeStep
                * makeValue<Value>(Simd(static_cast<double>(step)), Simd(0.0));
            if (config.scheme == PdeScheme::BackwardEuler)
                thetaStep(input, grid, op, backward, 1.0, timeStep, newTime,
                          begin, previous, next, rhs, solution, solver);
            else
                thetaStep(input, grid, op, crank, 0.5, timeStep, newTime,
                          begin, previous, next, rhs, solution, solver);
        }
    }

    std::array<double, simdWidth> price{}, derivative{};
    for (std::size_t lane = 0; lane < active; ++lane)
    {
        const auto location = grid.locate(options[begin + lane].spot);
        const auto lower = primal(previous[location.lowerIndex]);
        const auto upper = primal(previous[location.upperIndex]);
        const auto lowerD = tangent(previous[location.lowerIndex]);
        const auto upperD = tangent(previous[location.upperIndex]);
        price[lane] = lower[static_cast<int>(lane)] * (1.0 - location.upperWeight)
            + upper[static_cast<int>(lane)] * location.upperWeight;
        derivative[lane] = lowerD[static_cast<int>(lane)] * (1.0 - location.upperWeight)
            + upperD[static_cast<int>(lane)] * location.upperWeight;
    }
    return {select(input.active, pack(price), Simd(0.0)),
            select(input.active, pack(derivative), Simd(0.0))};
}

void validateBatch(const std::vector<VanillaOption>& options, const PdeConfig& config)
{
    if (config.spaceSteps < 3 || config.timeSteps < 1
        || !std::isfinite(config.minimumSpot) || !std::isfinite(config.maximumSpot)
        || config.maximumSpot <= config.minimumSpot)
        throw std::invalid_argument("batched PDE configuration is invalid");
    if (config.scheme != PdeScheme::Explicit && config.scheme != PdeScheme::BackwardEuler
        && config.scheme != PdeScheme::CrankNicolson
        && config.scheme != PdeScheme::CrankNicolsonRannacher)
        throw std::invalid_argument("batched PDE scheme is invalid");
    for (std::size_t lane = 0; lane < options.size(); ++lane)
    {
        try
        {
            detail::validate(options[lane], {1});
            if (options[lane].spot < config.minimumSpot
                || options[lane].spot > config.maximumSpot)
                throw std::invalid_argument("spot is outside the shared grid");
        }
        catch (const std::exception& error)
        {
            throw std::invalid_argument("batched PDE lane " + std::to_string(lane)
                                        + ": " + error.what());
        }
    }
}

template <typename Value>
BatchedPdeSensitivityResult solveAll(const std::vector<VanillaOption>& options,
                                     const PdeConfig& config,
                                     const std::vector<double>& rateSeeds,
                                     const std::vector<double>& volatilitySeeds)
{
    validateBatch(options, config);
    BatchedPdeSensitivityResult result;
    result.prices.resize(options.size());
    result.sensitivities.resize(options.size());
    result.simdWidth = simdWidth;
    for (std::size_t begin = 0; begin < options.size(); begin += simdWidth)
    {
        const std::size_t active = std::min(simdWidth, options.size() - begin);
        const auto packed = solveChunk<Value>(options, begin, active, config,
                                              rateSeeds, volatilitySeeds);
        for (std::size_t lane = 0; lane < active; ++lane)
        {
            result.prices[begin + lane] = packed.first[static_cast<int>(lane)];
            result.sensitivities[begin + lane] = packed.second[static_cast<int>(lane)];
        }
    }
    return result;
}

} // namespace

BatchedPdeResult batchedEuropeanPdePrices(const std::vector<VanillaOption>& options,
                                          const PdeConfig& config)
{
    auto result = solveAll<Simd>(options, config, {}, {});
    return {std::move(result.prices), result.simdWidth};
}

BatchedPdeSensitivityResult batchedEuropeanPdeForwardSensitivities(
    const std::vector<VanillaOption>& options,
    const PdeConfig& config,
    ForwardPdeParameter parameter)
{
    if (parameter != ForwardPdeParameter::Rate
        && parameter != ForwardPdeParameter::Volatility)
        throw std::invalid_argument("unsupported forward PDE sensitivity parameter");
    std::vector<double> rateSeeds(options.size(), 0.0);
    std::vector<double> volatilitySeeds(options.size(), 0.0);
    auto& seeds = parameter == ForwardPdeParameter::Rate ? rateSeeds : volatilitySeeds;
    std::fill(seeds.begin(), seeds.end(), 1.0);
    return solveAll<ForwardDual>(options, config, rateSeeds, volatilitySeeds);
}

CurvePdeSensitivityResult batchedEuropeanPdeCurveNodeSensitivities(
    const std::vector<VanillaOption>& options,
    const PdeConfig& config,
    const std::vector<double>& curvePillars,
    const std::vector<double>& zeroRates)
{
    const dr3::numerics::Curve<double> curve(curvePillars, zeroRates);
    auto curveOptions = options;
    std::vector<std::vector<double>> weights(options.size());
    for (std::size_t option = 0; option < options.size(); ++option)
    {
        curveOptions[option].rate = curve.evaluate(options[option].maturity);
        weights[option] = curve.nodeSensitivities(options[option].maturity);
    }
    CurvePdeSensitivityResult result;
    result.nodeSensitivities.assign(options.size(), std::vector<double>(zeroRates.size()));
    result.simdWidth = simdWidth;
    for (std::size_t node = 0; node < zeroRates.size(); ++node)
    {
        std::vector<double> seeds(options.size());
        for (std::size_t option = 0; option < options.size(); ++option)
            seeds[option] = weights[option][node];
        auto nodeResult = solveAll<ForwardDual>(curveOptions, config, seeds,
                                                std::vector<double>(options.size(), 0.0));
        if (node == 0) result.prices = std::move(nodeResult.prices);
        for (std::size_t option = 0; option < options.size(); ++option)
            result.nodeSensitivities[option][node] = nodeResult.sensitivities[option];
    }
    return result;
}

} // namespace dr3::lattice
