#pragma once

#include "european_pde.h"

#include <cstddef>
#include <vector>

namespace dr3::lattice
{

enum class ForwardPdeParameter
{
    Rate,
    Volatility
};

struct BatchedPdeResult
{
    std::vector<double> prices;
    std::size_t simdWidth;
};

struct BatchedPdeSensitivityResult
{
    std::vector<double> prices;
    std::vector<double> sensitivities;
    std::size_t simdWidth;
};

struct CurvePdeSensitivityResult
{
    std::vector<double> prices;
    // One complete curve-node sensitivity row per option.
    std::vector<std::vector<double>> nodeSensitivities;
    std::size_t simdWidth;
};

BatchedPdeResult batchedEuropeanPdePrices(const std::vector<VanillaOption>& options,
                                          const PdeConfig& config);

BatchedPdeSensitivityResult batchedEuropeanPdeForwardSensitivities(
    const std::vector<VanillaOption>& options,
    const PdeConfig& config,
    ForwardPdeParameter parameter);

// Forward sensitivities are provided for smooth European payoffs. Derivatives
// at American early-exercise kinks are explicitly outside this API's guarantee.
CurvePdeSensitivityResult batchedEuropeanPdeCurveNodeSensitivities(
    const std::vector<VanillaOption>& options,
    const PdeConfig& config,
    const std::vector<double>& curvePillars,
    const std::vector<double>& zeroRates);

} // namespace dr3::lattice
