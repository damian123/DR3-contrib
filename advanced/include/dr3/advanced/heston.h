#pragma once

#include "dr3/advanced/adi.h"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace dr3::advanced
{

enum class EuropeanOption
{
    Call,
    Put
};

struct HestonParameters
{
    double spot{};
    double strike{};
    double maturity{};
    double rate{};
    double dividendYield{};
    double initialVariance{};
    double longRunVariance{};
    double meanReversion{};
    double volatilityOfVariance{};
    double correlation{};
    EuropeanOption option{EuropeanOption::Call};

    void validate() const;
    bool fellerConditionSatisfied() const noexcept;
};

struct HestonReferenceSettings
{
    double absoluteTolerance{1.0e-8};
    double integrationLimit{150.0};
    std::size_t initialIntervals{512};
    std::size_t maxRefinements{5};

    void validate() const;
};

double blackScholesPrice(EuropeanOption option, double spot, double strike,
                         double maturity, double rate, double dividendYield,
                         double volatility);

double hestonReferencePrice(const HestonParameters& parameters,
                            const HestonReferenceSettings& settings = {});

struct HestonPdeSettings
{
    Axis1D spotAxis{Axis1D::uniform(0.0, 400.0, 81)};
    Axis1D varianceAxis{Axis1D::uniform(0.0, 1.0, 41)};
    std::size_t timeSteps{80};
    double theta{1.0 / 3.0};
    std::size_t rannacherSteps{2};
    // Small negative stencil undershoots up to 2e-4 are clamped on return.
    double nonnegativeTolerance{2.0e-4};

    void validate(const HestonParameters& parameters) const;
};

struct HestonPdeResult
{
    bool success{};
    double price{std::numeric_limits<double>::quiet_NaN()};
    bool fellerConditionSatisfied{};
    std::string error;
    std::vector<double> finalSurface;
};

SplitOperator2D makeHestonOperator(const HestonParameters& parameters,
                                   const Grid2D& grid);

HestonPdeResult hestonPdePrice(const HestonParameters& parameters,
                               const HestonPdeSettings& settings = {});

} // namespace dr3::advanced
