#include "dr3/advanced/heston.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace dr3::advanced
{
namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;

double normalCdf(double value)
{
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double deterministicVarianceIntegral(const HestonParameters& parameters)
{
    if (parameters.maturity == 0.0)
    {
        return 0.0;
    }
    if (parameters.meanReversion == 0.0)
    {
        return parameters.initialVariance * parameters.maturity;
    }
    return parameters.longRunVariance * parameters.maturity
        + (parameters.initialVariance - parameters.longRunVariance)
            * (1.0 - std::exp(-parameters.meanReversion * parameters.maturity))
            / parameters.meanReversion;
}

std::complex<double> characteristic(const HestonParameters& parameters,
                                    std::complex<double> argument)
{
    const std::complex<double> imaginary(0.0, 1.0);
    const double sigma = parameters.volatilityOfVariance;
    const double kappa = parameters.meanReversion;
    const double theta = parameters.longRunVariance;
    const double rho = parameters.correlation;
    const double time = parameters.maturity;
    const std::complex<double> iu = imaginary * argument;
    const std::complex<double> beta = kappa - rho * sigma * iu;
    const std::complex<double> discriminant = beta * beta
        + sigma * sigma * (argument * argument + iu);
    std::complex<double> d = std::sqrt(discriminant);
    // Choose the branch with nonnegative real part (the little Heston trap).
    if (d.real() < 0.0)
    {
        d = -d;
    }
    const std::complex<double> g = (beta - d) / (beta + d);
    const std::complex<double> exponential = std::exp(-d * time);
    const std::complex<double> logRatio = std::log((1.0 - g * exponential) / (1.0 - g));
    const std::complex<double> c = iu
            * (std::log(parameters.spot)
               + (parameters.rate - parameters.dividendYield) * time)
        + (kappa * theta / (sigma * sigma))
            * ((beta - d) * time - 2.0 * logRatio);
    const std::complex<double> bigD = ((beta - d) / (sigma * sigma))
        * ((1.0 - exponential) / (1.0 - g * exponential));
    return std::exp(c + bigD * parameters.initialVariance);
}

double probabilityIntegrand(const HestonParameters& parameters,
                            double frequency, bool firstProbability,
                            const std::complex<double>& phiMinusI)
{
    const std::complex<double> imaginary(0.0, 1.0);
    const std::complex<double> u(frequency, 0.0);
    const std::complex<double> shifted = firstProbability
        ? characteristic(parameters, u - imaginary) / phiMinusI
        : characteristic(parameters, u);
    const std::complex<double> oscillation = std::exp(
        -imaginary * frequency * std::log(parameters.strike));
    return (oscillation * shifted / (imaginary * frequency)).real();
}

double simpsonProbability(const HestonParameters& parameters,
                          const HestonReferenceSettings& settings,
                          std::size_t intervals,
                          bool firstProbability,
                          const std::complex<double>& phiMinusI)
{
    constexpr double lower = 1.0e-7;
    const double step = (settings.integrationLimit - lower)
        / static_cast<double>(intervals);
    double weighted = probabilityIntegrand(parameters, lower,
                                            firstProbability, phiMinusI)
        + probabilityIntegrand(parameters, settings.integrationLimit,
                               firstProbability, phiMinusI);
    for (std::size_t index = 1; index < intervals; ++index)
    {
        const double frequency = lower + step * static_cast<double>(index);
        weighted += (index % 2 == 0 ? 2.0 : 4.0)
            * probabilityIntegrand(parameters, frequency,
                                   firstProbability, phiMinusI);
    }
    return 0.5 + step * weighted / (3.0 * pi);
}

std::array<double, 3> secondWeights(const Axis1D& axis, std::size_t index)
{
    const double left = axis[index] - axis[index - 1];
    const double right = axis[index + 1] - axis[index];
    const double total = left + right;
    if (!std::isfinite(left) || !std::isfinite(right) || !std::isfinite(total)
        || left <= 0.0 || right <= 0.0)
    {
        throw std::domain_error("Heston nonuniform second derivative has an invalid spacing");
    }
    return {2.0 / (left * total), -2.0 / (left * right), 2.0 / (right * total)};
}

double payoff(EuropeanOption option, double spot, double strike)
{
    return option == EuropeanOption::Call ? std::max(spot - strike, 0.0)
                                          : std::max(strike - spot, 0.0);
}

double bilinearValue(const Surface2D<double>& surface, double firstValue, double secondValue)
{
    const Grid2D& grid = surface.grid();
    const auto bracket = [](const Axis1D& axis, double value)
    {
        const auto& coordinates = axis.coordinates();
        const auto upper = std::lower_bound(coordinates.begin(), coordinates.end(), value);
        if (upper == coordinates.begin())
        {
            return std::pair<std::size_t, std::size_t>{0, 0};
        }
        if (upper == coordinates.end())
        {
            const std::size_t last = coordinates.size() - 1;
            return std::pair<std::size_t, std::size_t>{last, last};
        }
        const std::size_t right = static_cast<std::size_t>(upper - coordinates.begin());
        if (*upper == value)
        {
            return std::pair<std::size_t, std::size_t>{right, right};
        }
        return std::pair<std::size_t, std::size_t>{right - 1, right};
    };
    const auto first = bracket(grid.firstAxis(), firstValue);
    const auto second = bracket(grid.secondAxis(), secondValue);
    const double firstWeight = first.first == first.second ? 0.0
        : (firstValue - grid.firstAxis()[first.first])
            / (grid.firstAxis()[first.second] - grid.firstAxis()[first.first]);
    const double secondWeight = second.first == second.second ? 0.0
        : (secondValue - grid.secondAxis()[second.first])
            / (grid.secondAxis()[second.second] - grid.secondAxis()[second.first]);
    const double lower = surface(first.first, second.first) * (1.0 - firstWeight)
        + surface(first.second, second.first) * firstWeight;
    const double upper = surface(first.first, second.second) * (1.0 - firstWeight)
        + surface(first.second, second.second) * firstWeight;
    return lower * (1.0 - secondWeight) + upper * secondWeight;
}

} // namespace

void HestonParameters::validate() const
{
    const double values[] = {spot, strike, maturity, rate, dividendYield,
                             initialVariance, longRunVariance, meanReversion,
                             volatilityOfVariance, correlation};
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("every Heston model value must be finite");
        }
    }
    if (spot < 0.0 || strike <= 0.0 || maturity < 0.0
        || initialVariance < 0.0 || longRunVariance < 0.0
        || meanReversion < 0.0 || volatilityOfVariance < 0.0
        || correlation < -1.0 || correlation > 1.0)
    {
        throw std::invalid_argument("Heston model parameters are outside their supported domains");
    }
}

bool HestonParameters::fellerConditionSatisfied() const noexcept
{
    return 2.0 * meanReversion * longRunVariance
        >= volatilityOfVariance * volatilityOfVariance;
}

void HestonReferenceSettings::validate() const
{
    if (!std::isfinite(absoluteTolerance) || absoluteTolerance <= 0.0
        || !std::isfinite(integrationLimit) || integrationLimit <= 1.0e-7
        || initialIntervals < 2 || initialIntervals % 2 != 0
        || maxRefinements == 0 || maxRefinements > 20)
    {
        throw std::invalid_argument("Heston integration tolerance controls are invalid");
    }
}

double blackScholesPrice(EuropeanOption option, double spot, double strike,
                         double maturity, double rate, double dividendYield,
                         double volatility)
{
    const double values[] = {spot, strike, maturity, rate, dividendYield, volatility};
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("Black-Scholes inputs must be finite");
        }
    }
    if (spot < 0.0 || strike <= 0.0 || maturity < 0.0 || volatility < 0.0)
    {
        throw std::invalid_argument("Black-Scholes inputs are outside their domains");
    }
    if (maturity == 0.0)
    {
        return payoff(option, spot, strike);
    }
    const double discountedSpot = spot * std::exp(-dividendYield * maturity);
    const double discountedStrike = strike * std::exp(-rate * maturity);
    if (volatility == 0.0 || spot == 0.0)
    {
        const double deterministicCall = std::max(discountedSpot - discountedStrike, 0.0);
        return option == EuropeanOption::Call ? deterministicCall
            : deterministicCall - discountedSpot + discountedStrike;
    }
    const double rootTime = std::sqrt(maturity);
    const double d1 = (std::log(spot / strike)
        + (rate - dividendYield + 0.5 * volatility * volatility) * maturity)
        / (volatility * rootTime);
    const double d2 = d1 - volatility * rootTime;
    const double call = discountedSpot * normalCdf(d1) - discountedStrike * normalCdf(d2);
    return option == EuropeanOption::Call ? call
        : call - discountedSpot + discountedStrike;
}

double hestonReferencePrice(const HestonParameters& parameters,
                            const HestonReferenceSettings& settings)
{
    parameters.validate();
    settings.validate();
    if (parameters.maturity == 0.0)
    {
        return payoff(parameters.option, parameters.spot, parameters.strike);
    }
    if (parameters.spot == 0.0)
    {
        return parameters.option == EuropeanOption::Call ? 0.0
            : parameters.strike * std::exp(-parameters.rate * parameters.maturity);
    }
    if (parameters.volatilityOfVariance <= 1.0e-10)
    {
        const double averageVariance = std::max(
            deterministicVarianceIntegral(parameters) / parameters.maturity, 0.0);
        return blackScholesPrice(parameters.option, parameters.spot, parameters.strike,
                                 parameters.maturity, parameters.rate,
                                 parameters.dividendYield, std::sqrt(averageVariance));
    }

    const std::complex<double> phiMinusI = characteristic(
        parameters, std::complex<double>(0.0, -1.0));
    if (!std::isfinite(phiMinusI.real()) || !std::isfinite(phiMinusI.imag())
        || std::abs(phiMinusI) == 0.0)
    {
        throw std::runtime_error("Heston characteristic normalization failed");
    }

    std::size_t intervals = settings.initialIntervals;
    double previous = std::numeric_limits<double>::quiet_NaN();
    double call = 0.0;
    for (std::size_t refinement = 0; refinement < settings.maxRefinements; ++refinement)
    {
        const double probability1 = simpsonProbability(parameters, settings, intervals,
                                                        true, phiMinusI);
        const double probability2 = simpsonProbability(parameters, settings, intervals,
                                                        false, phiMinusI);
        call = parameters.spot * std::exp(-parameters.dividendYield * parameters.maturity)
                * probability1
            - parameters.strike * std::exp(-parameters.rate * parameters.maturity)
                * probability2;
        if (!std::isfinite(call))
        {
            throw std::runtime_error("Heston integration produced a non-finite price");
        }
        if (refinement != 0 && std::abs(call - previous) <= settings.absoluteTolerance)
        {
            break;
        }
        previous = call;
        intervals *= 2;
    }
    const double price = parameters.option == EuropeanOption::Call ? call
        : call - parameters.spot * std::exp(-parameters.dividendYield * parameters.maturity)
            + parameters.strike * std::exp(-parameters.rate * parameters.maturity);
    if (!std::isfinite(price))
    {
        throw std::runtime_error("Heston integration returned a non-finite price");
    }
    return std::max(price, 0.0);
}

void HestonPdeSettings::validate(const HestonParameters& parameters) const
{
    const Grid2D grid(spotAxis, varianceAxis);
    (void)grid;
    if (timeSteps == 0 || !std::isfinite(theta) || theta <= 0.0 || theta > 1.0
        || rannacherSteps > timeSteps || !std::isfinite(nonnegativeTolerance)
        || nonnegativeTolerance < 0.0)
    {
        throw std::invalid_argument("Heston PDE solver controls are invalid");
    }
    if (parameters.spot < spotAxis.coordinates().front()
        || parameters.spot > spotAxis.coordinates().back()
        || parameters.initialVariance < varianceAxis.coordinates().front()
        || parameters.initialVariance > varianceAxis.coordinates().back())
    {
        throw std::invalid_argument("Heston PDE grid must contain spot and initial variance");
    }
}

SplitOperator2D makeHestonOperator(const HestonParameters& parameters,
                                   const Grid2D& grid)
{
    parameters.validate();
    SplitOperator2D result(grid);
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        const double variance = grid.secondAxis()[second];
        const auto firstVariance = stencil2d::firstWeights(grid.secondAxis(), second);
        const auto secondVariance = secondWeights(grid.secondAxis(), second);
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            const double spot = grid.firstAxis()[first];
            const auto firstSpot = stencil2d::firstWeights(grid.firstAxis(), first);
            const auto secondSpot = secondWeights(grid.firstAxis(), first);
            const std::size_t center = grid.index(first, second);
            const double spotDiffusion = 0.5 * variance * spot * spot;
            const double spotDrift = (parameters.rate - parameters.dividendYield) * spot;
            result.firstLower(center) = spotDiffusion * secondSpot[0]
                + spotDrift * firstSpot[0];
            // The reaction -r occurs exactly once, in A1.
            result.firstDiagonal(center) = spotDiffusion * secondSpot[1]
                + spotDrift * firstSpot[1] - parameters.rate;
            result.firstUpper(center) = spotDiffusion * secondSpot[2]
                + spotDrift * firstSpot[2];

            const double varianceDiffusion = 0.5
                * parameters.volatilityOfVariance * parameters.volatilityOfVariance * variance;
            const double varianceDrift = parameters.meanReversion
                * (parameters.longRunVariance - variance);
            result.secondLower(center) = varianceDiffusion * secondVariance[0]
                + varianceDrift * firstVariance[0];
            result.secondDiagonal(center) = varianceDiffusion * secondVariance[1]
                + varianceDrift * firstVariance[1];
            result.secondUpper(center) = varianceDiffusion * secondVariance[2]
                + varianceDrift * firstVariance[2];

            // At rho==0 the mixed operator remains bitwise zero.
            const double mixedScale = parameters.correlation == 0.0 ? 0.0
                : parameters.correlation * parameters.volatilityOfVariance * variance * spot;
            if (mixedScale != 0.0)
            {
                for (int secondOffset = -1; secondOffset <= 1; ++secondOffset)
                {
                    for (int firstOffset = -1; firstOffset <= 1; ++firstOffset)
                    {
                        result.explicitCoefficient(firstOffset, secondOffset, center)
                            = mixedScale * firstSpot[static_cast<std::size_t>(firstOffset + 1)]
                                * firstVariance[static_cast<std::size_t>(secondOffset + 1)];
                    }
                }
            }
        }
    }
    result.validate();
    return result;
}

HestonPdeResult hestonPdePrice(const HestonParameters& parameters,
                               const HestonPdeSettings& settings)
{
    parameters.validate();
    settings.validate(parameters);
    HestonPdeResult result;
    result.fellerConditionSatisfied = parameters.fellerConditionSatisfied();
    try
    {
        const Grid2D grid(settings.spotAxis, settings.varianceAxis);
        Surface2D<double> current(grid, 0.0);
        Surface2D<double> next(grid, 0.0);
        Surface2D<double> scratch(grid, 0.0);

        if (parameters.maturity == 0.0)
        {
            for (std::size_t second = 0; second < grid.secondSize(); ++second)
            {
                for (std::size_t first = 0; first < grid.firstSize(); ++first)
                {
                    current(first, second) = payoff(parameters.option,
                                                    grid.firstAxis()[first], parameters.strike);
                }
            }
            result.success = true;
            result.price = payoff(parameters.option, parameters.spot, parameters.strike);
            result.finalSurface = current.values();
            return result;
        }

        if (parameters.volatilityOfVariance == 0.0)
        {
            const double averageVariance = std::max(
                deterministicVarianceIntegral(parameters) / parameters.maturity, 0.0);
            for (std::size_t second = 0; second < grid.secondSize(); ++second)
            {
                for (std::size_t first = 0; first < grid.firstSize(); ++first)
                {
                    current(first, second) = blackScholesPrice(
                        parameters.option, grid.firstAxis()[first], parameters.strike,
                        parameters.maturity, parameters.rate, parameters.dividendYield,
                        std::sqrt(averageVariance));
                }
            }
            result.success = true;
            result.price = blackScholesPrice(
                parameters.option, parameters.spot, parameters.strike,
                parameters.maturity, parameters.rate, parameters.dividendYield,
                std::sqrt(averageVariance));
            result.finalSurface = current.values();
            return result;
        }

        for (std::size_t second = 0; second < grid.secondSize(); ++second)
        {
            for (std::size_t first = 0; first < grid.firstSize(); ++first)
            {
                current(first, second) = payoff(parameters.option,
                                                grid.firstAxis()[first], parameters.strike);
            }
        }

        const auto boundary = [&parameters](double time, const Grid2D& boundaryGrid,
                                             double* values)
        {
            const std::size_t lastFirst = boundaryGrid.firstSize() - 1;
            const std::size_t lastSecond = boundaryGrid.secondSize() - 1;
            // At variance zero the PDE is degenerate; linear extrapolation from
            // the first two interior rows supplies the one-sided limiting value.
            // At v_max a zero-normal-gradient closure truncates the far boundary.
            for (std::size_t first = 1; first < lastFirst; ++first)
            {
                const double v0 = boundaryGrid.secondAxis()[0];
                const double v1 = boundaryGrid.secondAxis()[1];
                const double v2 = boundaryGrid.secondAxis()[2];
                const double slope = (values[boundaryGrid.index(first, 2)]
                    - values[boundaryGrid.index(first, 1)]) / (v2 - v1);
                values[boundaryGrid.index(first, 0)]
                    = std::max(values[boundaryGrid.index(first, 1)] + slope * (v0 - v1), 0.0);
                values[boundaryGrid.index(first, lastSecond)]
                    = values[boundaryGrid.index(first, lastSecond - 1)];
            }
            const double discountedStrike = parameters.strike
                * std::exp(-parameters.rate * time);
            const double maxSpot = boundaryGrid.firstAxis()[lastFirst];
            const double discountedMaxSpot = maxSpot
                * std::exp(-parameters.dividendYield * time);
            for (std::size_t second = 0; second <= lastSecond; ++second)
            {
                if (parameters.option == EuropeanOption::Call)
                {
                    values[boundaryGrid.index(0, second)] = 0.0;
                    values[boundaryGrid.index(lastFirst, second)]
                        = std::max(discountedMaxSpot - discountedStrike, 0.0);
                }
                else
                {
                    values[boundaryGrid.index(0, second)] = discountedStrike;
                    values[boundaryGrid.index(lastFirst, second)] = 0.0;
                }
            }
        };
        boundary(0.0, grid, current.data());

        SplitOperator2D splitOperator = makeHestonOperator(parameters, grid);
        const double timeStep = parameters.maturity / static_cast<double>(settings.timeSteps);
        AdiWorkspace mcsWorkspace;
        mcsWorkspace.initialize(splitOperator, timeStep, settings.theta);
        AdiWorkspace rannacherWorkspace;
        if (settings.rannacherSteps != 0)
        {
            rannacherWorkspace.initialize(splitOperator, 0.5 * timeStep, 1.0);
        }

        double time = 0.0;
        for (std::size_t step = 0; step < settings.timeSteps; ++step)
        {
            if (step < settings.rannacherSteps)
            {
                AdiSolver::rannacherStep(splitOperator, current, scratch, next,
                                         time, boundary, rannacherWorkspace);
            }
            else
            {
                AdiSolver::step(splitOperator, current, next, time, boundary,
                                mcsWorkspace, AdiScheme::ModifiedCraigSneyd);
            }
            current.values().swap(next.values());
            time += timeStep;
        }

        const auto minimum = std::min_element(current.values().begin(), current.values().end());
        if (minimum == current.values().end() || !std::isfinite(*minimum)
            || *minimum < -settings.nonnegativeTolerance)
        {
            throw std::runtime_error("Heston PDE surface violates its nonnegative tolerance: minimum="
                                     + std::to_string(minimum == current.values().end()
                                         ? std::numeric_limits<double>::quiet_NaN() : *minimum));
        }
        for (double& value : current.values())
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("Heston PDE returned a non-finite surface");
            }
            value = std::max(value, 0.0);
        }
        result.price = bilinearValue(current, parameters.spot, parameters.initialVariance);
        if (!std::isfinite(result.price))
        {
            throw std::runtime_error("Heston PDE interpolation returned a non-finite price");
        }
        result.finalSurface = current.values();
        result.success = true;
    }
    catch (const std::exception& exception)
    {
        result.success = false;
        result.price = std::numeric_limits<double>::quiet_NaN();
        result.error = exception.what();
        result.finalSurface.clear();
    }
    return result;
}

} // namespace dr3::advanced
