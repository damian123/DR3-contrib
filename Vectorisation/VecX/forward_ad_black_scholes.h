#pragma once

#include "operations.h"
#include "vec_d.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace dr3::numerics
{

template <typename INS_VEC>
void validateBlackScholesInput(const VecD<INS_VEC>& value,
                               const char* name,
                               bool mustBePositive)
{
    const auto& primal = value.value();
    const auto validate = [name, mustBePositive](double element)
    {
        if (!std::isfinite(element) || (mustBePositive && element <= 0.0))
        {
            throw std::invalid_argument(std::string("invalid Black-Scholes ") + name);
        }
    };
    if (primal.isScalar())
    {
        validate(static_cast<double>(primal.getScalarValue()));
        return;
    }
    for (int index = 0; index < primal.size(); ++index)
    {
        validate(static_cast<double>(primal[static_cast<size_t>(index)]));
    }
}

template <typename INS_VEC>
VecD<INS_VEC> blackScholesCallForwardAd(const VecD<INS_VEC>& spot,
                                        const VecD<INS_VEC>& strike,
                                        const VecD<INS_VEC>& volatility,
                                        const VecD<INS_VEC>& rate,
                                        const VecD<INS_VEC>& dividendYield,
                                        const VecD<INS_VEC>& maturity)
{
    validateBlackScholesInput(spot, "spot", true);
    validateBlackScholesInput(strike, "strike", true);
    validateBlackScholesInput(volatility, "volatility", true);
    validateBlackScholesInput(rate, "rate", false);
    validateBlackScholesInput(dividendYield, "dividend yield", false);
    validateBlackScholesInput(maturity, "maturity", true);

    const auto rootTime = sqrt(maturity);
    const auto variance = volatility * volatility;
    const auto d1 = (log(spot / strike)
                     + (rate - dividendYield + 0.5 * variance) * maturity)
        / (volatility * rootTime);
    const auto d2 = d1 - volatility * rootTime;
    return spot * exp(-dividendYield * maturity) * cdfnorm(d1)
        - strike * exp(-rate * maturity) * cdfnorm(d2);
}

} // namespace dr3::numerics
