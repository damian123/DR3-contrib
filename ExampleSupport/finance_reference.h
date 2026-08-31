#pragma once

#include <cmath>

namespace dr3::examples {

inline double normal_cdf(double value)
{
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

inline double black_scholes_call(double spot,
                                 double strike,
                                 double time,
                                 double rate,
                                 double volatility)
{
    const double root_time = std::sqrt(time);
    const double d1 = (std::log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * time)
        / (volatility * root_time);
    const double d2 = d1 - volatility * root_time;
    return spot * normal_cdf(d1)
        - strike * std::exp(-rate * time) * normal_cdf(d2);
}

} // namespace dr3::examples
