#pragma once

#include <algorithm>
#include <cmath>

namespace dr3::test
{

template <typename Function>
double centralDifference(Function&& function, double input,
                         double relativeBump = 1.0e-6)
{
    const double bump = relativeBump * std::max(1.0, std::abs(input));
    return (function(input + bump) - function(input - bump)) / (2.0 * bump);
}

} // namespace dr3::test
