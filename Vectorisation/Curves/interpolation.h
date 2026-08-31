#pragma once

namespace dr3::numerics
{

enum class InterpolationPolicy
{
    Linear,
    Flat
};

enum class ExtrapolationPolicy
{
    Linear,
    Flat
};

template <typename Value>
Value linearInterpolate(const Value& lower, const Value& upper, double upperWeight)
{
    return lower * (1.0 - upperWeight) + upper * upperWeight;
}

} // namespace dr3::numerics
