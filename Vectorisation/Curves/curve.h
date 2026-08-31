#pragma once

#include "interpolation.h"
#include "../VecX/operations.h"
#include "../VecX/vec_d.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dr3::numerics
{
namespace detail
{

template <typename Value>
bool curveValueIsFinite(const Value& value)
{
    return std::isfinite(static_cast<double>(value));
}

template <typename INS_VEC>
bool curveValueIsFinite(const Vec<INS_VEC>& value)
{
    if (value.isScalar()) return std::isfinite(static_cast<double>(value.getScalarValue()));
    for (int index = 0; index < value.size(); ++index)
        if (!std::isfinite(static_cast<double>(value[static_cast<std::size_t>(index)])))
            return false;
    return true;
}

template <typename INS_VEC>
bool curveValueIsFinite(const VecD<INS_VEC>& value)
{
    return curveValueIsFinite(value.value()) && curveValueIsFinite(value.derivative());
}

template <typename Value>
bool sameCurveValueShape(const Value&, const Value&)
{
    return true;
}

template <typename INS_VEC>
bool sameCurveValueShape(const Vec<INS_VEC>& lhs, const Vec<INS_VEC>& rhs)
{
    return lhs.isScalar() == rhs.isScalar()
        && (lhs.isScalar() || lhs.size() == rhs.size());
}

template <typename INS_VEC>
bool sameCurveValueShape(const VecD<INS_VEC>& lhs, const VecD<INS_VEC>& rhs)
{
    return sameCurveValueShape(lhs.value(), rhs.value())
        && sameCurveValueShape(lhs.derivative(), rhs.derivative());
}

} // namespace detail

template <typename Value>
class Curve
{
public:
    Curve(std::vector<double> pillars,
          std::vector<Value> values,
          InterpolationPolicy interpolation = InterpolationPolicy::Linear,
          ExtrapolationPolicy leftExtrapolation = ExtrapolationPolicy::Flat,
          ExtrapolationPolicy rightExtrapolation = ExtrapolationPolicy::Flat)
        : interpolation_(interpolation),
          leftExtrapolation_(leftExtrapolation),
          rightExtrapolation_(rightExtrapolation)
    {
        reset(std::move(pillars), std::move(values));
    }

    void reset(std::vector<double> pillars, std::vector<Value> values)
    {
        validate(pillars, values);
        pillars_ = std::move(pillars);
        values_ = std::move(values);
        ++revision_;
    }

    const std::vector<double>& pillars() const noexcept { return pillars_; }
    const std::vector<Value>& values() const noexcept { return values_; }
    std::size_t revision() const noexcept { return revision_; }

    Value evaluate(double query) const
    {
        if (!std::isfinite(query))
            throw std::invalid_argument("curve query must be finite");

        const auto upper = std::lower_bound(pillars_.begin(), pillars_.end(), query);
        if (upper != pillars_.end() && *upper == query)
            return values_[static_cast<std::size_t>(upper - pillars_.begin())];

        if (upper == pillars_.begin())
            return leftExtrapolation_ == ExtrapolationPolicy::Flat || pillars_.size() == 1
                ? values_.front() : interpolateSegment(0, 1, query);
        if (upper == pillars_.end())
        {
            const std::size_t final = pillars_.size() - 1;
            return rightExtrapolation_ == ExtrapolationPolicy::Flat || pillars_.size() == 1
                ? values_.back() : interpolateSegment(final - 1, final, query);
        }

        const std::size_t upperIndex = static_cast<std::size_t>(upper - pillars_.begin());
        if (interpolation_ == InterpolationPolicy::Flat)
            return values_[upperIndex - 1];
        return interpolateSegment(upperIndex - 1, upperIndex, query);
    }

    void evaluateSorted(const std::vector<double>& queries,
                        std::vector<Value>& output) const
    {
        if (output.size() < queries.size())
            throw std::invalid_argument("bulk curve output must hold every query");
        if (!std::is_sorted(queries.begin(), queries.end()))
            throw std::invalid_argument("bulk curve queries must be sorted");
        for (std::size_t index = 0; index < queries.size(); ++index)
            output[index] = evaluate(queries[index]);
    }

    std::vector<double> nodeSensitivities(double query) const
    {
        if (!std::isfinite(query))
            throw std::invalid_argument("curve query must be finite");
        std::vector<double> weights(pillars_.size(), 0.0);
        const auto upper = std::lower_bound(pillars_.begin(), pillars_.end(), query);
        if (upper != pillars_.end() && *upper == query)
        {
            weights[static_cast<std::size_t>(upper - pillars_.begin())] = 1.0;
            return weights;
        }
        if (upper == pillars_.begin())
        {
            if (leftExtrapolation_ == ExtrapolationPolicy::Flat || pillars_.size() == 1)
                weights.front() = 1.0;
            else
                setLinearWeights(weights, 0, 1, query);
            return weights;
        }
        if (upper == pillars_.end())
        {
            const std::size_t final = pillars_.size() - 1;
            if (rightExtrapolation_ == ExtrapolationPolicy::Flat || pillars_.size() == 1)
                weights.back() = 1.0;
            else
                setLinearWeights(weights, final - 1, final, query);
            return weights;
        }
        const std::size_t upperIndex = static_cast<std::size_t>(upper - pillars_.begin());
        if (interpolation_ == InterpolationPolicy::Flat)
            weights[upperIndex - 1] = 1.0;
        else
            setLinearWeights(weights, upperIndex - 1, upperIndex, query);
        return weights;
    }

private:
    static void validate(const std::vector<double>& pillars,
                         const std::vector<Value>& values)
    {
        if (pillars.empty())
            throw std::invalid_argument("a curve requires at least one pillar");
        if (pillars.size() != values.size())
            throw std::invalid_argument("curve pillar and value counts must match");
        for (std::size_t index = 0; index < pillars.size(); ++index)
        {
            if (!std::isfinite(pillars[index]) || !detail::curveValueIsFinite(values[index]))
                throw std::invalid_argument("curve pillars and values must be finite");
            if (index > 0 && pillars[index] <= pillars[index - 1])
                throw std::invalid_argument("curve pillars must be strictly increasing");
            if (index > 0 && !detail::sameCurveValueShape(values.front(), values[index]))
                throw std::invalid_argument("curve value shapes must match");
        }
    }

    Value interpolateSegment(std::size_t lower, std::size_t upper, double query) const
    {
        const double weight = (query - pillars_[lower])
            / (pillars_[upper] - pillars_[lower]);
        return linearInterpolate(values_[lower], values_[upper], weight);
    }

    void setLinearWeights(std::vector<double>& weights,
                          std::size_t lower,
                          std::size_t upper,
                          double query) const
    {
        const double weight = (query - pillars_[lower])
            / (pillars_[upper] - pillars_[lower]);
        weights[lower] = 1.0 - weight;
        weights[upper] = weight;
    }

    std::vector<double> pillars_;
    std::vector<Value> values_;
    InterpolationPolicy interpolation_;
    ExtrapolationPolicy leftExtrapolation_;
    ExtrapolationPolicy rightExtrapolation_;
    std::size_t revision_{0};
};

template <typename Value>
Value discountFactor(const Curve<Value>& zeroRateCurve, double time)
{
    if (!std::isfinite(time) || time < 0.0)
        throw std::invalid_argument("discount-factor time must be finite and non-negative");
    return exp(-zeroRateCurve.evaluate(time) * time);
}

template <typename INS_VEC>
std::vector<double> linearCurveNodeSensitivities(const std::vector<double>& pillars,
                                                 const std::vector<double>& values,
                                                 double query)
{
    Curve<double> scalarCurve(pillars, values);
    (void)scalarCurve.evaluate(query);
    const std::size_t width = static_cast<std::size_t>(INS_VEC::size());
    std::vector<double> sensitivities(values.size(), 0.0);
    for (std::size_t block = 0; block < values.size(); block += width)
    {
        std::vector<VecD<INS_VEC>> seededValues;
        seededValues.reserve(values.size());
        for (std::size_t node = 0; node < values.size(); ++node)
        {
            std::vector<typename InstructionTraits<INS_VEC>::FloatType> primal(
                width, static_cast<typename InstructionTraits<INS_VEC>::FloatType>(values[node]));
            std::vector<typename InstructionTraits<INS_VEC>::FloatType> derivative(width, 0.0);
            if (node >= block && node < block + width)
                derivative[node - block] = 1.0;
            seededValues.emplace_back(Vec<INS_VEC>(primal), Vec<INS_VEC>(derivative));
        }
        const Curve<VecD<INS_VEC>> forwardCurve(pillars, std::move(seededValues));
        const auto result = forwardCurve.evaluate(query);
        const std::size_t active = std::min(width, values.size() - block);
        for (std::size_t lane = 0; lane < active; ++lane)
            sensitivities[block + lane] = result.derivative()[lane];
    }
    return sensitivities;
}

} // namespace dr3::numerics
