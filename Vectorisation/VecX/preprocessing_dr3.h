#pragma once

#include "dr3.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace dr3::ai {

// DR3-native overloads keep the feature pipeline on the library's existing
// transform/reduce/filter facilities when the caller already owns a Vec.
template <typename INS_VEC>
Vec<INS_VEC> absolute_z_outlier_scores_dr3(const Vec<INS_VEC>& values,
                                           typename InstructionTraits<INS_VEC>::FloatType epsilon =
                                               static_cast<typename InstructionTraits<INS_VEC>::FloatType>(1.0e-12))
{
    using Float = typename InstructionTraits<INS_VEC>::FloatType;
    if (values.size() == 0U) throw std::invalid_argument("feature vector must not be empty");
    auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
    const Float mean = reduce(values, add) / static_cast<Float>(values.size());
    auto squared_delta = [mean](auto value) {
        const auto delta = value - INS_VEC(mean);
        return delta * delta;
    };
    const Vec<INS_VEC> squared = transform(squared_delta, values);
    const Float deviation = static_cast<Float>(std::sqrt(
        reduce(squared, add) / static_cast<Float>(values.size())));
    if (deviation <= epsilon) return Vec<INS_VEC>(Float{0}, values.size());
    auto score = [mean, deviation](auto value) {
        return abs((value - INS_VEC(mean)) / INS_VEC(deviation));
    };
    return transform(score, values);
}

template <typename INS_VEC>
std::vector<typename InstructionTraits<INS_VEC>::FloatType> threshold_filter_dr3(
    const Vec<INS_VEC>& scores,
    typename InstructionTraits<INS_VEC>::FloatType threshold)
{
    auto above_threshold = [threshold](auto value) { return value >= INS_VEC(threshold); };
    auto selected = filter(above_threshold, scores);
    return static_cast<std::vector<typename InstructionTraits<INS_VEC>::FloatType>>(selected);
}

} // namespace dr3::ai
