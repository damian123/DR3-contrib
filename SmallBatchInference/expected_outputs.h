#pragma once

#include "model_weights.h"

#include <array>

namespace dr3_mlp {
namespace fixture {

// Generated once from the scalar-double reference in SmallBatchInference.cpp.
// Values are probabilities in row-major batch order.
constexpr std::array<float, kBatch * kOutput> kExpectedOutputs = {{
    0.256683022f, 0.347429007f, 0.395887971f,
    0.378753781f, 0.335887671f, 0.285358548f,
}};

} // namespace fixture
} // namespace dr3_mlp
