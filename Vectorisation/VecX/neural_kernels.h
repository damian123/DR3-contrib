/************************ neural_kernels.h **********************************
 * Stable float32 preprocessing kernels for small CPU inference workloads.
 * AVX2 (VecF8F) is the default tested path; callers own all storage.
 *****************************************************************************/
#pragma once

#include "dr3.h"
#include "target_name_space.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace DRC {
namespace AI {

namespace NeuralSimd = VecF8F;

enum class Activation {
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu
};

inline void require_buffer(const float* p, std::size_t n, const char* what)
{
    if (p == nullptr && n != 0) {
        throw std::invalid_argument(what);
    }
}

inline NeuralSimd::VecXX load_neural_vec(const float* p, std::size_t n)
{
    require_buffer(p, n, "null input buffer");
    if (n == 0) {
        return NeuralSimd::VecXX();
    }
    return NeuralSimd::VecXX(std::vector<float>(p, p + n));
}

inline void store_neural_vec(const NeuralSimd::VecXX& v, float* out, std::size_t n)
{
    require_buffer(out, n, "null output buffer");
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = v[static_cast<int>(i)];
    }
}

inline NeuralSimd::VecXX activate_vec(
    const NeuralSimd::VecXX& x, Activation activation, float leaky_alpha = 0.01f)
{
    switch (activation) {
    case Activation::Relu:
        return max(x, 0.f);
    case Activation::LeakyRelu:
        return max(x, 0.f) + leaky_alpha * min(x, 0.f);
    case Activation::Sigmoid:
        return 1.f / (1.f + exp(-x));
    case Activation::Tanh: {
        auto op = [](auto z) { return tanh(z); };
        return transform(op, x);
    }
    case Activation::Gelu: {
        // Hendrycks/Gimpel tanh approximation used by many inference runtimes.
        auto op = [](auto z) {
            using Value = decltype(z);
            const Value inner = Value(0.7978845608028654f) *
                (z + Value(0.044715f) * z * z * z);
            return Value(0.5f) * z * (Value(1.f) + tanh(inner));
        };
        return transform(op, x);
    }
    case Activation::Silu:
        return x / (1.f + exp(-x));
    }
    throw std::invalid_argument("unknown activation");
}

inline void activation(
    const float* input, std::size_t n, float* output,
    Activation kind, float leaky_alpha = 0.01f)
{
    require_buffer(output, n, "activation: null output");
    if (n == 0) {
        return;
    }
    const auto x = load_neural_vec(input, n);
    store_neural_vec(activate_vec(x, kind, leaky_alpha), output, n);
}

inline void activation_inplace(
    float* values, std::size_t n, Activation kind, float leaky_alpha = 0.01f)
{
    activation(values, n, values, kind, leaky_alpha);
}

inline void relu(const float* x, std::size_t n, float* out)
{
    activation(x, n, out, Activation::Relu);
}

inline void leaky_relu(const float* x, std::size_t n, float* out, float alpha = 0.01f)
{
    if (alpha < 0.f) {
        throw std::invalid_argument("leaky_relu: alpha must be non-negative");
    }
    activation(x, n, out, Activation::LeakyRelu, alpha);
}

inline void sigmoid(const float* x, std::size_t n, float* out)
{
    activation(x, n, out, Activation::Sigmoid);
}

inline void tanh_activation(const float* x, std::size_t n, float* out)
{
    activation(x, n, out, Activation::Tanh);
}

inline void gelu(const float* x, std::size_t n, float* out)
{
    activation(x, n, out, Activation::Gelu);
}

inline void silu(const float* x, std::size_t n, float* out)
{
    activation(x, n, out, Activation::Silu);
}

inline void clip(const float* x, std::size_t n, float lo, float hi, float* out)
{
    if (lo > hi) {
        throw std::invalid_argument("clip: lower bound exceeds upper bound");
    }
    require_buffer(out, n, "clip: null output");
    if (n == 0) {
        return;
    }
    store_neural_vec(max(min(load_neural_vec(x, n), hi), lo), out, n);
}

inline void clip_inplace(float* x, std::size_t n, float lo, float hi)
{
    clip(x, n, lo, hi, x);
}

inline void bias_plus_activation(
    const float* input, const float* bias, std::size_t n, float* output,
    Activation kind, float leaky_alpha = 0.01f)
{
    require_buffer(bias, n, "bias_plus_activation: null bias");
    require_buffer(output, n, "bias_plus_activation: null output");
    if (n == 0) {
        return;
    }
    const auto z = load_neural_vec(input, n) + load_neural_vec(bias, n);
    store_neural_vec(activate_vec(z, kind, leaky_alpha), output, n);
}

inline float log_sum_exp(const float* input, std::size_t n)
{
    require_buffer(input, n, "log_sum_exp: null input");
    if (n == 0) {
        return -std::numeric_limits<float>::infinity();
    }
    const float xmax = *std::max_element(input, input + n);
    const auto shifted_exp = exp(load_neural_vec(input, n) - xmax);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(shifted_exp[static_cast<int>(i)]);
    }
    return xmax + static_cast<float>(std::log(sum));
}

inline void softmax(const float* input, std::size_t n, float* output)
{
    require_buffer(input, n, "softmax: null input");
    require_buffer(output, n, "softmax: null output");
    if (n == 0) {
        return;
    }
    const float xmax = *std::max_element(input, input + n);
    const auto numerators = exp(load_neural_vec(input, n) - xmax);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(numerators[static_cast<int>(i)]);
    }
    store_neural_vec(numerators / static_cast<float>(sum), output, n);
}

inline void softmax_inplace(float* values, std::size_t n)
{
    softmax(values, n, values);
}

inline void log_softmax(const float* input, std::size_t n, float* output)
{
    require_buffer(input, n, "log_softmax: null input");
    require_buffer(output, n, "log_softmax: null output");
    if (n == 0) {
        return;
    }
    const float lse = log_sum_exp(input, n);
    store_neural_vec(load_neural_vec(input, n) - lse, output, n);
}

inline void log_softmax_inplace(float* values, std::size_t n)
{
    log_softmax(values, n, values);
}

inline void layer_norm(
    const float* input, std::size_t n, float epsilon, float* output,
    const float* scale = nullptr, const float* bias = nullptr)
{
    require_buffer(input, n, "layer_norm: null input");
    require_buffer(output, n, "layer_norm: null output");
    if (!(epsilon > 0.f)) {
        throw std::invalid_argument("layer_norm: epsilon must be positive");
    }
    if (n == 0) {
        return;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += input[i];
    }
    const double mean = sum / static_cast<double>(n);
    double squared = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(input[i]) - mean;
        squared += d * d;
    }
    const float inv_std = static_cast<float>(1.0 /
        std::sqrt(squared / static_cast<double>(n) + epsilon));
    auto normalized = (load_neural_vec(input, n) - static_cast<float>(mean)) * inv_std;
    if (scale != nullptr) {
        normalized *= load_neural_vec(scale, n);
    }
    if (bias != nullptr) {
        normalized += load_neural_vec(bias, n);
    }
    store_neural_vec(normalized, output, n);
}

inline void layer_norm_inplace(
    float* values, std::size_t n, float epsilon,
    const float* scale = nullptr, const float* bias = nullptr)
{
    layer_norm(values, n, epsilon, values, scale, bias);
}

inline void rms_norm(
    const float* input, std::size_t n, float epsilon, float* output,
    const float* scale = nullptr)
{
    require_buffer(input, n, "rms_norm: null input");
    require_buffer(output, n, "rms_norm: null output");
    if (!(epsilon > 0.f)) {
        throw std::invalid_argument("rms_norm: epsilon must be positive");
    }
    if (n == 0) {
        return;
    }
    double squared = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = input[i];
        squared += x * x;
    }
    const float inv_rms = static_cast<float>(1.0 /
        std::sqrt(squared / static_cast<double>(n) + epsilon));
    auto normalized = load_neural_vec(input, n) * inv_rms;
    if (scale != nullptr) {
        normalized *= load_neural_vec(scale, n);
    }
    store_neural_vec(normalized, output, n);
}

inline void rms_norm_inplace(
    float* values, std::size_t n, float epsilon, const float* scale = nullptr)
{
    rms_norm(values, n, epsilon, values, scale);
}

} // namespace AI
} // namespace DRC
