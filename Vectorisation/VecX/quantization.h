#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define DR3_QUANTIZATION_HAS_AVX2 1
#else
#define DR3_QUANTIZATION_HAS_AVX2 0
#endif

namespace dr3::ai {

struct QuantizationParameters {
    float scale{1.0F};
    std::int32_t zero_point{0};
};

inline void validate_quantization_parameters(const QuantizationParameters& params,
                                             std::int32_t qmin,
                                             std::int32_t qmax)
{
    if (!(params.scale > 0.0F) || !std::isfinite(params.scale)) {
        throw std::invalid_argument("quantization scale must be finite and positive");
    }
    if (params.zero_point < qmin || params.zero_point > qmax) {
        throw std::invalid_argument("quantization zero point is outside the destination range");
    }
}

inline QuantizationParameters symmetric_s8_parameters(float max_abs)
{
    if (!std::isfinite(max_abs) || max_abs < 0.0F) {
        throw std::invalid_argument("max_abs must be finite and non-negative");
    }
    return {max_abs == 0.0F ? 1.0F : max_abs / 127.0F, 0};
}

inline QuantizationParameters asymmetric_s8_parameters(float minimum, float maximum)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
        throw std::invalid_argument("invalid quantization range");
    }
    if (minimum == maximum) {
        return {1.0F, std::clamp(static_cast<std::int32_t>(std::lround(-minimum)), -128, 127)};
    }
    const float scale = (maximum - minimum) / 255.0F;
    const auto zero = static_cast<std::int32_t>(std::lround(-128.0F - minimum / scale));
    return {scale, std::clamp(zero, -128, 127)};
}

inline QuantizationParameters asymmetric_u8_parameters(float minimum, float maximum)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
        throw std::invalid_argument("invalid quantization range");
    }
    if (minimum == maximum) {
        return {1.0F, std::clamp(static_cast<std::int32_t>(std::lround(-minimum)), 0, 255)};
    }
    const float scale = (maximum - minimum) / 255.0F;
    const auto zero = static_cast<std::int32_t>(std::lround(-minimum / scale));
    return {scale, std::clamp(zero, 0, 255)};
}

template <typename Quantized>
inline Quantized quantize_one(float value,
                              const QuantizationParameters& params,
                              std::int32_t qmin,
                              std::int32_t qmax)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument("quantize input must be finite");
    }
    // std::round specifies halfway cases away from zero. Make that choice
    // explicit instead of depending on the process floating-point mode.
    const double shifted = std::round(static_cast<double>(value) / params.scale)
        + static_cast<double>(params.zero_point);
    const double saturated = std::clamp(shifted,
                                        static_cast<double>(qmin),
                                        static_cast<double>(qmax));
    return static_cast<Quantized>(static_cast<std::int32_t>(saturated));
}

inline void quantize_s8(const float* input,
                        std::size_t size,
                        std::int8_t* output,
                        const QuantizationParameters& params)
{
    validate_quantization_parameters(params, -128, 127);
    if ((size != 0U) && (input == nullptr || output == nullptr)) {
        throw std::invalid_argument("quantize_s8 received a null buffer");
    }
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = quantize_one<std::int8_t>(input[i], params, -128, 127);
    }
}

inline std::vector<std::int8_t> quantize_s8(const std::vector<float>& input,
                                           const QuantizationParameters& params)
{
    std::vector<std::int8_t> output(input.size());
    quantize_s8(input.data(), input.size(), output.data(), params);
    return output;
}

inline void quantize_u8(const float* input,
                        std::size_t size,
                        std::uint8_t* output,
                        const QuantizationParameters& params)
{
    validate_quantization_parameters(params, 0, 255);
    if ((size != 0U) && (input == nullptr || output == nullptr)) {
        throw std::invalid_argument("quantize_u8 received a null buffer");
    }
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = quantize_one<std::uint8_t>(input[i], params, 0, 255);
    }
}

inline std::vector<std::uint8_t> quantize_u8(const std::vector<float>& input,
                                            const QuantizationParameters& params)
{
    std::vector<std::uint8_t> output(input.size());
    quantize_u8(input.data(), input.size(), output.data(), params);
    return output;
}

inline void dequantize_s8(const std::int8_t* input,
                          std::size_t size,
                          float* output,
                          const QuantizationParameters& params)
{
    validate_quantization_parameters(params, -128, 127);
    if ((size != 0U) && (input == nullptr || output == nullptr)) {
        throw std::invalid_argument("dequantize_s8 received a null buffer");
    }
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = (static_cast<std::int32_t>(input[i]) - params.zero_point) * params.scale;
    }
}

inline std::vector<float> dequantize_s8(const std::vector<std::int8_t>& input,
                                       const QuantizationParameters& params)
{
    std::vector<float> output(input.size());
    dequantize_s8(input.data(), input.size(), output.data(), params);
    return output;
}

namespace detail {

#if DR3_QUANTIZATION_HAS_AVX2
inline std::int64_t horizontal_sum_i32(__m256i value)
{
    alignas(32) std::int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), value);
    std::int64_t sum = 0;
    for (std::int32_t lane : lanes) sum += lane;
    return sum;
}
#endif

inline std::int32_t checked_i32(std::int64_t value)
{
    if (value < std::numeric_limits<std::int32_t>::min()
        || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("quantized dot product exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace detail

inline std::int32_t dot_s8_s8(const std::int8_t* lhs,
                              const std::int8_t* rhs,
                              std::size_t size)
{
    if ((size != 0U) && (lhs == nullptr || rhs == nullptr)) {
        throw std::invalid_argument("dot_s8_s8 received a null buffer");
    }
    std::int64_t sum = 0;
    std::size_t i = 0;
#if DR3_QUANTIZATION_HAS_AVX2
    // Widen 16 signed bytes from each operand, then pairwise multiply/add.
    for (; i + 16U <= size; i += 16U) {
        const __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs + i));
        const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs + i));
        const __m256i a16 = _mm256_cvtepi8_epi16(a8);
        const __m256i b16 = _mm256_cvtepi8_epi16(b8);
        sum += detail::horizontal_sum_i32(_mm256_madd_epi16(a16, b16));
    }
#endif
    for (; i < size; ++i) {
        sum += static_cast<std::int32_t>(lhs[i]) * static_cast<std::int32_t>(rhs[i]);
    }
    return detail::checked_i32(sum);
}

inline std::int32_t dot_s8_s8(const std::vector<std::int8_t>& lhs,
                              const std::vector<std::int8_t>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::invalid_argument("dot vectors must have equal lengths");
    return dot_s8_s8(lhs.data(), rhs.data(), lhs.size());
}

inline std::int32_t dot_u8_s8(const std::uint8_t* lhs,
                              const std::int8_t* rhs,
                              std::size_t size)
{
    if ((size != 0U) && (lhs == nullptr || rhs == nullptr)) {
        throw std::invalid_argument("dot_u8_s8 received a null buffer");
    }
    std::int64_t sum = 0;
    std::size_t i = 0;
#if DR3_QUANTIZATION_HAS_AVX2
    for (; i + 16U <= size; i += 16U) {
        const __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lhs + i));
        const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rhs + i));
        const __m256i a16 = _mm256_cvtepu8_epi16(a8);
        const __m256i b16 = _mm256_cvtepi8_epi16(b8);
        sum += detail::horizontal_sum_i32(_mm256_madd_epi16(a16, b16));
    }
#endif
    for (; i < size; ++i) {
        sum += static_cast<std::int32_t>(lhs[i]) * static_cast<std::int32_t>(rhs[i]);
    }
    return detail::checked_i32(sum);
}

inline std::int32_t dot_u8_s8(const std::vector<std::uint8_t>& lhs,
                              const std::vector<std::int8_t>& rhs)
{
    if (lhs.size() != rhs.size()) throw std::invalid_argument("dot vectors must have equal lengths");
    return dot_u8_s8(lhs.data(), rhs.data(), lhs.size());
}

inline float dequantized_dot(const std::int8_t* lhs,
                             const QuantizationParameters& lhs_params,
                             const std::int8_t* rhs,
                             const QuantizationParameters& rhs_params,
                             std::size_t size)
{
    validate_quantization_parameters(lhs_params, -128, 127);
    validate_quantization_parameters(rhs_params, -128, 127);
    if ((size != 0U) && (lhs == nullptr || rhs == nullptr)) {
        throw std::invalid_argument("dequantized_dot received a null buffer");
    }
    std::int64_t corrected = 0;
    for (std::size_t i = 0; i < size; ++i) {
        corrected += (static_cast<std::int32_t>(lhs[i]) - lhs_params.zero_point)
            * (static_cast<std::int32_t>(rhs[i]) - rhs_params.zero_point);
    }
    detail::checked_i32(corrected);
    return static_cast<float>(corrected)
        * lhs_params.scale * rhs_params.scale;
}

inline float dequantized_dot(const std::uint8_t* lhs,
                             const QuantizationParameters& lhs_params,
                             const std::int8_t* rhs,
                             const QuantizationParameters& rhs_params,
                             std::size_t size)
{
    validate_quantization_parameters(lhs_params, 0, 255);
    validate_quantization_parameters(rhs_params, -128, 127);
    if ((size != 0U) && (lhs == nullptr || rhs == nullptr)) {
        throw std::invalid_argument("dequantized_dot received a null buffer");
    }
    std::int64_t corrected = 0;
    for (std::size_t i = 0; i < size; ++i) {
        corrected += (static_cast<std::int32_t>(lhs[i]) - lhs_params.zero_point)
            * (static_cast<std::int32_t>(rhs[i]) - rhs_params.zero_point);
    }
    detail::checked_i32(corrected);
    return static_cast<float>(corrected)
        * lhs_params.scale * rhs_params.scale;
}

} // namespace dr3::ai
