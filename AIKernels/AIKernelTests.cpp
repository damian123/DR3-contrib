#include "VecX/kmeans.h"
#include "VecX/preprocessing.h"
#include "VecX/preprocessing_dr3.h"
#include "VecX/quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_near(double actual, double expected, double tolerance, const char* message)
{
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const char* message)
{
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_quantization()
{
    using namespace dr3::ai;
    const QuantizationParameters symmetric{0.5F, 0};
    const std::vector<float> values{-1000.0F, -64.0F, -1.25F, -0.25F,
                                    0.0F, 0.25F, 1.25F, 63.5F, 1000.0F};
    const auto quantized = quantize_s8(values, symmetric);
    require(quantized.front() == std::numeric_limits<std::int8_t>::min(),
            "s8 lower saturation failed");
    require(quantized.back() == std::numeric_limits<std::int8_t>::max(),
            "s8 upper saturation failed");
    require(quantized[2] == -3 && quantized[3] == -1,
            "negative halfway rounding must be away from zero");
    require(quantized[4] == 0 && quantized[5] == 1 && quantized[6] == 3,
            "positive halfway rounding or exact zero failed");

    const QuantizationParameters signed_asymmetric{0.25F, -7};
    const auto signed_zero = quantize_s8(std::vector<float>{-0.25F, 0.0F, 0.25F},
                                        signed_asymmetric);
    require(signed_zero[0] == -8 && signed_zero[1] == -7 && signed_zero[2] == -6,
            "signed zero point mapping failed");
    const QuantizationParameters unsigned_asymmetric{0.25F, 123};
    const auto unsigned_zero = quantize_u8(std::vector<float>{-0.25F, 0.0F, 0.25F},
                                          unsigned_asymmetric);
    require(unsigned_zero[0] == 122U && unsigned_zero[1] == 123U && unsigned_zero[2] == 124U,
            "unsigned zero point mapping failed");

    const auto restored = dequantize_s8(quantized, symmetric);
    double max_error = 0.0;
    const std::vector<float> in_range{-3.1F, -1.2F, 0.0F, 2.2F, 9.9F, 12.3F, 15.7F,
                                      18.2F, 20.1F}; // tail beyond one AVX2 register
    const auto in_range_q = quantize_s8(in_range, symmetric);
    const auto in_range_dq = dequantize_s8(in_range_q, symmetric);
    for (std::size_t i = 0; i < in_range.size(); ++i) {
        max_error = std::max(max_error,
            std::fabs(static_cast<double>(in_range[i]) - in_range_dq[i]));
    }
    require(max_error <= symmetric.scale * 0.5F + 1.0e-6F,
            "quantize/dequantize error exceeded half a step");
    require(restored.front() == -64.0F && restored.back() == 63.5F,
            "dequantization of saturated endpoints failed");
    require(symmetric_s8_parameters(12.7F).zero_point == 0,
            "symmetric parameter zero point failed");
    require(asymmetric_s8_parameters(-2.0F, 3.0F).scale > 0.0F,
            "signed asymmetric parameter generation failed");
    require(asymmetric_u8_parameters(-2.0F, 3.0F).scale > 0.0F,
            "unsigned asymmetric parameter generation failed");
    require_throws<std::invalid_argument>(
        [] { quantize_s8(std::vector<float>{1.0F}, {0.0F, 0}); },
        "zero scale must be rejected");

    std::cout << "quantize-dequantize max_abs_error=" << max_error
              << " scale=" << symmetric.scale << '\n';
}

void test_quantized_dot()
{
    using namespace dr3::ai;
    const std::vector<std::int8_t> lhs{-128, -17, 0, 1, 12, 127, 3, -8, 4};
    const std::vector<std::int8_t> rhs{127, 9, -4, 1, -12, 126, 7, -2, 11};
    std::int64_t scalar = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) scalar += lhs[i] * rhs[i];
    require(dot_s8_s8(lhs, rhs) == scalar, "signed quantized dot mismatch");

    const std::vector<std::uint8_t> unsigned_lhs{0, 1, 127, 128, 255, 4, 9, 12, 44};
    scalar = 0;
    for (std::size_t i = 0; i < rhs.size(); ++i) scalar += unsigned_lhs[i] * rhs[i];
    require(dot_u8_s8(unsigned_lhs, rhs) == scalar, "mixed quantized dot mismatch");

    const QuantizationParameters lhs_params{0.25F, -3};
    const QuantizationParameters rhs_params{0.125F, 5};
    std::int64_t corrected = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        corrected += (static_cast<int>(lhs[i]) - lhs_params.zero_point)
            * (static_cast<int>(rhs[i]) - rhs_params.zero_point);
    }
    require_near(dequantized_dot(lhs.data(), lhs_params, rhs.data(), rhs_params, lhs.size()),
                 static_cast<double>(corrected) * lhs_params.scale * rhs_params.scale,
                 1.0e-5,
                 "dequantized dot scale/zero-point math failed");

    const std::size_t safe_count = static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max() / (127 * 127));
    std::vector<std::int8_t> near_limit_lhs(safe_count, 127);
    std::vector<std::int8_t> near_limit_rhs(safe_count, 127);
    const auto near_limit = dot_s8_s8(near_limit_lhs, near_limit_rhs);
    require(near_limit > 2'000'000'000, "near-int32-limit dot test was not near the limit");
    near_limit_lhs.push_back(127);
    near_limit_rhs.push_back(127);
    require_throws<std::overflow_error>(
        [&] { (void)dot_s8_s8(near_limit_lhs, near_limit_rhs); },
        "overflowing quantized dot must be rejected");

    const std::vector<float> float_lhs{-1.0F, 0.4F, 2.0F, 1.2F, -0.2F};
    const std::vector<float> float_rhs{0.2F, 1.1F, -0.8F, 0.3F, 2.0F};
    const QuantizationParameters qparams{0.02F, 0};
    const auto qlhs = quantize_s8(float_lhs, qparams);
    const auto qrhs = quantize_s8(float_rhs, qparams);
    const float quantized_model_output = dequantized_dot(
        qlhs.data(), qparams, qrhs.data(), qparams, qlhs.size());
    const double float_model_output = std::inner_product(
        float_lhs.begin(), float_lhs.end(), float_rhs.begin(), 0.0);
    const double model_error = std::fabs(quantized_model_output - float_model_output);
    require(model_error < 0.05, "quantized model output error is excessive");
    std::cout << "float_vs_quantized_output_abs_error=" << model_error << '\n';
}

void test_preprocess()
{
    using namespace dr3::ai;
    const std::vector<float> tail_values{-2.0F, -1.0F, 0.0F, 1.0F, 2.0F,
                                          3.0F, 4.0F, 5.0F, 6.0F};
    const double scalar_mean = std::accumulate(tail_values.begin(), tail_values.end(), 0.0)
        / static_cast<double>(tail_values.size());
    double scalar_variance = 0.0;
    for (float value : tail_values) scalar_variance += (value - scalar_mean) * (value - scalar_mean);
    scalar_variance /= tail_values.size();
    require_near(feature_mean(tail_values), scalar_mean, 1.0e-12,
                 "SIMD/tail mean differs from scalar double reference");
    require_near(feature_variance(tail_values), scalar_variance, 1.0e-12,
                 "variance differs from scalar double reference");
    require_near(feature_stddev(tail_values), std::sqrt(scalar_variance), 1.0e-12,
                 "stddev differs from scalar double reference");

    const auto standardized = z_score(tail_values);
    require_near(feature_mean(standardized), 0.0, 1.0e-7, "z-score mean is not zero");
    require_near(feature_variance(standardized), 1.0, 1.0e-6, "z-score variance is not one");
    const auto scaled = min_max_scale(tail_values, -1.0F, 1.0F);
    require_near(scaled.front(), -1.0, 0.0, "min-max lower endpoint failed");
    require_near(scaled.back(), 1.0, 0.0, "min-max upper endpoint failed");
    const auto clipped_values = clipped(tail_values, -1.0F, 2.0F);
    require(clipped_values.front() == -1.0F && clipped_values.back() == 2.0F,
            "feature clipping failed");

    std::vector<float> missing{1.0F, std::numeric_limits<float>::quiet_NaN(), 3.0F};
    const auto mask = missing_value_mask(missing);
    require(mask == std::vector<std::uint8_t>({0U, 1U, 0U}), "missing-value mask failed");
    impute_constant_inplace(missing.data(), missing.size(), 2.0F);
    require(missing == std::vector<float>({1.0F, 2.0F, 3.0F}), "constant imputation failed");

    const std::vector<float> batch{1.0F, 5.0F, 2.0F,
                                   2.0F, 5.0F, 4.0F,
                                   3.0F, 5.0F, 6.0F};
    const auto batch_result = standardize_columns(batch.data(), 3U, 3U);
    require(batch_result.values[1] == 0.0F
            && batch_result.values[4] == 0.0F
            && batch_result.values[7] == 0.0F,
            "constant batch column must standardize to zero");

    RunningStatistics running;
    RollingStatistics rolling(3U);
    for (double value : {1.0, 2.0, 3.0, 4.0}) {
        running.push(value);
        rolling.push(value);
    }
    require_near(running.mean(), 2.5, 1.0e-15, "running mean failed");
    require_near(running.variance(), 1.25, 1.0e-15, "running variance failed");
    require_near(rolling.mean(), 3.0, 1.0e-15, "rolling mean failed");
    require_near(rolling.variance(), 2.0 / 3.0, 1.0e-15, "rolling variance failed");

    const std::vector<float> outlier_fixture{0.0F, 0.0F, 0.0F, 0.0F, 10.0F};
    const auto scores = absolute_z_outlier_scores(outlier_fixture);
    const auto outliers = threshold_filter(scores, 1.5F);
    require(outliers == std::vector<std::size_t>{4U}, "outlier threshold filter failed");

    const std::vector<float> dr3_fixture{0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                         0.0F, 0.0F, 0.0F, 20.0F};
    DRC::VecF8F::VecXX dr3_values(dr3_fixture);
    const auto dr3_scores = absolute_z_outlier_scores_dr3(dr3_values);
    const auto dr3_outliers = threshold_filter_dr3(dr3_scores, 2.0F);
    require(dr3_outliers.size() == 1U && dr3_outliers[0] > 2.0F,
            "DR3 reduce/filter outlier path failed");

    for (std::size_t dimensions : {1U, 7U, 8U, 9U, 17U}) {
        const std::vector<float> constant(dimensions, 42.0F);
        const auto zeros = z_score(constant);
        require(std::all_of(zeros.begin(), zeros.end(), [](float value) { return value == 0.0F; }),
                "constant/tail z-score must be zero");
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        AllAllocatorsGuard<float> allocator_guard;
        const std::string mode = argc > 1 ? argv[1] : "--all";
        if (mode == "--all" || mode == "--quantization") test_quantization();
        if (mode == "--all" || mode == "--quantized-dot") test_quantized_dot();
        if (mode == "--all" || mode == "--preprocess") test_preprocess();
        if (mode != "--all" && mode != "--quantization"
            && mode != "--quantized-dot" && mode != "--preprocess") {
            throw std::invalid_argument("unknown test mode");
        }
        std::cout << "AI kernel tests passed (" << mode << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "AI kernel test failure: " << error.what() << '\n';
        return 1;
    }
}
