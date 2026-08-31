#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define DR3_PREPROCESS_HAS_AVX2 1
#else
#define DR3_PREPROCESS_HAS_AVX2 0
#endif

namespace dr3::ai {

inline void validate_feature_buffer(const float* values, std::size_t size)
{
    if (size == 0U) throw std::invalid_argument("feature vector must not be empty");
    if (values == nullptr) throw std::invalid_argument("feature buffer must not be null");
}

inline double feature_mean(const float* values, std::size_t size)
{
    validate_feature_buffer(values, size);
    double sum = 0.0;
    std::size_t i = 0;
#if DR3_PREPROCESS_HAS_AVX2
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    for (; i + 8U <= size; i += 8U) {
        const __m256 x = _mm256_loadu_ps(values + i);
        sum0 = _mm256_add_pd(sum0, _mm256_cvtps_pd(_mm256_castps256_ps128(x)));
        sum1 = _mm256_add_pd(sum1, _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)));
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, _mm256_add_pd(sum0, sum1));
    sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#endif
    for (; i < size; ++i) sum += values[i];
    return sum / static_cast<double>(size);
}

inline double feature_mean(const std::vector<float>& values)
{
    return feature_mean(values.data(), values.size());
}

inline double feature_variance(const float* values,
                               std::size_t size,
                               bool sample = false)
{
    validate_feature_buffer(values, size);
    if (sample && size < 2U) throw std::invalid_argument("sample variance needs two values");
    const double mean = feature_mean(values, size);
    double squared = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double delta = static_cast<double>(values[i]) - mean;
        squared += delta * delta;
    }
    return squared / static_cast<double>(sample ? size - 1U : size);
}

inline double feature_variance(const std::vector<float>& values, bool sample = false)
{
    return feature_variance(values.data(), values.size(), sample);
}

inline double feature_stddev(const float* values,
                             std::size_t size,
                             bool sample = false)
{
    return std::sqrt(feature_variance(values, size, sample));
}

inline double feature_stddev(const std::vector<float>& values, bool sample = false)
{
    return feature_stddev(values.data(), values.size(), sample);
}

inline std::vector<float> z_score(const float* values,
                                  std::size_t size,
                                  float epsilon = 1.0e-12F)
{
    validate_feature_buffer(values, size);
    if (!(epsilon >= 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("epsilon must be finite and non-negative");
    }
    const double mean = feature_mean(values, size);
    const double deviation = feature_stddev(values, size);
    std::vector<float> output(size, 0.0F);
    if (deviation <= static_cast<double>(epsilon)) return output;
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = static_cast<float>((static_cast<double>(values[i]) - mean) / deviation);
    }
    return output;
}

inline std::vector<float> z_score(const std::vector<float>& values,
                                  float epsilon = 1.0e-12F)
{
    return z_score(values.data(), values.size(), epsilon);
}

inline std::vector<float> min_max_scale(const float* values,
                                        std::size_t size,
                                        float output_min = 0.0F,
                                        float output_max = 1.0F)
{
    validate_feature_buffer(values, size);
    if (!std::isfinite(output_min) || !std::isfinite(output_max) || output_min > output_max) {
        throw std::invalid_argument("invalid output range");
    }
    const auto bounds = std::minmax_element(values, values + size);
    std::vector<float> output(size, output_min);
    if (*bounds.first == *bounds.second) return output;
    const double scale = static_cast<double>(output_max - output_min)
        / static_cast<double>(*bounds.second - *bounds.first);
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = static_cast<float>(output_min
            + (static_cast<double>(values[i]) - *bounds.first) * scale);
    }
    return output;
}

inline std::vector<float> min_max_scale(const std::vector<float>& values,
                                        float output_min = 0.0F,
                                        float output_max = 1.0F)
{
    return min_max_scale(values.data(), values.size(), output_min, output_max);
}

inline void clip_inplace(float* values, std::size_t size, float minimum, float maximum)
{
    if ((size != 0U) && values == nullptr) throw std::invalid_argument("clip buffer is null");
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
        throw std::invalid_argument("invalid clip range");
    }
    for (std::size_t i = 0; i < size; ++i) values[i] = std::clamp(values[i], minimum, maximum);
}

inline std::vector<float> clipped(const std::vector<float>& values,
                                  float minimum,
                                  float maximum)
{
    std::vector<float> output(values);
    clip_inplace(output.data(), output.size(), minimum, maximum);
    return output;
}

inline std::vector<std::uint8_t> missing_value_mask(const float* values, std::size_t size)
{
    if ((size != 0U) && values == nullptr) throw std::invalid_argument("mask buffer is null");
    std::vector<std::uint8_t> mask(size);
    for (std::size_t i = 0; i < size; ++i) mask[i] = std::isnan(values[i]) ? 1U : 0U;
    return mask;
}

inline std::vector<std::uint8_t> missing_value_mask(const std::vector<float>& values)
{
    return missing_value_mask(values.data(), values.size());
}

inline void impute_constant_inplace(float* values,
                                    std::size_t size,
                                    float replacement)
{
    if ((size != 0U) && values == nullptr) throw std::invalid_argument("impute buffer is null");
    if (!std::isfinite(replacement)) throw std::invalid_argument("replacement must be finite");
    for (std::size_t i = 0; i < size; ++i) {
        if (std::isnan(values[i])) values[i] = replacement;
    }
}

struct BatchStandardizationResult {
    std::vector<float> values;
    std::vector<double> means;
    std::vector<double> standard_deviations;
};

inline BatchStandardizationResult standardize_columns(const float* row_major,
                                                       std::size_t rows,
                                                       std::size_t columns,
                                                       float epsilon = 1.0e-12F)
{
    if (rows == 0U || columns == 0U) throw std::invalid_argument("batch dimensions must be non-zero");
    if (row_major == nullptr) throw std::invalid_argument("batch buffer is null");
    BatchStandardizationResult result;
    result.values.assign(row_major, row_major + rows * columns);
    result.means.assign(columns, 0.0);
    result.standard_deviations.assign(columns, 0.0);
    for (std::size_t column = 0; column < columns; ++column) {
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) sum += row_major[row * columns + column];
        const double mean = sum / static_cast<double>(rows);
        double squared = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double delta = static_cast<double>(row_major[row * columns + column]) - mean;
            squared += delta * delta;
        }
        const double deviation = std::sqrt(squared / static_cast<double>(rows));
        result.means[column] = mean;
        result.standard_deviations[column] = deviation;
        for (std::size_t row = 0; row < rows; ++row) {
            result.values[row * columns + column] = deviation <= epsilon
                ? 0.0F
                : static_cast<float>((row_major[row * columns + column] - mean) / deviation);
        }
    }
    return result;
}

class RunningStatistics {
public:
    void push(double value)
    {
        if (!std::isfinite(value)) throw std::invalid_argument("running statistic value must be finite");
        ++count_;
        const double delta = value - mean_;
        mean_ += delta / static_cast<double>(count_);
        m2_ += delta * (value - mean_);
        minimum_ = std::min(minimum_, value);
        maximum_ = std::max(maximum_, value);
    }

    std::size_t count() const noexcept { return count_; }
    double mean() const { require_values(); return mean_; }
    double variance() const { require_values(); return m2_ / static_cast<double>(count_); }
    double stddev() const { return std::sqrt(variance()); }
    double minimum() const { require_values(); return minimum_; }
    double maximum() const { require_values(); return maximum_; }

private:
    void require_values() const
    {
        if (count_ == 0U) throw std::logic_error("running statistic is empty");
    }

    std::size_t count_{0};
    double mean_{0.0};
    double m2_{0.0};
    double minimum_{std::numeric_limits<double>::infinity()};
    double maximum_{-std::numeric_limits<double>::infinity()};
};

class RollingStatistics {
public:
    explicit RollingStatistics(std::size_t window) : window_(window)
    {
        if (window == 0U) throw std::invalid_argument("rolling window must be non-zero");
    }

    void push(double value)
    {
        if (!std::isfinite(value)) throw std::invalid_argument("rolling value must be finite");
        values_.push_back(value);
        sum_ += value;
        square_sum_ += value * value;
        if (values_.size() > window_) {
            const double old = values_.front();
            values_.pop_front();
            sum_ -= old;
            square_sum_ -= old * old;
        }
    }

    std::size_t count() const noexcept { return values_.size(); }
    double mean() const
    {
        require_values();
        return sum_ / static_cast<double>(values_.size());
    }
    double variance() const
    {
        require_values();
        const double current_mean = mean();
        return std::max(0.0, square_sum_ / static_cast<double>(values_.size())
            - current_mean * current_mean);
    }

private:
    void require_values() const
    {
        if (values_.empty()) throw std::logic_error("rolling statistic is empty");
    }

    std::size_t window_;
    std::deque<double> values_;
    double sum_{0.0};
    double square_sum_{0.0};
};

inline std::vector<float> absolute_z_outlier_scores(const std::vector<float>& values,
                                                    float epsilon = 1.0e-12F)
{
    std::vector<float> scores = z_score(values, epsilon);
    for (float& score : scores) score = std::fabs(score);
    return scores;
}

inline std::vector<std::size_t> threshold_filter(const std::vector<float>& scores,
                                                 float threshold)
{
    if (!std::isfinite(threshold)) throw std::invalid_argument("threshold must be finite");
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isnan(scores[i]) && scores[i] >= threshold) indices.push_back(i);
    }
    return indices;
}

} // namespace dr3::ai
