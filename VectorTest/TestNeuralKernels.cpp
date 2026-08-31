#include "pch.h"

#include "../Vectorisation/VecX/neural_kernels.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

bool close(double actual, double expected, double abs_tol = 2e-5, double rel_tol = 2e-5)
{
    const double diff = std::abs(actual - expected);
    return diff <= abs_tol || diff <= rel_tol * std::max({1.0, std::abs(actual), std::abs(expected)});
}

std::vector<double> softmax_ref(const std::vector<float>& x)
{
    const double xmax = *std::max_element(x.begin(), x.end());
    std::vector<double> out(x.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(static_cast<double>(x[i]) - xmax);
        sum += out[i];
    }
    for (double& v : out) {
        v /= sum;
    }
    return out;
}

} // namespace

TEST(Softmax, UniformSingleAndTails)
{
    for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9}, std::size_t{17}}) {
        std::vector<float> x(n, 4.25f);
        std::vector<float> y(n);
        DRC::AI::softmax(x.data(), n, y.data());
        for (float v : y) {
            EXPECT_NEAR(v, 1.f / static_cast<float>(n), 2e-6f) << "n=" << n;
        }
    }
}

TEST(Softmax, LargeValuesAreStableAndMatchDoubleReference)
{
    const std::vector<float> x = {10000.f, 9999.f, -10000.f, -9999.f, 0.f, 7.f, -4.f, 10001.f, 2.f};
    std::vector<float> y(x.size());
    DRC::AI::softmax(x.data(), x.size(), y.data());
    const auto ref = softmax_ref(x);
    double sum = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_TRUE(std::isfinite(y[i]));
        sum += y[i];
        const double abs_error = std::abs(static_cast<double>(y[i]) - ref[i]);
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, abs_error / std::max(1e-30, std::abs(ref[i])));
    }
    RecordProperty("max_abs_error", max_abs_error);
    RecordProperty("max_rel_error", max_rel_error);
    EXPECT_NEAR(sum, 1.0, 2e-6);
    EXPECT_LT(max_abs_error, 2e-5);
}

TEST(Softmax, LogSoftmaxAndLogSumExpAreConsistent)
{
    const std::vector<float> x = {-5.f, -1.f, 0.f, 2.f, 9.f, 3.f, -7.f, 1.f, 0.5f};
    std::vector<float> soft(x.size()), logs(x.size());
    std::vector<float> soft_inplace = x, logs_inplace = x;
    DRC::AI::softmax(x.data(), x.size(), soft.data());
    DRC::AI::log_softmax(x.data(), x.size(), logs.data());
    DRC::AI::softmax_inplace(soft_inplace.data(), soft_inplace.size());
    DRC::AI::log_softmax_inplace(logs_inplace.data(), logs_inplace.size());
    const float lse = DRC::AI::log_sum_exp(x.data(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(logs[i], std::log(soft[i]), 2e-5f);
        EXPECT_NEAR(logs[i], x[i] - lse, 2e-6f);
        EXPECT_FLOAT_EQ(soft[i], soft_inplace[i]);
        EXPECT_FLOAT_EQ(logs[i], logs_inplace[i]);
    }
    EXPECT_EQ(DRC::AI::log_sum_exp(nullptr, 0), -std::numeric_limits<float>::infinity());
}

TEST(LayerNorm, MixedSignsScaleBiasAndTail)
{
    const std::vector<float> x = {-4.f, -1.f, 0.f, 2.f, 3.f, 7.f, -2.f, 1.f, 5.f};
    std::vector<float> scale(x.size()), bias(x.size()), y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        scale[i] = 0.8f + static_cast<float>(i) * 0.03f;
        bias[i] = -0.2f + static_cast<float>(i) * 0.01f;
    }
    DRC::AI::layer_norm(x.data(), x.size(), 1e-5f, y.data(), scale.data(), bias.data());
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double variance = 0.0;
    for (float v : x) {
        variance += (v - mean) * (v - mean);
    }
    variance /= x.size();
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double ref = ((x[i] - mean) / std::sqrt(variance + 1e-5)) * scale[i] + bias[i];
        EXPECT_TRUE(close(y[i], ref));
    }
}

TEST(LayerNorm, ZeroVarianceUsesEpsilonAndInPlaceMatches)
{
    std::vector<float> x(9, 3.f), out(x.size()), inplace = x;
    DRC::AI::layer_norm(x.data(), x.size(), 1e-5f, out.data());
    DRC::AI::layer_norm_inplace(inplace.data(), inplace.size(), 1e-5f);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_TRUE(std::isfinite(out[i]));
        EXPECT_NEAR(out[i], 0.f, 1e-7f);
        EXPECT_FLOAT_EQ(out[i], inplace[i]);
    }
    EXPECT_THROW(DRC::AI::layer_norm(x.data(), x.size(), 0.f, out.data()), std::invalid_argument);
}

TEST(RMSNorm, MatchesDoubleReferenceAndInPlace)
{
    const std::vector<float> x = {-4.f, -1.f, 0.f, 2.f, 3.f, 7.f, -2.f, 1.f, 5.f};
    std::vector<float> out(x.size()), inplace = x;
    DRC::AI::rms_norm(x.data(), x.size(), 1e-5f, out.data());
    DRC::AI::rms_norm_inplace(inplace.data(), inplace.size(), 1e-5f);
    double mean_square = 0.0;
    for (float v : x) {
        mean_square += static_cast<double>(v) * v;
    }
    mean_square /= x.size();
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_TRUE(close(out[i], x[i] / std::sqrt(mean_square + 1e-5)));
        EXPECT_FLOAT_EQ(out[i], inplace[i]);
    }
}

TEST(Activation, AllFunctionsMatchScalarReference)
{
    const std::vector<float> x = {-8.f, -2.f, -0.5f, 0.f, 0.25f, 1.f, 3.f, 8.f, -1.f};
    std::vector<float> out(x.size());
    DRC::AI::relu(x.data(), x.size(), out.data());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_FLOAT_EQ(out[i], std::max(x[i], 0.f));
    DRC::AI::leaky_relu(x.data(), x.size(), out.data(), 0.1f);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(out[i], x[i] >= 0 ? x[i] : 0.1f * x[i], 1e-6f);
    DRC::AI::sigmoid(x.data(), x.size(), out.data());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(out[i], 1.0 / (1.0 + std::exp(-x[i])), 2e-6);
    DRC::AI::tanh_activation(x.data(), x.size(), out.data());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(out[i], std::tanh(x[i]), 2e-6);
    DRC::AI::gelu(x.data(), x.size(), out.data());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double ref = 0.5 * x[i] * (1.0 + std::tanh(0.7978845608028654 * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
        EXPECT_NEAR(out[i], ref, 3e-5);
    }
    DRC::AI::silu(x.data(), x.size(), out.data());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(out[i], x[i] / (1.0 + std::exp(-x[i])), 3e-6);
}

TEST(Activation, ClipBiasHelperAndInPlace)
{
    const std::vector<float> x = {-3.f, -1.f, 0.f, 1.f, 3.f, 7.f, -7.f, 2.f, 4.f};
    const std::vector<float> bias(x.size(), 0.5f);
    std::vector<float> out(x.size()), inplace = x;
    DRC::AI::clip(x.data(), x.size(), -1.f, 2.f, out.data());
    DRC::AI::clip_inplace(inplace.data(), inplace.size(), -1.f, 2.f);
    EXPECT_EQ(out, inplace);
    DRC::AI::bias_plus_activation(x.data(), bias.data(), x.size(), out.data(), DRC::AI::Activation::Relu);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_FLOAT_EQ(out[i], std::max(0.f, x[i] + 0.5f));
    EXPECT_THROW(DRC::AI::clip(x.data(), x.size(), 2.f, -1.f, out.data()), std::invalid_argument);
}

TEST(Activation, InPlaceMatchesOutOfPlace)
{
    const std::vector<float> source = {-4.f, -1.f, 0.f, 0.5f, 2.f, 5.f, -3.f, 1.f, 7.f};
    for (DRC::AI::Activation kind : {DRC::AI::Activation::Relu, DRC::AI::Activation::LeakyRelu,
             DRC::AI::Activation::Sigmoid, DRC::AI::Activation::Tanh,
             DRC::AI::Activation::Gelu, DRC::AI::Activation::Silu}) {
        std::vector<float> out(source.size()), inplace = source;
        DRC::AI::activation(source.data(), source.size(), out.data(), kind);
        DRC::AI::activation_inplace(inplace.data(), inplace.size(), kind);
        EXPECT_EQ(out, inplace);
    }
}
