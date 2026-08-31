// Deterministic small-batch MLP inference using DR3 float32 SIMD kernels.

#include "../Vectorisation/VecX/alloc_policy.h"
#include "../Vectorisation/VecX/neural_kernels.h"
#include "DenseLayer.h"
#include "expected_outputs.h"
#include "model_weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

AllAllocatorsGuard<float> allocGuard;

namespace {

constexpr float kEpsilon = 1.0e-5f;
constexpr double kSelfTestTolerance = 5.0e-5;

struct OwnedModel {
    std::size_t input_size;
    std::size_t hidden_size;
    std::size_t output_size;
    std::vector<float> weights1;
    std::vector<float> bias1;
    std::vector<float> weights2;
    std::vector<float> bias2;

    dr3_mlp::DenseLayerView first() const
    {
        return {weights1.data(), bias1.data(), input_size, hidden_size};
    }

    dr3_mlp::DenseLayerView second() const
    {
        return {weights2.data(), bias2.data(), hidden_size, output_size};
    }
};

float deterministic_value(std::size_t i, float scale, float phase)
{
    return scale * static_cast<float>(
        std::sin(static_cast<double>(i + 1) * 0.173 + phase) +
        0.5 * std::cos(static_cast<double>(i + 1) * 0.071 - phase));
}

OwnedModel make_model(std::size_t input_size, std::size_t hidden_size, std::size_t output_size)
{
    OwnedModel model;
    model.input_size = input_size;
    model.hidden_size = hidden_size;
    model.output_size = output_size;
    model.weights1.resize(input_size * hidden_size);
    model.bias1.resize(hidden_size);
    model.weights2.resize(hidden_size * output_size);
    model.bias2.resize(output_size);
    const float first_scale = 0.35f / std::sqrt(static_cast<float>(input_size));
    const float second_scale = 0.35f / std::sqrt(static_cast<float>(hidden_size));
    for (std::size_t i = 0; i < model.weights1.size(); ++i) {
        model.weights1[i] = deterministic_value(i, first_scale, 0.3f);
    }
    for (std::size_t i = 0; i < model.weights2.size(); ++i) {
        model.weights2[i] = deterministic_value(i, second_scale, 1.1f);
    }
    for (std::size_t i = 0; i < hidden_size; ++i) {
        model.bias1[i] = deterministic_value(i, 0.03f, 0.7f);
    }
    for (std::size_t i = 0; i < output_size; ++i) {
        model.bias2[i] = deterministic_value(i, 0.03f, 1.7f);
    }
    return model;
}

std::vector<float> make_inputs(std::size_t batch, std::size_t input_size)
{
    std::vector<float> inputs(batch * input_size);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = deterministic_value(i, 0.8f, 2.3f);
    }
    return inputs;
}

double gelu_scalar(double x)
{
    return 0.5 * x * (1.0 + std::tanh(
        0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

void predict_scalar(
    const dr3_mlp::DenseLayerView& first,
    const dr3_mlp::DenseLayerView& second,
    const float* input,
    float* output,
    std::vector<float>& hidden,
    std::vector<float>& logits)
{
    dr3_mlp::dense_forward_scalar(first, input, first.input_size,
        hidden.data(), first.output_size);
    double mean_square = 0.0;
    for (float value : hidden) {
        mean_square += static_cast<double>(value) * value;
    }
    mean_square /= static_cast<double>(hidden.size());
    const double inv_rms = 1.0 / std::sqrt(mean_square + kEpsilon);
    for (float& value : hidden) {
        value = static_cast<float>(gelu_scalar(static_cast<double>(value) * inv_rms));
    }
    dr3_mlp::dense_forward_scalar(second, hidden.data(), hidden.size(),
        logits.data(), logits.size());
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float value : logits) {
        sum += std::exp(static_cast<double>(value) - max_logit);
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        output[i] = static_cast<float>(
            std::exp(static_cast<double>(logits[i]) - max_logit) / sum);
    }
}

void predict_simd(
    const dr3_mlp::DenseLayerView& first,
    const dr3_mlp::DenseLayerView& second,
    const float* input,
    float* output,
    std::vector<float>& hidden,
    std::vector<float>& logits)
{
    dr3_mlp::dense_forward_simd(first, input, first.input_size,
        hidden.data(), first.output_size);
    DRC::AI::rms_norm_inplace(hidden.data(), hidden.size(), kEpsilon);
    DRC::AI::activation_inplace(hidden.data(), hidden.size(), DRC::AI::Activation::Gelu);
    dr3_mlp::dense_forward_simd(second, hidden.data(), hidden.size(),
        logits.data(), logits.size());
    DRC::AI::softmax(logits.data(), logits.size(), output);
}

void predict_batch(
    bool simd,
    const dr3_mlp::DenseLayerView& first,
    const dr3_mlp::DenseLayerView& second,
    const float* inputs,
    std::size_t batch,
    float* outputs)
{
    if (batch == 0 || batch > 16) {
        throw std::invalid_argument("predict_batch: batch must be in [1, 16]");
    }
    std::vector<float> hidden(first.output_size);
    std::vector<float> logits(second.output_size);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        const float* input = inputs + sample * first.input_size;
        float* output = outputs + sample * second.output_size;
        if (simd) {
            predict_simd(first, second, input, output, hidden, logits);
        } else {
            predict_scalar(first, second, input, output, hidden, logits);
        }
    }
}

int fail(const std::string& message)
{
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int run_self_test()
{
    using namespace dr3_mlp::fixture;
    const dr3_mlp::DenseLayerView first{
        kWeights1.data(), kBias1.data(), kInput, kHidden};
    const dr3_mlp::DenseLayerView second{
        kWeights2.data(), kBias2.data(), kHidden, kOutput};
    std::vector<float> scalar(kBatch * kOutput);
    std::vector<float> simd(kBatch * kOutput);
    predict_batch(false, first, second, kInputs.data(), kBatch, scalar.data());
    predict_batch(true, first, second, kInputs.data(), kBatch, simd.data());

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < simd.size(); ++i) {
        const double abs_error = std::abs(static_cast<double>(simd[i]) - scalar[i]);
        const double rel_error = abs_error /
            std::max(1.0e-12, std::abs(static_cast<double>(scalar[i])));
        max_abs = std::max(max_abs, abs_error);
        max_rel = std::max(max_rel, rel_error);
        if (abs_error > kSelfTestTolerance) {
            return fail("SIMD output differs from scalar reference at index " + std::to_string(i));
        }
        if (std::abs(static_cast<double>(simd[i]) - kExpectedOutputs[i]) > kSelfTestTolerance) {
            return fail("SIMD output differs from committed fixture at index " + std::to_string(i));
        }
    }
    for (std::size_t sample = 0; sample < kBatch; ++sample) {
        double probability_sum = 0.0;
        for (std::size_t i = 0; i < kOutput; ++i) {
            probability_sum += simd[sample * kOutput + i];
        }
        if (std::abs(probability_sum - 1.0) > 2.0e-6) {
            return fail("softmax probabilities do not sum to one");
        }
    }

    try {
        std::vector<float> output(kHidden);
        dr3_mlp::dense_forward_simd(first, kInputs.data(), kInput - 1,
            output.data(), output.size());
        return fail("dense layer accepted a mismatched input length");
    } catch (const std::invalid_argument&) {
    }
    try {
        predict_batch(true, first, second, kInputs.data(), 17, simd.data());
        return fail("batch size above 16 was accepted");
    } catch (const std::invalid_argument&) {
    }

    // Exercise every supported batch size with deterministic, non-width dimensions.
    for (std::size_t batch = 1; batch <= 16; ++batch) {
        auto model = make_model(9, 17, 5);
        auto inputs = make_inputs(batch, model.input_size);
        std::vector<float> a(batch * model.output_size);
        std::vector<float> b(batch * model.output_size);
        predict_batch(true, model.first(), model.second(), inputs.data(), batch, a.data());
        predict_batch(true, model.first(), model.second(), inputs.data(), batch, b.data());
        if (a != b) {
            return fail("non-deterministic output for batch " + std::to_string(batch));
        }
    }

    std::cout << "SmallBatchInference self-test passed: max_abs=" << max_abs
              << " max_rel=" << max_rel
              << " tolerance=" << kSelfTestTolerance << '\n';
    return 0;
}

void emit_expected()
{
    using namespace dr3_mlp::fixture;
    const dr3_mlp::DenseLayerView first{
        kWeights1.data(), kBias1.data(), kInput, kHidden};
    const dr3_mlp::DenseLayerView second{
        kWeights2.data(), kBias2.data(), kHidden, kOutput};
    std::vector<float> output(kBatch * kOutput);
    predict_batch(false, first, second, kInputs.data(), kBatch, output.data());
    std::cout << std::setprecision(9);
    for (float value : output) {
        std::cout << value << "f,\n";
    }
}

const char* compiler_name()
{
#if defined(_MSC_VER)
    return "MSVC";
#elif defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GCC";
#else
    return "unknown";
#endif
}

const char* build_type()
{
#ifdef NDEBUG
    return "Release";
#else
    return "Debug";
#endif
}

void benchmark_shape(
    std::size_t input_size, std::size_t hidden_size,
    std::size_t output_size, std::size_t batch)
{
    auto model = make_model(input_size, hidden_size, output_size);
    auto inputs = make_inputs(batch, input_size);
    std::vector<float> scalar(batch * output_size);
    std::vector<float> simd(batch * output_size);
    predict_batch(false, model.first(), model.second(), inputs.data(), batch, scalar.data());
    predict_batch(true, model.first(), model.second(), inputs.data(), batch, simd.data());

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        const double abs_error = std::abs(static_cast<double>(simd[i]) - scalar[i]);
        max_abs = std::max(max_abs, abs_error);
        max_rel = std::max(max_rel, abs_error /
            std::max(1.0e-12, std::abs(static_cast<double>(scalar[i]))));
    }

    const std::size_t work = input_size * hidden_size + hidden_size * output_size;
    const std::size_t repetitions = std::max<std::size_t>(3, 3000000 / std::max<std::size_t>(1, work * batch));
    volatile float checksum = 0.f;
    auto time_path = [&](bool use_simd, std::vector<float>& output) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < repetitions; ++i) {
            predict_batch(use_simd, model.first(), model.second(),
                inputs.data(), batch, output.data());
            checksum += output[i % output.size()];
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    const double scalar_seconds = time_path(false, scalar);
    const double simd_seconds = time_path(true, simd);
    const double predictions = static_cast<double>(repetitions * batch);

    std::cout << input_size << "->" << hidden_size << "->" << output_size
              << " batch=" << batch
              << " scalar_us=" << scalar_seconds * 1.0e6 / predictions
              << " simd_us=" << simd_seconds * 1.0e6 / predictions
              << " scalar_predictions_per_s=" << predictions / scalar_seconds
              << " simd_predictions_per_s=" << predictions / simd_seconds
              << " max_abs=" << max_abs
              << " max_rel=" << max_rel
              << " checksum=" << checksum << '\n';
}

void run_benchmarks()
{
    std::cout << "compiler=" << compiler_name()
              << " INSTRSET=" << INSTRSET
              << " build=" << build_type() << '\n';
    benchmark_shape(32, 64, 2, 1);
    benchmark_shape(64, 128, 8, 4);
    benchmark_shape(128, 256, 16, 8);
    benchmark_shape(768, 256, 2, 16);
    std::cout << "Measurements depend on CPU, compiler, instruction set, build type, and workload.\n";
}

} // namespace

int main(int argc, char** argv)
{
    bool self_test = false;
    bool emit_fixture = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--self-test") {
            self_test = true;
        } else if (arg == "--emit-expected") {
            emit_fixture = true;
        } else {
            std::cerr << "Unknown option: " << arg << '\n';
            return 2;
        }
    }
    if (emit_fixture) {
        emit_expected();
        return 0;
    }
    if (self_test) {
        return run_self_test();
    }
    run_benchmarks();
    return 0;
}
