#include "VecX/dr3.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace DRC::VecD4D;

constexpr double kFiniteDifferenceStep = 1.0e-6;
constexpr double kSensitivityTolerance = 2.0e-8;

double logistic_score(const std::vector<double>& input,
                      const std::vector<double>& weights,
                      double bias)
{
    if (input.size() != weights.size()) throw std::invalid_argument("model lengths differ");
    double logit = bias;
    for (std::size_t i = 0; i < input.size(); ++i) logit += input[i] * weights[i];
    return 1.0 / (1.0 + std::exp(-logit));
}

std::pair<double, double> score_and_forward_sensitivity(
    const std::vector<double>& input,
    const std::vector<double>& weights,
    std::size_t active_input)
{
    VecXX input_vector(input);
    VecXX weight_vector(weights);
    auto active = D(input_vector);
    for (std::size_t i = 0; i < input.size(); ++i) {
        active.derivative()[i] = i == active_input ? 1.0 : 0.0;
    }
    const auto weighted = active * C(weight_vector);
    double logit = -0.3;
    double tangent = 0.0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        logit += weighted.value()[i];
        tangent += weighted.derivative()[i];
    }
    const VecxD dual_logit(logit, tangent);
    const auto score = 1.0 / (1.0 + exp(-dual_logit));
    return {score.getScalarValue(), score.getScalarDeriv()};
}

int run(bool self_test)
{
    const std::vector<double> input{0.25, -0.5, 1.25, 2.0};
    const std::vector<double> weights{0.7, -1.1, 0.2, 0.45};
    const double reference_score = logistic_score(input, weights, -0.3);
    bool passed = true;
    std::cout << std::setprecision(12)
              << "logistic score=" << reference_score << '\n';
    for (std::size_t active = 0; active < input.size(); ++active) {
        const auto aad = score_and_forward_sensitivity(input, weights, active);
        std::vector<double> plus(input);
        std::vector<double> minus(input);
        plus[active] += kFiniteDifferenceStep;
        minus[active] -= kFiniteDifferenceStep;
        const double finite_difference = (logistic_score(plus, weights, -0.3)
            - logistic_score(minus, weights, -0.3)) / (2.0 * kFiniteDifferenceStep);
        const double error = std::fabs(aad.second - finite_difference);
        std::cout << "dscore/dx[" << active << "] AAD=" << aad.second
                  << " finite_difference=" << finite_difference
                  << " abs_error=" << error << '\n';
        passed = passed && std::fabs(aad.first - reference_score) <= 1.0e-12
            && error <= kSensitivityTolerance;
    }
    if (self_test) {
        std::cout << "self-test tolerance=" << kSensitivityTolerance << '\n';
    }
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        AllAllocatorsGuard<typename VecXX::SCALA_TYPE> allocator_guard;
        const bool self_test = argc > 1 && std::string(argv[1]) == "--self-test";
        if (argc > 1 && !self_test) throw std::invalid_argument("expected --self-test");
        return run(self_test);
    } catch (const std::exception& error) {
        std::cerr << "ModelSensitivity failed: " << error.what() << '\n';
        return 1;
    }
}
