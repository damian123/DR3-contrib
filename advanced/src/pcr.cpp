#include "dr3/advanced/pcr.h"

#include "Vectorisation/VecX/vcl_latest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dr3::advanced
{
namespace
{

bool finiteVector(const std::vector<double>& values)
{
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
}

bool overlaps(const double* left, std::size_t leftSize,
              const double* right, std::size_t rightSize)
{
    if (left == nullptr || right == nullptr || leftSize == 0 || rightSize == 0)
    {
        return false;
    }
    const auto leftBegin = reinterpret_cast<std::uintptr_t>(left);
    const auto leftEnd = leftBegin + leftSize * sizeof(double);
    const auto rightBegin = reinterpret_cast<std::uintptr_t>(right);
    const auto rightEnd = rightBegin + rightSize * sizeof(double);
    return leftBegin < rightEnd && rightBegin < leftEnd;
}

void checkDenominator(double value, double tolerance)
{
    if (!std::isfinite(value) || std::abs(value) <= tolerance)
    {
        throw std::domain_error("PCR encountered a near-zero stage denominator");
    }
}

void scalarStageValue(std::size_t index, std::size_t stride, std::size_t count,
                      const double* lower, const double* diagonal,
                      const double* upper, const double* right,
                      double* nextLower, double* nextDiagonal,
                      double* nextUpper, double* nextRight,
                      double tolerance)
{
    double alpha = 0.0;
    double beta = 0.0;
    if (index >= stride)
    {
        checkDenominator(diagonal[index - stride], tolerance);
        alpha = -lower[index] / diagonal[index - stride];
    }
    if (index + stride < count)
    {
        checkDenominator(diagonal[index + stride], tolerance);
        beta = -upper[index] / diagonal[index + stride];
    }
    nextLower[index] = index >= stride ? alpha * lower[index - stride] : 0.0;
    nextUpper[index] = index + stride < count ? beta * upper[index + stride] : 0.0;
    nextDiagonal[index] = diagonal[index]
        + (index >= stride ? alpha * upper[index - stride] : 0.0)
        + (index + stride < count ? beta * lower[index + stride] : 0.0);
    nextRight[index] = right[index]
        + (index >= stride ? alpha * right[index - stride] : 0.0)
        + (index + stride < count ? beta * right[index + stride] : 0.0);
    checkDenominator(nextDiagonal[index], tolerance);
    if (!std::isfinite(nextLower[index]) || !std::isfinite(nextUpper[index])
        || !std::isfinite(nextRight[index]))
    {
        throw std::domain_error("PCR stage produced non-finite coefficients");
    }
}

void simdInteriorBlock(std::size_t index, std::size_t stride,
                       const double* lower, const double* diagonal,
                       const double* upper, const double* right,
                       double* nextLower, double* nextDiagonal,
                       double* nextUpper, double* nextRight,
                       double tolerance)
{
    Vec4d diagonalLeft;
    Vec4d diagonalRight;
    diagonalLeft.load(diagonal + index - stride);
    diagonalRight.load(diagonal + index + stride);
    alignas(32) double denominatorValues[8];
    diagonalLeft.store(denominatorValues);
    diagonalRight.store(denominatorValues + 4);
    for (double denominator : denominatorValues)
    {
        checkDenominator(denominator, tolerance);
    }

    Vec4d a;
    Vec4d b;
    Vec4d c;
    Vec4d d;
    Vec4d aLeft;
    Vec4d cLeft;
    Vec4d aRight;
    Vec4d cRight;
    Vec4d dLeft;
    Vec4d dRight;
    a.load(lower + index);
    b.load(diagonal + index);
    c.load(upper + index);
    d.load(right + index);
    aLeft.load(lower + index - stride);
    cLeft.load(upper + index - stride);
    aRight.load(lower + index + stride);
    cRight.load(upper + index + stride);
    dLeft.load(right + index - stride);
    dRight.load(right + index + stride);
    const Vec4d alpha = -a / diagonalLeft;
    const Vec4d beta = -c / diagonalRight;
    const Vec4d resultLower = alpha * aLeft;
    const Vec4d resultUpper = beta * cRight;
    const Vec4d resultDiagonal = b + alpha * cLeft + beta * aRight;
    const Vec4d resultRight = d + alpha * dLeft + beta * dRight;
    resultLower.store(nextLower + index);
    resultDiagonal.store(nextDiagonal + index);
    resultUpper.store(nextUpper + index);
    resultRight.store(nextRight + index);
    for (std::size_t lane = 0; lane < 4; ++lane)
    {
        checkDenominator(nextDiagonal[index + lane], tolerance);
        if (!std::isfinite(nextLower[index + lane])
            || !std::isfinite(nextUpper[index + lane])
            || !std::isfinite(nextRight[index + lane]))
        {
            throw std::domain_error("SIMD PCR stage produced non-finite coefficients");
        }
    }
}

} // namespace

void PcrSystem::validate() const
{
    if (diagonal.empty() || lower.size() + 1 != diagonal.size()
        || upper.size() + 1 != diagonal.size())
    {
        throw std::invalid_argument("PCR coefficient dimensions are inconsistent");
    }
    if (!finiteVector(lower) || !finiteVector(diagonal) || !finiteVector(upper))
    {
        throw std::invalid_argument("PCR coefficients must be finite");
    }
}

std::size_t PcrWorkspace::checkedNextPowerOfTwo(std::size_t value)
{
    if (value == 0)
    {
        throw std::invalid_argument("PCR workspace logical size must be positive");
    }
    std::size_t result = 1;
    while (result < value)
    {
        if (result > std::numeric_limits<std::size_t>::max() / 2)
        {
            throw std::overflow_error("PCR padded size overflows size_t");
        }
        result *= 2;
    }
    return result;
}

PcrWorkspace::PcrWorkspace(std::size_t logicalSize)
    : logicalSize_(logicalSize), paddedSize_(checkedNextPowerOfTwo(logicalSize)),
      lower0_(paddedSize_), diagonal0_(paddedSize_), upper0_(paddedSize_),
      right0_(paddedSize_), lower1_(paddedSize_), diagonal1_(paddedSize_),
      upper1_(paddedSize_), right1_(paddedSize_)
{
    std::size_t count = paddedSize_;
    while (count > 1)
    {
        ++stageCount_;
        count /= 2;
    }
}

void PcrSolver::solve(const PcrSystem& system,
                      const double* rightHandSide,
                      std::size_t rightHandSideSize,
                      double* output,
                      std::size_t outputSize,
                      PcrWorkspace& workspace,
                      bool useSimdStages,
                      double denominatorTolerance)
{
    system.validate();
    const std::size_t logicalSize = system.size();
    if (rightHandSideSize != logicalSize || outputSize != logicalSize
        || workspace.logicalSize_ != logicalSize
        || rightHandSide == nullptr || output == nullptr)
    {
        throw std::invalid_argument("PCR coefficient, RHS, output, and workspace sizes must agree");
    }
    if (!std::isfinite(denominatorTolerance) || denominatorTolerance <= 0.0)
    {
        throw std::invalid_argument("PCR denominator tolerance must be finite and positive");
    }
    if (overlaps(rightHandSide, logicalSize, output, logicalSize)
        || overlaps(system.lower.data(), system.lower.size(), output, logicalSize)
        || overlaps(system.diagonal.data(), system.diagonal.size(), output, logicalSize)
        || overlaps(system.upper.data(), system.upper.size(), output, logicalSize))
    {
        throw std::invalid_argument("PCR output may not alias its inputs");
    }
    for (std::size_t index = 0; index < logicalSize; ++index)
    {
        if (!std::isfinite(rightHandSide[index]))
        {
            throw std::invalid_argument("PCR right-hand side must be finite");
        }
        workspace.lower0_[index] = index == 0 ? 0.0 : system.lower[index - 1];
        workspace.diagonal0_[index] = system.diagonal[index];
        workspace.upper0_[index] = index + 1 == logicalSize ? 0.0 : system.upper[index];
        workspace.right0_[index] = rightHandSide[index];
        checkDenominator(workspace.diagonal0_[index], denominatorTolerance);
    }
    // Independent identity rows are the documented non-power-of-two padding.
    for (std::size_t index = logicalSize; index < workspace.paddedSize_; ++index)
    {
        workspace.lower0_[index] = 0.0;
        workspace.diagonal0_[index] = 1.0;
        workspace.upper0_[index] = 0.0;
        workspace.right0_[index] = 0.0;
    }

    for (std::size_t stride = 1; stride < workspace.paddedSize_; stride *= 2)
    {
        const std::size_t count = workspace.paddedSize_;
        std::size_t index = 0;
        if (useSimdStages && stride * 2 < count)
        {
            for (; index < stride; ++index)
            {
                scalarStageValue(index, stride, count,
                    workspace.lower0_.data(), workspace.diagonal0_.data(),
                    workspace.upper0_.data(), workspace.right0_.data(),
                    workspace.lower1_.data(), workspace.diagonal1_.data(),
                    workspace.upper1_.data(), workspace.right1_.data(),
                    denominatorTolerance);
            }
            const std::size_t interiorEnd = count - stride;
            for (; index + 4 <= interiorEnd; index += 4)
            {
                simdInteriorBlock(index, stride,
                    workspace.lower0_.data(), workspace.diagonal0_.data(),
                    workspace.upper0_.data(), workspace.right0_.data(),
                    workspace.lower1_.data(), workspace.diagonal1_.data(),
                    workspace.upper1_.data(), workspace.right1_.data(),
                    denominatorTolerance);
            }
        }
        for (; index < count; ++index)
        {
            scalarStageValue(index, stride, count,
                workspace.lower0_.data(), workspace.diagonal0_.data(),
                workspace.upper0_.data(), workspace.right0_.data(),
                workspace.lower1_.data(), workspace.diagonal1_.data(),
                workspace.upper1_.data(), workspace.right1_.data(),
                denominatorTolerance);
        }
        workspace.lower0_.swap(workspace.lower1_);
        workspace.diagonal0_.swap(workspace.diagonal1_);
        workspace.upper0_.swap(workspace.upper1_);
        workspace.right0_.swap(workspace.right1_);
    }

    for (std::size_t index = 0; index < logicalSize; ++index)
    {
        checkDenominator(workspace.diagonal0_[index], denominatorTolerance);
        output[index] = workspace.right0_[index] / workspace.diagonal0_[index];
        if (!std::isfinite(output[index]))
        {
            throw std::domain_error("PCR produced a non-finite solution");
        }
    }
}

double PcrSolver::infinityResidual(const PcrSystem& system,
                                   const std::vector<double>& rightHandSide,
                                   const std::vector<double>& solution)
{
    system.validate();
    if (rightHandSide.size() != system.size() || solution.size() != system.size())
    {
        throw std::invalid_argument("PCR residual dimensions are inconsistent");
    }
    double residual = 0.0;
    for (std::size_t row = 0; row < system.size(); ++row)
    {
        double product = system.diagonal[row] * solution[row];
        if (row != 0)
        {
            product += system.lower[row - 1] * solution[row - 1];
        }
        if (row + 1 != system.size())
        {
            product += system.upper[row] * solution[row + 1];
        }
        const double rowResidual = std::abs(product - rightHandSide[row]);
        if (!std::isfinite(rowResidual))
        {
            throw std::domain_error("PCR residual is not finite");
        }
        residual = std::max(residual, rowResidual);
    }
    return residual;
}

} // namespace dr3::advanced
