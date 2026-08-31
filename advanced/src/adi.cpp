#include "dr3/advanced/adi.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dr3::advanced
{
namespace
{

bool finiteBuffer(const double* values, std::size_t count)
{
    for (std::size_t index = 0; index < count; ++index)
    {
        if (!std::isfinite(values[index]))
        {
            return false;
        }
    }
    return true;
}

void requireFinite(const std::vector<double>& values, const char* stage)
{
    if (!finiteBuffer(values.data(), values.size()))
    {
        throw std::runtime_error(std::string("ADI stage produced a non-finite surface: ") + stage);
    }
}

void applyBoundary(const AdiBoundary& boundary, double time,
                   const Grid2D& grid, std::vector<double>& values)
{
    boundary(time, grid, values.data());
    requireFinite(values, "boundary refresh");
}

} // namespace

SplitOperator2D::SplitOperator2D(Grid2D grid)
    : grid_(std::move(grid)), firstLower_(size(), 0.0), firstDiagonal_(size(), 0.0),
      firstUpper_(size(), 0.0), secondLower_(size(), 0.0),
      secondDiagonal_(size(), 0.0), secondUpper_(size(), 0.0)
{
    for (auto& coefficient : explicit_)
    {
        coefficient.assign(size(), 0.0);
    }
}

std::size_t SplitOperator2D::explicitSlot(int firstOffset, int secondOffset)
{
    if (firstOffset < -1 || firstOffset > 1 || secondOffset < -1 || secondOffset > 1)
    {
        throw std::out_of_range("explicit stencil offset must be in [-1, 1]");
    }
    return static_cast<std::size_t>((secondOffset + 1) * 3 + firstOffset + 1);
}

double& SplitOperator2D::explicitCoefficient(int firstOffset, int secondOffset,
                                             std::size_t logicalIndex)
{
    return explicit_[explicitSlot(firstOffset, secondOffset)].at(logicalIndex);
}

double SplitOperator2D::explicitCoefficient(int firstOffset, int secondOffset,
                                            std::size_t logicalIndex) const
{
    return explicit_[explicitSlot(firstOffset, secondOffset)].at(logicalIndex);
}

void SplitOperator2D::validate() const
{
    const auto validateVector = [this](const std::vector<double>& values)
    {
        if (values.size() != size() || !finiteBuffer(values.data(), values.size()))
        {
            throw std::invalid_argument("ADI operator coefficients must match the grid and be finite");
        }
    };
    for (const auto& values : explicit_)
    {
        validateVector(values);
    }
    validateVector(firstLower_);
    validateVector(firstDiagonal_);
    validateVector(firstUpper_);
    validateVector(secondLower_);
    validateVector(secondDiagonal_);
    validateVector(secondUpper_);
}

void SplitOperator2D::validateBuffers(const double* input, double* output,
                                      std::size_t count) const
{
    if (count != size() || input == nullptr || output == nullptr)
    {
        throw std::invalid_argument("ADI operator buffers must exactly match the grid");
    }
    if (input == output)
    {
        throw std::invalid_argument("in-place ADI operator evaluation is unsupported");
    }
    if (!finiteBuffer(input, count))
    {
        throw std::invalid_argument("ADI operator input surface must be finite");
    }
}

void SplitOperator2D::applyExplicit(const double* input, double* output,
                                    std::size_t count) const
{
    validateBuffers(input, output, count);
    std::fill(output, output + count, 0.0);
    for (std::size_t second = 1; second + 1 < grid_.secondSize(); ++second)
    {
        for (std::size_t first = 1; first + 1 < grid_.firstSize(); ++first)
        {
            const std::size_t center = grid_.index(first, second);
            double result = 0.0;
            for (int secondOffset = -1; secondOffset <= 1; ++secondOffset)
            {
                for (int firstOffset = -1; firstOffset <= 1; ++firstOffset)
                {
                    const std::size_t neighbor = grid_.index(
                        static_cast<std::size_t>(static_cast<int>(first) + firstOffset),
                        static_cast<std::size_t>(static_cast<int>(second) + secondOffset));
                    result += explicitCoefficient(firstOffset, secondOffset, center)
                        * input[neighbor];
                }
            }
            output[center] = result;
        }
    }
}

void SplitOperator2D::applyFirst(const double* input, double* output,
                                 std::size_t count) const
{
    validateBuffers(input, output, count);
    std::fill(output, output + count, 0.0);
    for (std::size_t second = 1; second + 1 < grid_.secondSize(); ++second)
    {
        for (std::size_t first = 1; first + 1 < grid_.firstSize(); ++first)
        {
            const std::size_t center = grid_.index(first, second);
            output[center] = firstLower_[center] * input[grid_.index(first - 1, second)]
                + firstDiagonal_[center] * input[center]
                + firstUpper_[center] * input[grid_.index(first + 1, second)];
        }
    }
}

void SplitOperator2D::applySecond(const double* input, double* output,
                                  std::size_t count) const
{
    validateBuffers(input, output, count);
    std::fill(output, output + count, 0.0);
    for (std::size_t second = 1; second + 1 < grid_.secondSize(); ++second)
    {
        for (std::size_t first = 1; first + 1 < grid_.firstSize(); ++first)
        {
            const std::size_t center = grid_.index(first, second);
            output[center] = secondLower_[center] * input[grid_.index(first, second - 1)]
                + secondDiagonal_[center] * input[center]
                + secondUpper_[center] * input[grid_.index(first, second + 1)];
        }
    }
}

void SplitOperator2D::applyFull(const double* input, double* output,
                                std::size_t count) const
{
    validateBuffers(input, output, count);
    std::fill(output, output + count, 0.0);
    for (std::size_t second = 1; second + 1 < grid_.secondSize(); ++second)
    {
        for (std::size_t first = 1; first + 1 < grid_.firstSize(); ++first)
        {
            const std::size_t center = grid_.index(first, second);
            double result = firstLower_[center] * input[grid_.index(first - 1, second)]
                + firstDiagonal_[center] * input[center]
                + firstUpper_[center] * input[grid_.index(first + 1, second)]
                + secondLower_[center] * input[grid_.index(first, second - 1)]
                + secondDiagonal_[center] * input[center]
                + secondUpper_[center] * input[grid_.index(first, second + 1)];
            for (int secondOffset = -1; secondOffset <= 1; ++secondOffset)
            {
                for (int firstOffset = -1; firstOffset <= 1; ++firstOffset)
                {
                    result += explicitCoefficient(firstOffset, secondOffset, center)
                        * input[grid_.index(
                            static_cast<std::size_t>(static_cast<int>(first) + firstOffset),
                            static_cast<std::size_t>(static_cast<int>(second) + secondOffset))];
                }
            }
            output[center] = result;
        }
    }
}

void AdiWorkspace::initialize(const SplitOperator2D& splitOperator,
                              double timeStep, double theta, double pivotTolerance)
{
    splitOperator.validate();
    if (!std::isfinite(timeStep) || timeStep <= 0.0)
    {
        throw std::invalid_argument("ADI time step must be finite and positive");
    }
    // Douglas and MCS are supported for 0 < theta <= 1. MCS defaults to 1/3.
    if (!std::isfinite(theta) || theta <= 0.0 || theta > 1.0)
    {
        throw std::invalid_argument("ADI theta must be finite and in (0, 1]");
    }
    if (!std::isfinite(pivotTolerance) || pivotTolerance <= 0.0)
    {
        throw std::invalid_argument("ADI pivot tolerance must be finite and positive");
    }

    operatorIdentity_ = &splitOperator;
    timeStep_ = timeStep;
    theta_ = theta;
    pivotTolerance_ = pivotTolerance;
    const std::size_t count = splitOperator.size();
    firstPivots_.assign(count, 1.0);
    firstMultipliers_.assign(count, 0.0);
    secondPivots_.assign(count, 1.0);
    secondMultipliers_.assign(count, 0.0);
    lineForward_.assign(std::max(splitOperator.grid().firstSize(),
                                 splitOperator.grid().secondSize()), 0.0);
    y0_.assign(count, 0.0);
    y1_.assign(count, 0.0);
    y2_.assign(count, 0.0);
    hat_.assign(count, 0.0);
    corrected_.assign(count, 0.0);
    difference_.assign(count, 0.0);
    operatorWork_.assign(count, 0.0);
    rightHandSide_.assign(count, 0.0);

    const Grid2D& grid = splitOperator.grid();
    const double scale = theta_ * timeStep_;
    const auto checkPivot = [this](double pivot)
    {
        if (!std::isfinite(pivot) || std::abs(pivot) <= pivotTolerance_)
        {
            throw std::domain_error("ADI directional system has an invalid pivot");
        }
    };

    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        std::size_t center = grid.index(1, second);
        firstPivots_[center] = 1.0 - scale * splitOperator.firstDiagonal(center);
        checkPivot(firstPivots_[center]);
        for (std::size_t first = 2; first + 1 < grid.firstSize(); ++first)
        {
            center = grid.index(first, second);
            const std::size_t previous = grid.index(first - 1, second);
            const double matrixLower = -scale * splitOperator.firstLower(center);
            firstMultipliers_[center] = matrixLower / firstPivots_[previous];
            const double previousUpper = -scale * splitOperator.firstUpper(previous);
            firstPivots_[center] = 1.0 - scale * splitOperator.firstDiagonal(center)
                - firstMultipliers_[center] * previousUpper;
            checkPivot(firstPivots_[center]);
        }
    }

    for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
    {
        std::size_t center = grid.index(first, 1);
        secondPivots_[center] = 1.0 - scale * splitOperator.secondDiagonal(center);
        checkPivot(secondPivots_[center]);
        for (std::size_t second = 2; second + 1 < grid.secondSize(); ++second)
        {
            center = grid.index(first, second);
            const std::size_t previous = grid.index(first, second - 1);
            const double matrixLower = -scale * splitOperator.secondLower(center);
            secondMultipliers_[center] = matrixLower / secondPivots_[previous];
            const double previousUpper = -scale * splitOperator.secondUpper(previous);
            secondPivots_[center] = 1.0 - scale * splitOperator.secondDiagonal(center)
                - secondMultipliers_[center] * previousUpper;
            checkPivot(secondPivots_[center]);
        }
    }
    initialized_ = true;
    ++factorizationBuildCount_;
}

void AdiWorkspace::validateFor(const SplitOperator2D& splitOperator) const
{
    if (!initialized_)
    {
        throw std::logic_error("ADI workspace must be initialized before stepping");
    }
    if (operatorIdentity_ != &splitOperator || y0_.size() != splitOperator.size())
    {
        throw std::invalid_argument("ADI workspace and operator dimensions/identity differ");
    }
}

void AdiWorkspace::solveFirst(const SplitOperator2D& splitOperator,
                              const double* rightHandSide, double* output) noexcept
{
    const Grid2D& grid = splitOperator.grid();
    const double scale = theta_ * timeStep_;
    const std::size_t lastInterior = grid.firstSize() - 2;
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        std::size_t center = grid.index(1, second);
        double adjusted = rightHandSide[center]
            + scale * splitOperator.firstLower(center) * output[grid.index(0, second)];
        if (lastInterior == 1)
        {
            adjusted += scale * splitOperator.firstUpper(center)
                * output[grid.index(2, second)];
        }
        lineForward_[1] = adjusted;
        for (std::size_t first = 2; first + 1 < grid.firstSize(); ++first)
        {
            center = grid.index(first, second);
            adjusted = rightHandSide[center];
            if (first == lastInterior)
            {
                adjusted += scale * splitOperator.firstUpper(center)
                    * output[grid.index(first + 1, second)];
            }
            lineForward_[first] = adjusted
                - firstMultipliers_[center] * lineForward_[first - 1];
        }
        center = grid.index(lastInterior, second);
        output[center] = lineForward_[lastInterior] / firstPivots_[center];
        for (std::size_t first = lastInterior; first-- > 1;)
        {
            center = grid.index(first, second);
            const double matrixUpper = -scale * splitOperator.firstUpper(center);
            output[center] = (lineForward_[first]
                - matrixUpper * output[grid.index(first + 1, second)]) / firstPivots_[center];
        }
    }
}

void AdiWorkspace::solveSecond(const SplitOperator2D& splitOperator,
                               const double* rightHandSide, double* output) noexcept
{
    const Grid2D& grid = splitOperator.grid();
    const double scale = theta_ * timeStep_;
    const std::size_t lastInterior = grid.secondSize() - 2;
    for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
    {
        std::size_t center = grid.index(first, 1);
        double adjusted = rightHandSide[center]
            + scale * splitOperator.secondLower(center) * output[grid.index(first, 0)];
        if (lastInterior == 1)
        {
            adjusted += scale * splitOperator.secondUpper(center)
                * output[grid.index(first, 2)];
        }
        lineForward_[1] = adjusted;
        for (std::size_t second = 2; second + 1 < grid.secondSize(); ++second)
        {
            center = grid.index(first, second);
            adjusted = rightHandSide[center];
            if (second == lastInterior)
            {
                adjusted += scale * splitOperator.secondUpper(center)
                    * output[grid.index(first, second + 1)];
            }
            lineForward_[second] = adjusted
                - secondMultipliers_[center] * lineForward_[second - 1];
        }
        center = grid.index(first, lastInterior);
        output[center] = lineForward_[lastInterior] / secondPivots_[center];
        for (std::size_t second = lastInterior; second-- > 1;)
        {
            center = grid.index(first, second);
            const double matrixUpper = -scale * splitOperator.secondUpper(center);
            output[center] = (lineForward_[second]
                - matrixUpper * output[grid.index(first, second + 1)]) / secondPivots_[center];
        }
    }
}

void AdiSolver::step(const SplitOperator2D& splitOperator,
                     const Surface2D<double>& input,
                     Surface2D<double>& output,
                     double startTime,
                     const AdiBoundary& boundary,
                     AdiWorkspace& workspace,
                     AdiScheme scheme)
{
    workspace.validateFor(splitOperator);
    splitOperator.validate();
    if (input.grid() != splitOperator.grid() || output.grid() != splitOperator.grid())
    {
        throw std::invalid_argument("ADI operator, input, and output shapes must agree");
    }
    if (input.data() == output.data())
    {
        throw std::invalid_argument("in-place ADI stepping is unsupported");
    }
    if (!std::isfinite(startTime) || !boundary)
    {
        throw std::invalid_argument("ADI start time and boundary callback must be valid");
    }
    if (!finiteBuffer(input.data(), input.size()))
    {
        throw std::invalid_argument("ADI input surface must be finite");
    }

    const std::size_t count = splitOperator.size();
    const double timeStep = workspace.timeStep_;
    const double theta = workspace.theta_;
    const double endTime = startTime + timeStep;
    if (!std::isfinite(endTime))
    {
        throw std::invalid_argument("ADI end time must be finite");
    }

    // MCS equations (In 't Hout & Welfert notation, time-independent F):
    // Y0=U+dt*F(U); (I-theta*dt*Fj)Yj=Yj-1-theta*dt*Fj(U);
    // Yhat0=Y0+theta*dt*F0(Y2-U);
    // Ytilde0=Yhat0+(1/2-theta)*dt*F(Y2-U);
    // (I-theta*dt*Fj)Ytildej=Ytildej-1-theta*dt*Fj(U), j=1,2.
    splitOperator.applyFull(input.data(), workspace.operatorWork_.data(), count);
    for (std::size_t index = 0; index < count; ++index)
    {
        workspace.y0_[index] = input.data()[index] + timeStep * workspace.operatorWork_[index];
    }
    applyBoundary(boundary, endTime, splitOperator.grid(), workspace.y0_);

    splitOperator.applyFirst(input.data(), workspace.operatorWork_.data(), count);
    for (std::size_t index = 0; index < count; ++index)
    {
        workspace.rightHandSide_[index] = workspace.y0_[index]
            - theta * timeStep * workspace.operatorWork_[index];
        workspace.y1_[index] = workspace.rightHandSide_[index];
    }
    applyBoundary(boundary, endTime, splitOperator.grid(), workspace.y1_);
    workspace.solveFirst(splitOperator, workspace.rightHandSide_.data(), workspace.y1_.data());
    requireFinite(workspace.y1_, "first directional solve");

    splitOperator.applySecond(input.data(), workspace.operatorWork_.data(), count);
    for (std::size_t index = 0; index < count; ++index)
    {
        workspace.rightHandSide_[index] = workspace.y1_[index]
            - theta * timeStep * workspace.operatorWork_[index];
        workspace.y2_[index] = workspace.rightHandSide_[index];
    }
    applyBoundary(boundary, endTime, splitOperator.grid(), workspace.y2_);
    workspace.solveSecond(splitOperator, workspace.rightHandSide_.data(), workspace.y2_.data());
    requireFinite(workspace.y2_, "second directional solve");

    if (scheme == AdiScheme::ModifiedCraigSneyd)
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            workspace.difference_[index] = workspace.y2_[index] - input.data()[index];
        }
        splitOperator.applyExplicit(workspace.difference_.data(),
                                    workspace.operatorWork_.data(), count);
        for (std::size_t index = 0; index < count; ++index)
        {
            workspace.hat_[index] = workspace.y0_[index]
                + theta * timeStep * workspace.operatorWork_[index];
        }
        applyBoundary(boundary, endTime, splitOperator.grid(), workspace.hat_);

        splitOperator.applyFull(workspace.difference_.data(),
                                workspace.operatorWork_.data(), count);
        for (std::size_t index = 0; index < count; ++index)
        {
            workspace.corrected_[index] = workspace.hat_[index]
                + (0.5 - theta) * timeStep * workspace.operatorWork_[index];
        }
        applyBoundary(boundary, endTime, splitOperator.grid(), workspace.corrected_);

        splitOperator.applyFirst(input.data(), workspace.operatorWork_.data(), count);
        for (std::size_t index = 0; index < count; ++index)
        {
            workspace.rightHandSide_[index] = workspace.corrected_[index]
                - theta * timeStep * workspace.operatorWork_[index];
            workspace.y1_[index] = workspace.rightHandSide_[index];
        }
        applyBoundary(boundary, endTime, splitOperator.grid(), workspace.y1_);
        workspace.solveFirst(splitOperator, workspace.rightHandSide_.data(), workspace.y1_.data());
        requireFinite(workspace.y1_, "MCS corrected first solve");

        splitOperator.applySecond(input.data(), workspace.operatorWork_.data(), count);
        for (std::size_t index = 0; index < count; ++index)
        {
            workspace.rightHandSide_[index] = workspace.y1_[index]
                - theta * timeStep * workspace.operatorWork_[index];
            workspace.y2_[index] = workspace.rightHandSide_[index];
        }
        applyBoundary(boundary, endTime, splitOperator.grid(), workspace.y2_);
        workspace.solveSecond(splitOperator, workspace.rightHandSide_.data(), workspace.y2_.data());
        requireFinite(workspace.y2_, "MCS corrected second solve");
    }

    std::copy(workspace.y2_.begin(), workspace.y2_.end(), output.data());
}

void AdiSolver::rannacherStep(const SplitOperator2D& splitOperator,
                              const Surface2D<double>& input,
                              Surface2D<double>& scratch,
                              Surface2D<double>& output,
                              double startTime,
                              const AdiBoundary& boundary,
                              AdiWorkspace& halfStepDouglasWorkspace)
{
    if (!halfStepDouglasWorkspace.initialized()
        || std::abs(halfStepDouglasWorkspace.theta() - 1.0) > 1.0e-15)
    {
        throw std::invalid_argument(
            "Rannacher startup requires an initialized theta=1 half-step workspace");
    }
    if (scratch.grid() != splitOperator.grid() || output.grid() != splitOperator.grid()
        || scratch.data() == input.data() || output.data() == scratch.data())
    {
        throw std::invalid_argument("Rannacher surfaces must be shape-compatible and disjoint");
    }
    step(splitOperator, input, scratch, startTime, boundary,
         halfStepDouglasWorkspace, AdiScheme::Douglas);
    step(splitOperator, scratch, output,
         startTime + halfStepDouglasWorkspace.timeStep(), boundary,
         halfStepDouglasWorkspace, AdiScheme::Douglas);
}

} // namespace dr3::advanced
