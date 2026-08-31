#pragma once

#include "dr3/advanced/grid2d.h"

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace dr3::advanced
{

class SplitOperator2D
{
public:
    explicit SplitOperator2D(Grid2D grid);

    const Grid2D& grid() const noexcept { return grid_; }
    std::size_t size() const noexcept { return grid_.size(); }

    double& explicitCoefficient(int firstOffset, int secondOffset, std::size_t logicalIndex);
    double explicitCoefficient(int firstOffset, int secondOffset,
                               std::size_t logicalIndex) const;
    double& firstLower(std::size_t logicalIndex) { return firstLower_.at(logicalIndex); }
    double& firstDiagonal(std::size_t logicalIndex) { return firstDiagonal_.at(logicalIndex); }
    double& firstUpper(std::size_t logicalIndex) { return firstUpper_.at(logicalIndex); }
    double& secondLower(std::size_t logicalIndex) { return secondLower_.at(logicalIndex); }
    double& secondDiagonal(std::size_t logicalIndex) { return secondDiagonal_.at(logicalIndex); }
    double& secondUpper(std::size_t logicalIndex) { return secondUpper_.at(logicalIndex); }

    double firstLower(std::size_t logicalIndex) const { return firstLower_.at(logicalIndex); }
    double firstDiagonal(std::size_t logicalIndex) const { return firstDiagonal_.at(logicalIndex); }
    double firstUpper(std::size_t logicalIndex) const { return firstUpper_.at(logicalIndex); }
    double secondLower(std::size_t logicalIndex) const { return secondLower_.at(logicalIndex); }
    double secondDiagonal(std::size_t logicalIndex) const { return secondDiagonal_.at(logicalIndex); }
    double secondUpper(std::size_t logicalIndex) const { return secondUpper_.at(logicalIndex); }

    void validate() const;
    void applyExplicit(const double* input, double* output, std::size_t count) const;
    void applyFirst(const double* input, double* output, std::size_t count) const;
    void applySecond(const double* input, double* output, std::size_t count) const;
    void applyFull(const double* input, double* output, std::size_t count) const;

private:
    static std::size_t explicitSlot(int firstOffset, int secondOffset);
    void validateBuffers(const double* input, double* output, std::size_t count) const;

    Grid2D grid_;
    std::array<std::vector<double>, 9> explicit_;
    std::vector<double> firstLower_;
    std::vector<double> firstDiagonal_;
    std::vector<double> firstUpper_;
    std::vector<double> secondLower_;
    std::vector<double> secondDiagonal_;
    std::vector<double> secondUpper_;
};

enum class AdiScheme
{
    Douglas,
    ModifiedCraigSneyd
};

using AdiBoundary = std::function<void(double, const Grid2D&, double*)>;

class AdiWorkspace
{
public:
    AdiWorkspace() = default;

    void initialize(const SplitOperator2D& splitOperator,
                    double timeStep,
                    double theta = 1.0 / 3.0,
                    double pivotTolerance = 1.0e-13);

    bool initialized() const noexcept { return initialized_; }
    double timeStep() const noexcept { return timeStep_; }
    double theta() const noexcept { return theta_; }
    std::size_t factorizationBuildCount() const noexcept { return factorizationBuildCount_; }

private:
    friend class AdiSolver;

    void validateFor(const SplitOperator2D& splitOperator) const;
    void solveFirst(const SplitOperator2D& splitOperator,
                    const double* rightHandSide,
                    double* output) noexcept;
    void solveSecond(const SplitOperator2D& splitOperator,
                     const double* rightHandSide,
                     double* output) noexcept;

    const SplitOperator2D* operatorIdentity_{};
    bool initialized_{};
    double timeStep_{};
    double theta_{};
    double pivotTolerance_{};
    std::size_t factorizationBuildCount_{};
    std::vector<double> firstPivots_;
    std::vector<double> firstMultipliers_;
    std::vector<double> secondPivots_;
    std::vector<double> secondMultipliers_;
    std::vector<double> lineForward_;
    std::vector<double> y0_;
    std::vector<double> y1_;
    std::vector<double> y2_;
    std::vector<double> hat_;
    std::vector<double> corrected_;
    std::vector<double> difference_;
    std::vector<double> operatorWork_;
    std::vector<double> rightHandSide_;
};

class AdiSolver
{
public:
    static void step(const SplitOperator2D& splitOperator,
                     const Surface2D<double>& input,
                     Surface2D<double>& output,
                     double startTime,
                     const AdiBoundary& boundary,
                     AdiWorkspace& workspace,
                     AdiScheme scheme = AdiScheme::ModifiedCraigSneyd);

    static void rannacherStep(const SplitOperator2D& splitOperator,
                              const Surface2D<double>& input,
                              Surface2D<double>& scratch,
                              Surface2D<double>& output,
                              double startTime,
                              const AdiBoundary& boundary,
                              AdiWorkspace& halfStepDouglasWorkspace);
};

} // namespace dr3::advanced
