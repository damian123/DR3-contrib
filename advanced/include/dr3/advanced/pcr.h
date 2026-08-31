#pragma once

#include <cstddef>
#include <vector>

namespace dr3::advanced
{

struct PcrSystem
{
    std::vector<double> lower;
    std::vector<double> diagonal;
    std::vector<double> upper;

    std::size_t size() const noexcept { return diagonal.size(); }
    void validate() const;
};

class PcrWorkspace
{
public:
    explicit PcrWorkspace(std::size_t logicalSize);

    std::size_t logicalSize() const noexcept { return logicalSize_; }
    std::size_t paddedSize() const noexcept { return paddedSize_; }
    std::size_t stageCount() const noexcept { return stageCount_; }

private:
    friend class PcrSolver;
    static std::size_t checkedNextPowerOfTwo(std::size_t value);

    std::size_t logicalSize_{};
    std::size_t paddedSize_{};
    std::size_t stageCount_{};
    std::vector<double> lower0_;
    std::vector<double> diagonal0_;
    std::vector<double> upper0_;
    std::vector<double> right0_;
    std::vector<double> lower1_;
    std::vector<double> diagonal1_;
    std::vector<double> upper1_;
    std::vector<double> right1_;
};

class PcrSolver
{
public:
    // Non-power-of-two systems are padded to the next power of two with
    // independent identity rows. Those rows cannot alter the logical solution.
    static void solve(const PcrSystem& system,
                      const double* rightHandSide,
                      std::size_t rightHandSideSize,
                      double* output,
                      std::size_t outputSize,
                      PcrWorkspace& workspace,
                      bool useSimdStages = true,
                      double denominatorTolerance = 1.0e-13);

    static void solve(const PcrSystem& system,
                      const std::vector<double>& rightHandSide,
                      std::vector<double>& output,
                      PcrWorkspace& workspace,
                      bool useSimdStages = true,
                      double denominatorTolerance = 1.0e-13)
    {
        solve(system, rightHandSide.data(), rightHandSide.size(),
              output.data(), output.size(), workspace,
              useSimdStages, denominatorTolerance);
    }

    static double infinityResidual(const PcrSystem& system,
                                   const std::vector<double>& rightHandSide,
                                   const std::vector<double>& solution);
};

} // namespace dr3::advanced
