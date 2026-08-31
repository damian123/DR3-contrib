#pragma once

#include "european_pde.h"

#include <cstddef>
#include <vector>

namespace dr3::lattice
{

struct PsorConfig
{
    double omega{1.2};
    double tolerance{1.0e-9};
    std::size_t maximumIterations{10'000};
};

struct AmericanPdeResult
{
    double price;
    bool converged;
    std::size_t iterations;
    double residual;
    Grid1D grid;
    std::vector<double> values;
};

AmericanPdeResult americanPdePrice(const VanillaOption& option,
                                   const PdeConfig& config,
                                   const PsorConfig& psor = {});

} // namespace dr3::lattice
