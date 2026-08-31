#include "dr3/advanced/grid2d.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
using dr3::advanced::Axis1D;
using dr3::advanced::Grid2D;
using dr3::advanced::Surface2D;

Grid2D uniformGrid(std::size_t first = 7, std::size_t second = 6)
{
    return Grid2D(Axis1D::uniform(-1.0, 1.0, first),
                  Axis1D::uniform(-2.0, 2.0, second));
}

template <class Function>
Surface2D<double> makeSurface(const Grid2D& grid, Function function)
{
    Surface2D<double> surface(grid);
    for (std::size_t second = 0; second < grid.secondSize(); ++second)
    {
        for (std::size_t first = 0; first < grid.firstSize(); ++first)
        {
            surface(first, second) = function(grid.firstAxis()[first], grid.secondAxis()[second]);
        }
    }
    return surface;
}

TEST(Grid2D, RejectsShortAxis)
{
    EXPECT_THROW(Axis1D(std::vector<double>{0.0, 1.0}), std::invalid_argument);
}

TEST(Grid2D, RejectsNonFiniteAxis)
{
    EXPECT_THROW(Axis1D(std::vector<double>{0.0,
        std::numeric_limits<double>::infinity(), 2.0}), std::invalid_argument);
}

TEST(Grid2D, RejectsUnsortedAxis)
{
    EXPECT_THROW(Axis1D(std::vector<double>{0.0, 2.0, 1.0}), std::invalid_argument);
}

TEST(Grid2D, RejectsDuplicateCoordinates)
{
    EXPECT_THROW(Axis1D(std::vector<double>{0.0, 1.0, 1.0}), std::invalid_argument);
}

TEST(Grid2D, RowMajorIndexing)
{
    const Grid2D grid = uniformGrid(5, 4);
    EXPECT_EQ(grid.index(0, 0), 0U);
    EXPECT_EQ(grid.index(1, 0), 1U);
    EXPECT_EQ(grid.index(0, 1), 5U);
    EXPECT_THROW(grid.index(5, 0), std::out_of_range);
}

TEST(Grid2D, CornerAndBoundaryRanges)
{
    const Grid2D grid = uniformGrid(5, 4);
    EXPECT_EQ(grid.corners(), (std::array<std::size_t, 4>{0, 4, 15, 19}));
    EXPECT_EQ(grid.interiorFirst().size(), 3U);
    EXPECT_EQ(grid.interiorSecond().size(), 2U);
    EXPECT_TRUE(grid.isBoundary(0, 2));
    EXPECT_FALSE(grid.isBoundary(2, 2));
}

TEST(Grid2D, CheckedDimensionOverflow)
{
    EXPECT_THROW(Grid2D::checkedCellCount(std::numeric_limits<std::size_t>::max(), 2),
                 std::overflow_error);
}

TEST(Stencil2D, ConstantHasZeroDerivatives)
{
    const Grid2D grid = uniformGrid();
    const Surface2D<double> input(grid, 7.0);
    Surface2D<double> output(grid, 99.0);
    dr3::advanced::stencil2d::firstDerivativeFirst(input, output);
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
            EXPECT_NEAR(output(first, second), 0.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(output(0, 0), 99.0);
}

TEST(Stencil2D, LinearXFirstDerivative)
{
    const Grid2D grid = uniformGrid();
    const auto input = makeSurface(grid, [](double x, double) { return 3.0 * x + 2.0; });
    Surface2D<double> output(grid);
    dr3::advanced::stencil2d::firstDerivativeFirst(input, output);
    EXPECT_NEAR(output(3, 3), 3.0, 1.0e-13);
}

TEST(Stencil2D, LinearYFirstDerivative)
{
    const Grid2D grid = uniformGrid();
    const auto input = makeSurface(grid, [](double, double y) { return -2.0 * y + 1.0; });
    Surface2D<double> output(grid);
    dr3::advanced::stencil2d::firstDerivativeSecond(input, output);
    EXPECT_NEAR(output(3, 3), -2.0, 1.0e-13);
}

TEST(Stencil2D, QuadraticXSecondDerivative)
{
    const Grid2D grid = uniformGrid();
    const auto input = makeSurface(grid, [](double x, double) { return 4.0 * x * x; });
    Surface2D<double> output(grid);
    dr3::advanced::stencil2d::secondDerivativeFirst(input, output);
    EXPECT_NEAR(output(3, 3), 8.0, 1.0e-12);
}

TEST(Stencil2D, QuadraticYSecondDerivative)
{
    const Grid2D grid = uniformGrid();
    const auto input = makeSurface(grid, [](double, double y) { return -3.0 * y * y; });
    Surface2D<double> output(grid);
    dr3::advanced::stencil2d::secondDerivativeSecond(input, output);
    EXPECT_NEAR(output(3, 3), -6.0, 1.0e-12);
}

TEST(Stencil2D, XTimesYMixedDerivativeIsOne)
{
    const Grid2D grid = uniformGrid();
    const auto input = makeSurface(grid, [](double x, double y) { return x * y; });
    Surface2D<double> output(grid);
    dr3::advanced::stencil2d::mixedDerivative(input, output);
    EXPECT_NEAR(output(3, 3), 1.0, 1.0e-13);
}

TEST(Stencil2D, UniformGridPolynomialAccuracy)
{
    const Grid2D grid = uniformGrid(9, 9);
    const auto input = makeSurface(grid,
        [](double x, double y) { return x * x + 2.0 * x * y + 3.0 * y * y; });
    Surface2D<double> xx(grid), yy(grid), xy(grid);
    dr3::advanced::stencil2d::secondDerivativeFirst(input, xx);
    dr3::advanced::stencil2d::secondDerivativeSecond(input, yy);
    dr3::advanced::stencil2d::mixedDerivative(input, xy);
    EXPECT_NEAR(xx(4, 4), 2.0, 1.0e-12);
    EXPECT_NEAR(yy(4, 4), 6.0, 1.0e-12);
    EXPECT_NEAR(xy(4, 4), 2.0, 1.0e-12);
}

double nonuniformError(std::size_t count)
{
    std::vector<double> x(count), y(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        const double fraction = static_cast<double>(index) / static_cast<double>(count - 1);
        x[index] = fraction * fraction;
        y[index] = fraction * fraction;
    }
    const Grid2D grid{Axis1D(x), Axis1D(y)};
    const auto input = makeSurface(grid, [](double first, double) { return std::sin(first); });
    Surface2D<double> output(grid);
    dr3::advanced::stencil2d::firstDerivativeFirst(input, output);
    double error = 0.0;
    for (std::size_t index = 1; index + 1 < count; ++index)
    {
        error = std::max(error, std::abs(output(index, count / 2) - std::cos(x[index])));
    }
    return error;
}

TEST(Stencil2D, NonuniformGridConverges)
{
    EXPECT_LT(nonuniformError(17), nonuniformError(9));
}

TEST(Stencil2D, RejectsShapeMismatch)
{
    const Surface2D<double> input(uniformGrid(5, 5));
    Surface2D<double> output(uniformGrid(6, 5));
    EXPECT_THROW(dr3::advanced::stencil2d::firstDerivativeFirst(input, output),
                 std::invalid_argument);
}

TEST(Stencil2D, RejectsUnsupportedAliasing)
{
    Surface2D<double> surface(uniformGrid());
    EXPECT_THROW(dr3::advanced::stencil2d::mixedDerivative(surface, surface),
                 std::invalid_argument);
}

TEST(Stencil2D, NoAllocationDuringEvaluation)
{
    const Grid2D grid = uniformGrid(9, 9);
    const auto input = makeSurface(grid, [](double x, double y) { return x * y; });
    Surface2D<double> output(grid);
    const auto* before = output.data();
    const auto capacity = output.values().capacity();
    dr3::advanced::stencil2d::mixedDerivative(input, output);
    EXPECT_EQ(output.data(), before);
    EXPECT_EQ(output.values().capacity(), capacity);
}

} // namespace
