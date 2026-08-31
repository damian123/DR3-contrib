#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dr3::advanced
{

struct IndexRange
{
    std::size_t begin{};
    std::size_t end{};
    std::size_t size() const noexcept { return end - begin; }
    bool contains(std::size_t index) const noexcept { return index >= begin && index < end; }
};

class Axis1D
{
public:
    explicit Axis1D(std::vector<double> coordinates)
        : coordinates_(std::move(coordinates))
    {
        if (coordinates_.size() < 3)
        {
            throw std::invalid_argument("a Grid2D axis requires at least three coordinates");
        }
        for (std::size_t index = 0; index < coordinates_.size(); ++index)
        {
            if (!std::isfinite(coordinates_[index]))
            {
                throw std::invalid_argument("Grid2D axis coordinates must be finite");
            }
            if (index != 0)
            {
                const double spacing = coordinates_[index] - coordinates_[index - 1];
                if (!(coordinates_[index] > coordinates_[index - 1])
                    || !std::isfinite(spacing) || spacing == 0.0)
                {
                    throw std::invalid_argument(
                        "Grid2D axis coordinates must be strictly increasing with finite spacing");
                }
            }
        }
    }

    static Axis1D uniform(double first, double last, std::size_t count)
    {
        if (count < 3 || !std::isfinite(first) || !std::isfinite(last) || !(last > first))
        {
            throw std::invalid_argument("uniform Grid2D axis bounds/count are invalid");
        }
        std::vector<double> coordinates(count);
        const double spacing = (last - first) / static_cast<double>(count - 1);
        if (!std::isfinite(spacing) || spacing <= 0.0)
        {
            throw std::invalid_argument("uniform Grid2D axis spacing must be finite and positive");
        }
        for (std::size_t index = 0; index < count; ++index)
        {
            coordinates[index] = first + spacing * static_cast<double>(index);
        }
        coordinates.back() = last;
        return Axis1D(std::move(coordinates));
    }

    std::size_t size() const noexcept { return coordinates_.size(); }
    double operator[](std::size_t index) const { return coordinates_.at(index); }
    const std::vector<double>& coordinates() const noexcept { return coordinates_; }

    friend bool operator==(const Axis1D& left, const Axis1D& right) noexcept
    {
        return left.coordinates_ == right.coordinates_;
    }
    friend bool operator!=(const Axis1D& left, const Axis1D& right) noexcept
    {
        return !(left == right);
    }

private:
    std::vector<double> coordinates_;
};

class Grid2D
{
public:
    Grid2D(Axis1D firstCoordinate, Axis1D secondCoordinate)
        : first_(std::move(firstCoordinate)), second_(std::move(secondCoordinate)),
          cellCount_(checkedCellCount(first_.size(), second_.size()))
    {
    }

    static std::size_t checkedCellCount(std::size_t firstCount, std::size_t secondCount)
    {
        if (firstCount != 0
            && secondCount > std::numeric_limits<std::size_t>::max() / firstCount)
        {
            throw std::overflow_error("Grid2D dimension multiplication overflows size_t");
        }
        return firstCount * secondCount;
    }

    std::size_t firstSize() const noexcept { return first_.size(); }
    std::size_t secondSize() const noexcept { return second_.size(); }
    std::size_t size() const noexcept { return cellCount_; }
    const Axis1D& firstAxis() const noexcept { return first_; }
    const Axis1D& secondAxis() const noexcept { return second_; }

    // The first coordinate is contiguous: index(i,j) == j * firstSize() + i.
    std::size_t index(std::size_t firstIndex, std::size_t secondIndex) const
    {
        if (firstIndex >= firstSize() || secondIndex >= secondSize())
        {
            throw std::out_of_range("Grid2D logical index is outside the grid");
        }
        return secondIndex * firstSize() + firstIndex;
    }

    std::pair<std::size_t, std::size_t> coordinates(std::size_t logicalIndex) const
    {
        if (logicalIndex >= cellCount_)
        {
            throw std::out_of_range("Grid2D storage index is outside the logical grid");
        }
        return {logicalIndex % firstSize(), logicalIndex / firstSize()};
    }

    IndexRange interiorFirst() const noexcept { return {1, firstSize() - 1}; }
    IndexRange interiorSecond() const noexcept { return {1, secondSize() - 1}; }
    IndexRange fullFirst() const noexcept { return {0, firstSize()}; }
    IndexRange fullSecond() const noexcept { return {0, secondSize()}; }

    std::array<std::size_t, 4> corners() const
    {
        return {index(0, 0), index(firstSize() - 1, 0),
                index(0, secondSize() - 1), index(firstSize() - 1, secondSize() - 1)};
    }

    bool isBoundary(std::size_t firstIndex, std::size_t secondIndex) const
    {
        (void)index(firstIndex, secondIndex);
        return firstIndex == 0 || secondIndex == 0
            || firstIndex + 1 == firstSize() || secondIndex + 1 == secondSize();
    }

    friend bool operator==(const Grid2D& left, const Grid2D& right) noexcept
    {
        return left.first_ == right.first_ && left.second_ == right.second_;
    }
    friend bool operator!=(const Grid2D& left, const Grid2D& right) noexcept
    {
        return !(left == right);
    }

private:
    Axis1D first_;
    Axis1D second_;
    std::size_t cellCount_;
};

template <class Value>
class Surface2D
{
public:
    explicit Surface2D(Grid2D grid) : grid_(std::move(grid)), values_(grid_.size()) {}
    Surface2D(Grid2D grid, const Value& initialValue)
        : grid_(std::move(grid)), values_(grid_.size(), initialValue)
    {
    }

    const Grid2D& grid() const noexcept { return grid_; }
    std::size_t size() const noexcept { return values_.size(); }
    Value* data() noexcept { return values_.data(); }
    const Value* data() const noexcept { return values_.data(); }

    Value& operator()(std::size_t firstIndex, std::size_t secondIndex)
    {
        return values_[grid_.index(firstIndex, secondIndex)];
    }
    const Value& operator()(std::size_t firstIndex, std::size_t secondIndex) const
    {
        return values_[grid_.index(firstIndex, secondIndex)];
    }
    Value& atLogical(std::size_t logicalIndex) { return values_.at(logicalIndex); }
    const Value& atLogical(std::size_t logicalIndex) const { return values_.at(logicalIndex); }
    std::vector<Value>& values() noexcept { return values_; }
    const std::vector<Value>& values() const noexcept { return values_; }

private:
    Grid2D grid_;
    std::vector<Value> values_;
};

namespace stencil2d
{

inline void validate(const Surface2D<double>& input, const Surface2D<double>& output)
{
    if (input.grid() != output.grid())
    {
        throw std::invalid_argument("2D stencil input/output shapes must match");
    }
    if (input.data() == output.data())
    {
        throw std::invalid_argument("in-place 2D stencil evaluation is unsupported");
    }
}

inline std::array<double, 3> firstWeights(const Axis1D& axis, std::size_t index)
{
    const double left = axis[index] - axis[index - 1];
    const double right = axis[index + 1] - axis[index];
    const double total = left + right;
    if (!std::isfinite(left) || !std::isfinite(right) || !std::isfinite(total)
        || left <= 0.0 || right <= 0.0 || total == 0.0)
    {
        throw std::domain_error("nonuniform first-derivative denominator is invalid");
    }
    return {-right / (left * total), (right - left) / (left * right),
            left / (right * total)};
}

inline void firstDerivativeFirst(const Surface2D<double>& input, Surface2D<double>& output)
{
    validate(input, output);
    const Grid2D& grid = input.grid();
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            const auto weights = firstWeights(grid.firstAxis(), first);
            output(first, second) = weights[0] * input(first - 1, second)
                + weights[1] * input(first, second)
                + weights[2] * input(first + 1, second);
        }
    }
}

inline void firstDerivativeSecond(const Surface2D<double>& input, Surface2D<double>& output)
{
    validate(input, output);
    const Grid2D& grid = input.grid();
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        const auto weights = firstWeights(grid.secondAxis(), second);
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            output(first, second) = weights[0] * input(first, second - 1)
                + weights[1] * input(first, second)
                + weights[2] * input(first, second + 1);
        }
    }
}

inline void secondDerivativeFirst(const Surface2D<double>& input, Surface2D<double>& output)
{
    validate(input, output);
    const Grid2D& grid = input.grid();
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            const double left = grid.firstAxis()[first] - grid.firstAxis()[first - 1];
            const double right = grid.firstAxis()[first + 1] - grid.firstAxis()[first];
            const double total = left + right;
            if (!std::isfinite(total) || left <= 0.0 || right <= 0.0)
            {
                throw std::domain_error("nonuniform second-derivative denominator is invalid");
            }
            output(first, second) = 2.0 * (input(first - 1, second) / (left * total)
                - input(first, second) / (left * right)
                + input(first + 1, second) / (right * total));
        }
    }
}

inline void secondDerivativeSecond(const Surface2D<double>& input, Surface2D<double>& output)
{
    validate(input, output);
    const Grid2D& grid = input.grid();
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        const double left = grid.secondAxis()[second] - grid.secondAxis()[second - 1];
        const double right = grid.secondAxis()[second + 1] - grid.secondAxis()[second];
        const double total = left + right;
        if (!std::isfinite(total) || left <= 0.0 || right <= 0.0)
        {
            throw std::domain_error("nonuniform second-derivative denominator is invalid");
        }
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            output(first, second) = 2.0 * (input(first, second - 1) / (left * total)
                - input(first, second) / (left * right)
                + input(first, second + 1) / (right * total));
        }
    }
}

inline void mixedDerivative(const Surface2D<double>& input, Surface2D<double>& output)
{
    validate(input, output);
    const Grid2D& grid = input.grid();
    for (std::size_t second = 1; second + 1 < grid.secondSize(); ++second)
    {
        const auto secondWeights = firstWeights(grid.secondAxis(), second);
        for (std::size_t first = 1; first + 1 < grid.firstSize(); ++first)
        {
            const auto firstWeightsAt = firstWeights(grid.firstAxis(), first);
            double derivative = 0.0;
            for (std::size_t secondOffset = 0; secondOffset < 3; ++secondOffset)
            {
                for (std::size_t firstOffset = 0; firstOffset < 3; ++firstOffset)
                {
                    derivative += firstWeightsAt[firstOffset] * secondWeights[secondOffset]
                        * input(first + firstOffset - 1, second + secondOffset - 1);
                }
            }
            output(first, second) = derivative;
        }
    }
}

} // namespace stencil2d
} // namespace dr3::advanced
