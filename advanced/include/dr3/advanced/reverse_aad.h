#pragma once

#include "Vectorisation/VecX/vcl_latest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace dr3::advanced::aad
{

template <class Value>
struct ReverseTraits;

template <>
struct ReverseTraits<double>
{
    static constexpr std::size_t lanes = 1;
    static double constant(double value) noexcept { return value; }
    static double zero() noexcept { return 0.0; }
    static double one() noexcept { return 1.0; }
    static bool finite(double value) noexcept { return std::isfinite(value); }
    static double exp(double value) noexcept { return std::exp(value); }
    static double log(double value) noexcept { return std::log(value); }
    static double sqrt(double value) noexcept { return std::sqrt(value); }
    static double sin(double value) noexcept { return std::sin(value); }
    static double cos(double value) noexcept { return std::cos(value); }
    static double pow(double value, double exponent) noexcept
    {
        return std::pow(value, exponent);
    }
    static double normalPdf(double value) noexcept
    {
        constexpr double inverseSqrtTwoPi = 0.39894228040143267793994605993438;
        return inverseSqrtTwoPi * std::exp(-0.5 * value * value);
    }
    static double normalCdf(double value) noexcept
    {
        return 0.5 * std::erfc(-value / std::sqrt(2.0));
    }
    static double mask(double value, std::size_t validLanes) noexcept
    {
        return validLanes == 0 ? 0.0 : value;
    }
    static double lane(double value, std::size_t index)
    {
        if (index != 0)
        {
            throw std::out_of_range("scalar reverse value only has one lane");
        }
        return value;
    }
};

// AVX2 baseline specialization. A tape node owns exactly one Vec4d register.
template <>
struct ReverseTraits<Vec4d>
{
    static constexpr std::size_t lanes = 4;
    static Vec4d constant(double value) noexcept { return Vec4d(value); }
    static Vec4d zero() noexcept { return Vec4d(0.0); }
    static Vec4d one() noexcept { return Vec4d(1.0); }

    template <class Function>
    static Vec4d map(Vec4d value, Function function) noexcept
    {
        alignas(32) double values[4];
        value.store(values);
        for (double& laneValue : values)
        {
            laneValue = function(laneValue);
        }
        return Vec4d().load(values);
    }

    static bool finite(Vec4d value) noexcept
    {
        alignas(32) double values[4];
        value.store(values);
        return std::all_of(values, values + 4,
                           [](double laneValue) { return std::isfinite(laneValue); });
    }
    static Vec4d exp(Vec4d value) noexcept
    {
        return map(value, [](double x) { return std::exp(x); });
    }
    static Vec4d log(Vec4d value) noexcept
    {
        return map(value, [](double x) { return std::log(x); });
    }
    static Vec4d sqrt(Vec4d value) noexcept
    {
        return map(value, [](double x) { return std::sqrt(x); });
    }
    static Vec4d sin(Vec4d value) noexcept
    {
        return map(value, [](double x) { return std::sin(x); });
    }
    static Vec4d cos(Vec4d value) noexcept
    {
        return map(value, [](double x) { return std::cos(x); });
    }
    static Vec4d pow(Vec4d value, double exponent) noexcept
    {
        return map(value, [exponent](double x) { return std::pow(x, exponent); });
    }
    static Vec4d normalPdf(Vec4d value) noexcept
    {
        return map(value, [](double x)
        {
            constexpr double inverseSqrtTwoPi = 0.39894228040143267793994605993438;
            return inverseSqrtTwoPi * std::exp(-0.5 * x * x);
        });
    }
    static Vec4d normalCdf(Vec4d value) noexcept
    {
        return map(value, [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); });
    }
    static Vec4d mask(Vec4d value, std::size_t validLanes)
    {
        if (validLanes > lanes)
        {
            throw std::invalid_argument("valid SIMD lane count exceeds register width");
        }
        alignas(32) double values[4];
        value.store(values);
        for (std::size_t laneIndex = validLanes; laneIndex < lanes; ++laneIndex)
        {
            values[laneIndex] = 0.0;
        }
        return Vec4d().load(values);
    }
    static double lane(Vec4d value, std::size_t index)
    {
        if (index >= lanes)
        {
            throw std::out_of_range("SIMD lane is outside the register");
        }
        alignas(32) double values[4];
        value.store(values);
        return values[index];
    }
};

enum class SweepMode
{
    ResetAdjoints,
    AccumulateExistingAdjoints
};

template <class Value, class Traits = ReverseTraits<Value>>
class ReverseTape
{
public:
    class Active;

    struct Mark
    {
        std::size_t nodeCount{};
        const ReverseTape* tape{};
    };

    ReverseTape() = default;
    ReverseTape(const ReverseTape&) = delete;
    ReverseTape& operator=(const ReverseTape&) = delete;
    ReverseTape(ReverseTape&&) = delete;
    ReverseTape& operator=(ReverseTape&&) = delete;

    void reserve(std::size_t nodeCapacity) { nodes_.reserve(nodeCapacity); }
    std::size_t size() const noexcept { return nodes_.size(); }
    std::size_t capacity() const noexcept { return nodes_.capacity(); }
    const void* identity() const noexcept { return this; }

    Active variable(const Value& primal)
    {
        return append(primal, noNode, noNode, Traits::zero(), Traits::zero());
    }

    Active constant(const Value& primal) const
    {
        return Active(const_cast<ReverseTape*>(this), noNode, 0, primal);
    }

    Mark mark() const noexcept { return Mark{nodes_.size(), this}; }

    void rewind(Mark tapeMark)
    {
        if (tapeMark.tape != this || tapeMark.nodeCount > nodes_.size())
        {
            throw std::invalid_argument("reverse tape mark does not belong to this tape");
        }
        nodes_.resize(tapeMark.nodeCount);
    }

    void clear() noexcept { nodes_.clear(); }

    void zeroAdjoints() noexcept
    {
        for (Node& node : nodes_)
        {
            node.adjoint = Traits::zero();
        }
    }

    // ResetAdjoints is the deterministic default. AccumulateExistingAdjoints
    // deliberately propagates the complete existing state plus the new seed.
    // Call zeroAdjoints() when independent accumulated sweeps are desired.
    void reverse(const Active& output,
                 const Value& seed = Traits::one(),
                 SweepMode mode = SweepMode::ResetAdjoints)
    {
        validate(output, true);
        if (!Traits::finite(seed))
        {
            throw std::invalid_argument("reverse output seed must be finite");
        }
        if (mode == SweepMode::ResetAdjoints)
        {
            zeroAdjoints();
        }
        nodes_[output.nodeIndex_].adjoint = nodes_[output.nodeIndex_].adjoint + seed;
        for (std::size_t index = nodes_.size(); index-- > 0;)
        {
            const Node& node = nodes_[index];
            if (node.parent0 != noNode)
            {
                nodes_[node.parent0].adjoint = nodes_[node.parent0].adjoint
                    + node.adjoint * node.partial0;
            }
            if (node.parent1 != noNode)
            {
                nodes_[node.parent1].adjoint = nodes_[node.parent1].adjoint
                    + node.adjoint * node.partial1;
            }
        }
    }

    // Masks the output seed, so padded tail lanes cannot propagate an adjoint.
    void reverse(const Active& output,
                 const Value& seed,
                 std::size_t validLanes,
                 SweepMode mode = SweepMode::ResetAdjoints)
    {
        reverse(output, Traits::mask(seed, validLanes), mode);
    }

    Value adjoint(const Active& active) const
    {
        validate(active, true);
        return nodes_[active.nodeIndex_].adjoint;
    }

    class Active
    {
    public:
        Active() = default;

        const Value& primal() const noexcept { return primal_; }
        const void* tapeIdentity() const noexcept { return tapeIdentity_; }
        std::size_t nodeIndex() const noexcept { return nodeIndex_; }
        std::uint64_t generation() const noexcept { return generation_; }
        bool isConstant() const noexcept { return nodeIndex_ == noNode; }

        friend Active operator+(const Active& left, const Active& right)
        {
            ReverseTape& tape = compatibleTape(left, right);
            return tape.binary(left, right, left.primal_ + right.primal_,
                               Traits::one(), Traits::one());
        }
        friend Active operator-(const Active& left, const Active& right)
        {
            ReverseTape& tape = compatibleTape(left, right);
            return tape.binary(left, right, left.primal_ - right.primal_,
                               Traits::one(), -Traits::one());
        }
        friend Active operator*(const Active& left, const Active& right)
        {
            ReverseTape& tape = compatibleTape(left, right);
            return tape.binary(left, right, left.primal_ * right.primal_,
                               right.primal_, left.primal_);
        }
        friend Active operator/(const Active& left, const Active& right)
        {
            ReverseTape& tape = compatibleTape(left, right);
            return tape.binary(left, right, left.primal_ / right.primal_,
                               Traits::one() / right.primal_,
                               -left.primal_ / (right.primal_ * right.primal_));
        }
        friend Active operator-(const Active& argument)
        {
            ReverseTape& tape = argument.usableTape();
            return tape.unary(argument, -argument.primal_, -Traits::one());
        }

        friend Active operator+(const Active& left, double right)
        {
            return left + left.usableTape().constant(right);
        }
        friend Active operator+(double left, const Active& right) { return right + left; }
        friend Active operator-(const Active& left, double right)
        {
            return left - left.usableTape().constant(right);
        }
        friend Active operator-(double left, const Active& right)
        {
            return right.usableTape().constant(left) - right;
        }
        friend Active operator*(const Active& left, double right)
        {
            return left * left.usableTape().constant(right);
        }
        friend Active operator*(double left, const Active& right) { return right * left; }
        friend Active operator/(const Active& left, double right)
        {
            return left / left.usableTape().constant(right);
        }
        friend Active operator/(double left, const Active& right)
        {
            return right.usableTape().constant(left) / right;
        }

        friend Active exp(const Active& argument)
        {
            ReverseTape& tape = argument.usableTape();
            const Value primal = Traits::exp(argument.primal_);
            return tape.unary(argument, primal, primal);
        }
        friend Active log(const Active& argument)
        {
            ReverseTape& tape = argument.usableTape();
            return tape.unary(argument, Traits::log(argument.primal_),
                              Traits::one() / argument.primal_);
        }
        friend Active sqrt(const Active& argument)
        {
            ReverseTape& tape = argument.usableTape();
            const Value primal = Traits::sqrt(argument.primal_);
            return tape.unary(argument, primal, Traits::constant(0.5) / primal);
        }
        friend Active sin(const Active& argument)
        {
            ReverseTape& tape = argument.usableTape();
            return tape.unary(argument, Traits::sin(argument.primal_),
                              Traits::cos(argument.primal_));
        }
        friend Active cos(const Active& argument)
        {
            ReverseTape& tape = argument.usableTape();
            return tape.unary(argument, Traits::cos(argument.primal_),
                              -Traits::sin(argument.primal_));
        }
        friend Active pow(const Active& argument, double exponent)
        {
            ReverseTape& tape = argument.usableTape();
            const Value primal = Traits::pow(argument.primal_, exponent);
            const Value derivative = Traits::constant(exponent)
                * Traits::pow(argument.primal_, exponent - 1.0);
            return tape.unary(argument, primal, derivative);
        }
        friend Active normalCdf(const Active& argument)
        {
            ReverseTape& tape = argument.usableTape();
            return tape.unary(argument, Traits::normalCdf(argument.primal_),
                              Traits::normalPdf(argument.primal_));
        }

    private:
        friend class ReverseTape;

        Active(ReverseTape* tape, std::size_t nodeIndex,
               std::uint64_t generation, const Value& primal)
            : tape_(tape), tapeIdentity_(tape), nodeIndex_(nodeIndex),
              generation_(generation), primal_(primal)
        {
        }

        ReverseTape& usableTape() const
        {
            if (tape_ == nullptr || tapeIdentity_ != tape_)
            {
                throw std::logic_error("reverse active value has no tape");
            }
            tape_->validate(*this, false);
            return *tape_;
        }

        static ReverseTape& compatibleTape(const Active& left, const Active& right)
        {
            ReverseTape& tape = left.usableTape();
            right.usableTape();
            if (left.tapeIdentity_ != right.tapeIdentity_)
            {
                throw std::invalid_argument("reverse operands belong to different tapes");
            }
            return tape;
        }

        ReverseTape* tape_{};
        const void* tapeIdentity_{};
        std::size_t nodeIndex_{noNode};
        std::uint64_t generation_{};
        Value primal_{Traits::zero()};
    };

private:
    static constexpr std::size_t noNode = std::numeric_limits<std::size_t>::max();

    struct Node
    {
        Value adjoint{Traits::zero()};
        Value partial0{Traits::zero()};
        Value partial1{Traits::zero()};
        std::size_t parent0{noNode};
        std::size_t parent1{noNode};
        std::uint64_t generation{};
    };

    Active append(const Value& primal, std::size_t parent0, std::size_t parent1,
                  const Value& partial0, const Value& partial1)
    {
        if (nextGeneration_ == 0)
        {
            throw std::overflow_error("reverse tape generation counter exhausted");
        }
        const std::uint64_t generation = nextGeneration_++;
        nodes_.push_back(Node{Traits::zero(), partial0, partial1,
                              parent0, parent1, generation});
        return Active(this, nodes_.size() - 1, generation, primal);
    }

    Active unary(const Active& argument, const Value& primal, const Value& partial)
    {
        validate(argument, false);
        if (argument.nodeIndex_ == noNode)
        {
            return constant(primal);
        }
        return append(primal, argument.nodeIndex_, noNode, partial, Traits::zero());
    }

    Active binary(const Active& left, const Active& right, const Value& primal,
                  const Value& leftPartial, const Value& rightPartial)
    {
        validate(left, false);
        validate(right, false);
        if (left.tapeIdentity_ != right.tapeIdentity_)
        {
            throw std::invalid_argument("reverse operands belong to different tapes");
        }
        if (left.nodeIndex_ == noNode && right.nodeIndex_ == noNode)
        {
            return constant(primal);
        }
        return append(primal, left.nodeIndex_, right.nodeIndex_,
                      leftPartial, rightPartial);
    }

    void validate(const Active& active, bool requireNode) const
    {
        if (active.tape_ != this || active.tapeIdentity_ != this)
        {
            throw std::invalid_argument("reverse active value belongs to another tape");
        }
        if (active.nodeIndex_ == noNode)
        {
            if (requireNode)
            {
                throw std::invalid_argument("reverse output or adjoint must name a tape node");
            }
            return;
        }
        if (active.nodeIndex_ >= nodes_.size()
            || nodes_[active.nodeIndex_].generation != active.generation_)
        {
            throw std::logic_error("reverse active value is stale after rewind or clear");
        }
    }

    std::vector<Node> nodes_;
    std::uint64_t nextGeneration_{1};
};

using ScalarTape = ReverseTape<double>;
using SimdTape = ReverseTape<Vec4d>;

} // namespace dr3::advanced::aad
