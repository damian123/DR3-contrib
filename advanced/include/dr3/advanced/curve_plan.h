#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dr3::advanced
{

enum class CurveExtrapolation
{
    Flat,
    Linear,
    Reject
};

enum class CurveQueryClass
{
    ExactPillar,
    Interior,
    LeftExtrapolation,
    RightExtrapolation
};

class CurveEvaluationPlan
{
public:
    struct Entry
    {
        std::size_t left{};
        std::size_t right{};
        double rightWeight{};
        CurveQueryClass classification{CurveQueryClass::ExactPillar};
    };

    CurveEvaluationPlan(std::vector<double> pillars,
                        std::vector<double> queryTimes,
                        CurveExtrapolation extrapolation = CurveExtrapolation::Flat)
        : pillars_(std::move(pillars)), queryTimes_(std::move(queryTimes)),
          extrapolation_(extrapolation), signature_(gridSignature(pillars_))
    {
        validatePillars(pillars_);
        entries_.reserve(queryTimes_.size());
        for (const double query : queryTimes_)
        {
            if (!std::isfinite(query))
            {
                throw std::invalid_argument("curve query times must be finite");
            }
            entries_.push_back(classify(query));
        }
    }

    std::size_t size() const noexcept { return entries_.size(); }
    std::uint64_t pillarGridSignature() const noexcept { return signature_; }
    CurveExtrapolation extrapolation() const noexcept { return extrapolation_; }
    const std::vector<double>& queryTimes() const noexcept { return queryTimes_; }
    const Entry& entry(std::size_t index) const { return entries_.at(index); }

    template <class Value>
    void evaluate(const std::vector<double>& pillarGrid,
                  const std::vector<Value>& values,
                  Value* output,
                  std::size_t outputSize) const
    {
        validateEvaluation(pillarGrid, values.size(), output, outputSize);
        for (std::size_t queryIndex = 0; queryIndex < entries_.size(); ++queryIndex)
        {
            const Entry& item = entries_[queryIndex];
            if (item.left == item.right)
            {
                output[queryIndex] = values[item.left];
            }
            else
            {
                output[queryIndex] = values[item.left] * (1.0 - item.rightWeight)
                    + values[item.right] * item.rightWeight;
            }
        }
    }

    template <class Value>
    void evaluate(const std::vector<double>& pillarGrid,
                  const std::vector<Value>& values,
                  std::vector<Value>& output) const
    {
        evaluate(pillarGrid, values, output.data(), output.size());
    }

    template <class Value>
    Value evaluateOne(const std::vector<double>& pillarGrid,
                      const std::vector<Value>& values,
                      std::size_t queryIndex) const
    {
        if (queryIndex >= entries_.size())
        {
            throw std::out_of_range("curve plan query index is outside the plan");
        }
        if (gridSignature(pillarGrid) != signature_ || pillarGrid != pillars_)
        {
            throw std::invalid_argument("curve plan was constructed for a different pillar grid");
        }
        if (values.size() != pillars_.size())
        {
            throw std::invalid_argument("curve values must match the pillar grid");
        }
        const Entry& item = entries_[queryIndex];
        if (item.left == item.right)
        {
            return values[item.left];
        }
        return values[item.left] * (1.0 - item.rightWeight)
            + values[item.right] * item.rightWeight;
    }

    static std::uint64_t gridSignature(const std::vector<double>& pillars) noexcept
    {
        // Versioned FNV-1a over IEEE-754 bytes. Exact vector equality is also
        // checked at evaluation, so a hash collision cannot accept a grid.
        std::uint64_t hash = 1469598103934665603ULL;
        for (double pillar : pillars)
        {
            std::uint64_t bits{};
            static_assert(sizeof(bits) == sizeof(pillar), "unexpected double representation");
            std::memcpy(&bits, &pillar, sizeof(bits));
            for (unsigned byte = 0; byte < 8; ++byte)
            {
                hash ^= (bits >> (byte * 8U)) & 0xffU;
                hash *= 1099511628211ULL;
            }
        }
        hash ^= static_cast<std::uint64_t>(pillars.size());
        hash *= 1099511628211ULL;
        return hash;
    }

private:
    static void validatePillars(const std::vector<double>& pillars)
    {
        if (pillars.size() < 2)
        {
            throw std::invalid_argument("a curve requires at least two pillars");
        }
        for (std::size_t index = 0; index < pillars.size(); ++index)
        {
            if (!std::isfinite(pillars[index]))
            {
                throw std::invalid_argument("curve pillars must be finite");
            }
            if (index != 0 && !(pillars[index] > pillars[index - 1]))
            {
                throw std::invalid_argument("curve pillars must be strictly increasing");
            }
        }
    }

    Entry classify(double query) const
    {
        if (query < pillars_.front())
        {
            if (extrapolation_ == CurveExtrapolation::Reject)
            {
                throw std::out_of_range("curve query is left of the pillar grid");
            }
            if (extrapolation_ == CurveExtrapolation::Flat)
            {
                return Entry{0, 0, 0.0, CurveQueryClass::LeftExtrapolation};
            }
            return interpolationEntry(0, 1, query, CurveQueryClass::LeftExtrapolation);
        }
        if (query > pillars_.back())
        {
            if (extrapolation_ == CurveExtrapolation::Reject)
            {
                throw std::out_of_range("curve query is right of the pillar grid");
            }
            if (extrapolation_ == CurveExtrapolation::Flat)
            {
                const std::size_t last = pillars_.size() - 1;
                return Entry{last, last, 0.0, CurveQueryClass::RightExtrapolation};
            }
            return interpolationEntry(pillars_.size() - 2, pillars_.size() - 1,
                                      query, CurveQueryClass::RightExtrapolation);
        }

        const auto upper = std::lower_bound(pillars_.begin(), pillars_.end(), query);
        const std::size_t right = static_cast<std::size_t>(upper - pillars_.begin());
        if (upper != pillars_.end() && *upper == query)
        {
            return Entry{right, right, 0.0, CurveQueryClass::ExactPillar};
        }
        return interpolationEntry(right - 1, right, query, CurveQueryClass::Interior);
    }

    Entry interpolationEntry(std::size_t left, std::size_t right, double query,
                             CurveQueryClass classification) const
    {
        const double weight = (query - pillars_[left]) / (pillars_[right] - pillars_[left]);
        return Entry{left, right, weight, classification};
    }

    template <class Value>
    void validateEvaluation(const std::vector<double>& pillarGrid,
                            std::size_t valueCount,
                            Value* output,
                            std::size_t outputSize) const
    {
        if (gridSignature(pillarGrid) != signature_ || pillarGrid != pillars_)
        {
            throw std::invalid_argument("curve plan was constructed for a different pillar grid");
        }
        if (valueCount != pillars_.size())
        {
            throw std::invalid_argument("curve values must match the pillar grid");
        }
        if (outputSize != entries_.size() || (output == nullptr && outputSize != 0))
        {
            throw std::invalid_argument("curve plan output must match the query count");
        }
    }

    std::vector<double> pillars_;
    std::vector<double> queryTimes_;
    CurveExtrapolation extrapolation_;
    std::uint64_t signature_;
    std::vector<Entry> entries_;
};

template <class Value>
Value directCurveValue(const std::vector<double>& pillars,
                       const std::vector<Value>& values,
                       double query,
                       CurveExtrapolation extrapolation = CurveExtrapolation::Flat)
{
    CurveEvaluationPlan plan(pillars, {query}, extrapolation);
    return plan.evaluateOne(pillars, values, 0);
}

} // namespace dr3::advanced
