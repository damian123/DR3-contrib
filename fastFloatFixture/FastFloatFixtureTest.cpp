#include "VecX/dr3.h"

#include <fast_float/fast_float.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using namespace DRC::VecD4D;

struct Option {
    double spot;
    double strike;
    double time;
    double rate;
    double volatility;
};

std::uint64_t bits(double value)
{
    std::uint64_t result = 0U;
    static_assert(sizeof(result) == sizeof(value), "unexpected double size");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double parse_fast_float(const std::string& token)
{
    double value = 0.0;
    const auto parsed = fast_float::from_chars(
        token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
        throw std::runtime_error("fast_float rejected fixture token: " + token);
    }
    return value;
}

double parse_strtod(const std::string& token)
{
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (errno != 0 || end != token.c_str() + token.size()) {
        throw std::runtime_error("strtod rejected fixture token: " + token);
    }
    return value;
}

std::array<std::string, 5> split_row(const std::string& row)
{
    std::array<std::string, 5> fields;
    std::size_t begin = 0U;
    for (std::size_t field = 0; field < fields.size(); ++field) {
        const std::size_t comma = row.find(',', begin);
        if ((field + 1U < fields.size() && comma == std::string::npos)
            || (field + 1U == fields.size() && comma != std::string::npos)) {
            throw std::runtime_error("fixture row must contain exactly five columns");
        }
        fields[field] = row.substr(begin, comma == std::string::npos
            ? std::string::npos : comma - begin);
        begin = comma == std::string::npos ? row.size() : comma + 1U;
    }
    return fields;
}

std::vector<Option> load_fixture()
{
    std::ifstream input(DR3_FASTFLOAT_FIXTURE_PATH);
    if (!input) throw std::runtime_error("could not open options.csv");
    std::vector<Option> options;
    std::string row;
    while (std::getline(input, row)) {
        if (row.empty()) continue;
        const auto fields = split_row(row);
        std::array<double, 5> parsed{};
        for (std::size_t field = 0; field < fields.size(); ++field) {
            parsed[field] = parse_fast_float(fields[field]);
            const double libc_value = parse_strtod(fields[field]);
            if (bits(parsed[field]) != bits(libc_value)) {
                throw std::runtime_error("fast_float and strtod differ by at least one ULP");
            }
        }
        options.push_back({parsed[0], parsed[1], parsed[2], parsed[3], parsed[4]});
    }
    if (options.empty()) throw std::runtime_error("fixture must not be empty");
    return options;
}

std::vector<double> price_simd(const std::vector<Option>& options)
{
    std::vector<double> spots, strikes, times, rates, volatilities;
    spots.reserve(options.size());
    strikes.reserve(options.size());
    times.reserve(options.size());
    rates.reserve(options.size());
    volatilities.reserve(options.size());
    for (const Option& option : options) {
        spots.push_back(option.spot);
        strikes.push_back(option.strike);
        times.push_back(option.time);
        rates.push_back(option.rate);
        volatilities.push_back(option.volatility);
    }
    const VecXX spot_vector(spots);
    const VecXX strike_vector(strikes);
    const VecXX time_vector(times);
    const VecXX rate_vector(rates);
    const VecXX volatility_vector(volatilities);
    const auto root_time = sqrt(time_vector);
    const auto d1 = (log(spot_vector / strike_vector)
        + (rate_vector + 0.5 * volatility_vector * volatility_vector) * time_vector)
        / (volatility_vector * root_time);
    const auto d2 = d1 - volatility_vector * root_time;
    return static_cast<std::vector<double>>(
        spot_vector * cdfnorm(d1)
        - strike_vector * exp(-rate_vector * time_vector) * cdfnorm(d2));
}

int run_self_test()
{
    const std::vector<Option> parsed = load_fixture();
    const std::vector<Option> in_memory{
        {100, 100, 1, 0.05, 0.20},
        {95.5, 100, 0.5, 0.01, 0.15},
        {123.25, 110, 2.25, -0.005, 0.35},
        {80, 75, 0.125, 0.075, 0.5},
        {250.125, 240.5, 3, 0.025, 0.125},
        {42, 45, 1.75, 0, 0.3},
        {1000, 975.25, 0.01, 0.1, 0.05},
        {67.75, 70, 4.5, 0.0125, 0.225},
        {101.125, 99.875, 0.75, 0.033, 0.19}};
    if (parsed.size() != in_memory.size()) {
        throw std::runtime_error("fixture and in-memory option counts differ");
    }
    const auto parsed_prices = price_simd(parsed);
    const auto in_memory_prices = price_simd(in_memory);
    double max_error = 0.0;
    for (std::size_t i = 0; i < parsed_prices.size(); ++i) {
        max_error = std::max(max_error, std::fabs(parsed_prices[i] - in_memory_prices[i]));
    }
    if (max_error != 0.0) {
        throw std::runtime_error("parsed fixture and in-memory SIMD prices differ");
    }
    std::cout << "parsed " << parsed.size()
              << " options; fast_float_vs_strtod=0 ULP; SIMD_price_max_abs_error="
              << max_error << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        AllAllocatorsGuard<typename VecXX::SCALA_TYPE> allocator_guard;
        if (argc != 2 || std::string(argv[1]) != "--self-test") {
            throw std::invalid_argument("expected --self-test");
        }
        return run_self_test();
    } catch (const std::exception& error) {
        std::cerr << "FastFloatFixtureTest failed: " << error.what() << '\n';
        return 1;
    }
}
