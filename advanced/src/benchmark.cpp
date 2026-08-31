#include "dr3/advanced/benchmark.h"

#include "dr3/advanced/curve_plan.h"
#include "dr3/advanced/heston.h"
#include "dr3/advanced/pcr.h"
#include "dr3/advanced/reverse_aad.h"

#include "Vectorisation/VecX/forward_ad_black_scholes.h"
#include "Vectorisation/VecX/target_name_space.h"
#include "lattice/european_pde.h"
#include "lattice/tree_pricers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace dr3::advanced
{
namespace
{

std::string escapeJson(const std::string& value)
{
    std::string result;
    result.reserve(value.size());
    for (char character : value)
    {
        if (character == '\\' || character == '"')
        {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    return result;
}

std::size_t keyPosition(const std::string& json, const std::string& key)
{
    const std::string token = "\"" + key + "\":";
    const std::size_t position = json.find(token);
    if (position == std::string::npos)
    {
        throw std::invalid_argument("benchmark JSON is missing field: " + key);
    }
    return position + token.size();
}

std::string stringField(const std::string& json, const std::string& key)
{
    std::size_t position = keyPosition(json, key);
    if (position >= json.size() || json[position] != '"')
    {
        throw std::invalid_argument("benchmark JSON string field is malformed: " + key);
    }
    ++position;
    std::string result;
    bool escaped = false;
    for (; position < json.size(); ++position)
    {
        const char character = json[position];
        if (escaped)
        {
            result.push_back(character);
            escaped = false;
        }
        else if (character == '\\')
        {
            escaped = true;
        }
        else if (character == '"')
        {
            return result;
        }
        else
        {
            result.push_back(character);
        }
    }
    throw std::invalid_argument("benchmark JSON string is unterminated: " + key);
}

double numberField(const std::string& json, const std::string& key)
{
    const std::size_t position = keyPosition(json, key);
    std::size_t consumed = 0;
    const double value = std::stod(json.substr(position), &consumed);
    if (consumed == 0 || !std::isfinite(value))
    {
        throw std::invalid_argument("benchmark JSON numeric field is malformed: " + key);
    }
    return value;
}

std::uint64_t unsignedField(const std::string& json, const std::string& key)
{
    const std::size_t position = keyPosition(json, key);
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(json.substr(position), &consumed);
    if (consumed == 0)
    {
        throw std::invalid_argument("benchmark JSON unsigned field is malformed: " + key);
    }
    return static_cast<std::uint64_t>(value);
}

std::map<std::string, std::size_t> dimensionsField(const std::string& json)
{
    std::size_t position = keyPosition(json, "dimensions");
    if (position >= json.size() || json[position] != '{')
    {
        throw std::invalid_argument("benchmark JSON dimensions object is malformed");
    }
    ++position;
    std::map<std::string, std::size_t> result;
    while (position < json.size())
    {
        while (position < json.size() && (json[position] == ' ' || json[position] == ','))
            ++position;
        if (position < json.size() && json[position] == '}')
            break;
        if (position >= json.size() || json[position] != '"')
            throw std::invalid_argument("benchmark JSON dimension key is malformed");
        ++position;
        std::string key;
        bool escaped = false;
        for (; position < json.size(); ++position)
        {
            const char character = json[position];
            if (escaped)
            {
                key.push_back(character);
                escaped = false;
            }
            else if (character == '\\')
                escaped = true;
            else if (character == '"')
                break;
            else
                key.push_back(character);
        }
        if (++position >= json.size() || json[position] != ':')
            throw std::invalid_argument("benchmark JSON dimension separator is malformed");
        ++position;
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(json.substr(position), &consumed);
        if (consumed == 0 || value > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("benchmark JSON dimension value is malformed");
        result[key] = static_cast<std::size_t>(value);
        position += consumed;
    }
    if (result.empty())
        throw std::invalid_argument("benchmark JSON dimensions object is empty");
    return result;
}

std::vector<double> solveThomas(const PcrSystem& system, const std::vector<double>& right)
{
    const std::size_t count = system.size();
    std::vector<double> modifiedDiagonal = system.diagonal;
    std::vector<double> modifiedRight = right;
    std::vector<double> output(count);
    for (std::size_t row = 1; row < count; ++row)
    {
        const double multiplier = system.lower[row - 1] / modifiedDiagonal[row - 1];
        modifiedDiagonal[row] -= multiplier * system.upper[row - 1];
        modifiedRight[row] -= multiplier * modifiedRight[row - 1];
    }
    output.back() = modifiedRight.back() / modifiedDiagonal.back();
    for (std::size_t row = count - 1; row-- > 0;)
    {
        output[row] = (modifiedRight[row] - system.upper[row] * output[row + 1])
            / modifiedDiagonal[row];
    }
    return output;
}

dr3::lattice::VanillaOption benchmarkOption(double spot = 100.0)
{
    return {spot, 100.0, 0.2, 0.03, 0.0, 1.0, dr3::lattice::OptionType::Call};
}

double forwardAdWork()
{
    using DRC::VecD4D::VecxD;
    const VecxD spot(100.0, 1.0);
    const VecxD strike(100.0, 0.0);
    const VecxD volatility(0.2, 0.0);
    const VecxD rate(0.03, 0.0);
    const VecxD dividend(0.0, 0.0);
    const VecxD maturity(1.0, 0.0);
    return dr3::numerics::blackScholesCallForwardAd(
        spot, strike, volatility, rate, dividend, maturity).getScalarDeriv();
}

double crankNicolsonWork()
{
    const dr3::lattice::PdeConfig config{
        dr3::lattice::PdeScheme::CrankNicolsonRannacher, 101, 100, 0.0, 400.0};
    return dr3::lattice::europeanPdePrice(benchmarkOption(), config).price;
}

double scalarTreeWork(double spot = 100.0)
{
    return dr3::lattice::europeanTrinomial(benchmarkOption(spot), {64});
}

double simdTreeWork()
{
    constexpr int steps = 64;
    constexpr int nodeCount = 2 * steps + 1;
    const Vec4d spot(90.0, 100.0, 110.0, 120.0);
    const Vec4d strike(100.0);
    const Vec4d volatility(0.2);
    const Vec4d rate(0.03);
    const Vec4d dividend(0.0);
    const Vec4d maturity(1.0);
    const Vec4d dt = maturity / static_cast<double>(steps);
    const Vec4d dx = volatility * sqrt(3.0 * dt);
    const Vec4d drift = rate - dividend - 0.5 * volatility * volatility;
    const Vec4d varianceTerm = (dt * volatility * volatility
        + drift * drift * dt * dt) / (dx * dx);
    const Vec4d driftTerm = drift * dt / dx;
    const Vec4d probabilityUp = 0.5 * (varianceTerm + driftTerm);
    const Vec4d probabilityDown = 0.5 * (varianceTerm - driftTerm);
    const Vec4d probabilityMiddle = 1.0 - varianceTerm;
    const Vec4d discount = exp(-rate * dt);
    std::vector<Vec4d> prices(static_cast<std::size_t>(nodeCount));
    std::vector<Vec4d> scratch(static_cast<std::size_t>(nodeCount));
    for (int node = 0; node < nodeCount; ++node)
    {
        const Vec4d stock = spot * exp(static_cast<double>(node - steps) * dx);
        prices[static_cast<std::size_t>(node)] = select(
            stock > strike, stock - strike, Vec4d(0.0));
    }
    for (int level = 0; level < steps; ++level)
    {
        for (int node = level + 1; node < nodeCount - level - 1; ++node)
        {
            scratch[static_cast<std::size_t>(node)] = discount
                * (prices[static_cast<std::size_t>(node + 1)] * probabilityUp
                   + prices[static_cast<std::size_t>(node)] * probabilityMiddle
                   + prices[static_cast<std::size_t>(node - 1)] * probabilityDown);
        }
        prices.swap(scratch);
    }
    alignas(32) double result[4];
    prices[steps].store(result);
    return result[1];
}

double kernelWork(const std::string& name)
{
    if (name == "forward_ad")
    {
        return forwardAdWork();
    }
    if (name == "scalar_reverse_aad")
    {
        aad::ScalarTape tape;
        tape.reserve(4);
        const auto x = tape.variable(1.25);
        const auto y = exp(x) * x;
        tape.reverse(y);
        return tape.adjoint(x);
    }
    if (name == "simd_reverse_aad")
    {
        aad::SimdTape tape;
        tape.reserve(4);
        const auto x = tape.variable(Vec4d(1.0, 1.25, 1.5, 1.75));
        const auto y = exp(x) * x;
        tape.reverse(y);
        return aad::ReverseTraits<Vec4d>::lane(tape.adjoint(x), 1);
    }
    if (name == "direct_curve" || name == "planned_curve")
    {
        const std::vector<double> pillars{0.0, 1.0, 2.0, 3.0};
        const std::vector<double> values{1.0, 0.98, 0.94, 0.89};
        if (name == "direct_curve")
        {
            return directCurveValue(pillars, values, 1.25);
        }
        const CurveEvaluationPlan plan(pillars, {1.25});
        return plan.evaluateOne(pillars, values, 0);
    }
    if (name == "thomas")
    {
        PcrSystem system{{-1.0, -1.0}, {4.0, 4.0, 4.0}, {-1.0, -1.0}};
        return solveThomas(system, {2.0, 4.0, 10.0})[1];
    }
    if (name == "crank_nicolson_1d")
    {
        return crankNicolsonWork();
    }
    if (name == "pcr")
    {
        PcrSystem system{{-1.0, -1.0}, {4.0, 4.0, 4.0}, {-1.0, -1.0}};
        const std::vector<double> right{2.0, 4.0, 10.0};
        std::vector<double> output(3);
        PcrWorkspace workspace(3);
        PcrSolver::solve(system, right, output, workspace);
        return output[1];
    }
    if (name == "heston_mcs")
    {
        HestonParameters parameters{100.0, 100.0, 1.0, 0.03, 0.01,
                                    0.04, 0.04, 1.5, 0.3, -0.5,
                                    EuropeanOption::Call};
        const HestonPdeResult result = hestonPdePrice(parameters);
        if (!result.success)
        {
            throw std::runtime_error(result.error);
        }
        return result.price;
    }
    if (name == "scalar_tree")
    {
        return scalarTreeWork();
    }
    if (name == "simd_tree")
    {
        return simdTreeWork();
    }
    throw std::invalid_argument("unknown numerical benchmark kernel");
}

std::vector<std::string> kernelNames()
{
    return {"forward_ad", "scalar_reverse_aad", "simd_reverse_aad",
            "direct_curve", "planned_curve", "thomas", "pcr",
            "crank_nicolson_1d", "heston_mcs", "scalar_tree", "simd_tree"};
}

double kernelReference(const std::string& name)
{
    if (name == "forward_ad")
    {
        const double d1 = (0.03 + 0.5 * 0.2 * 0.2) / 0.2;
        return aad::ReverseTraits<double>::normalCdf(d1);
    }
    if (name == "scalar_reverse_aad" || name == "simd_reverse_aad")
    {
        return std::exp(1.25) * 2.25;
    }
    if (name == "direct_curve" || name == "planned_curve")
    {
        return 0.97;
    }
    if (name == "thomas" || name == "pcr")
    {
        return 2.0;
    }
    if (name == "crank_nicolson_1d" || name == "scalar_tree")
    {
        return blackScholesPrice(EuropeanOption::Call, 100.0, 100.0,
                                 1.0, 0.03, 0.0, 0.2);
    }
    if (name == "simd_tree")
    {
        return scalarTreeWork(100.0);
    }
    if (name == "heston_mcs")
    {
        const HestonParameters parameters{100.0, 100.0, 1.0, 0.03, 0.01,
                                          0.04, 0.04, 1.5, 0.3, -0.5,
                                          EuropeanOption::Call};
        return hestonReferencePrice(parameters);
    }
    throw std::invalid_argument("unknown benchmark reference kernel");
}

double kernelTolerance(const std::string& name)
{
    if (name == "heston_mcs") return 0.10;
    if (name == "crank_nicolson_1d" || name == "scalar_tree") return 0.10;
    if (name == "simd_tree") return 1.0e-10;
    return 1.0e-12;
}

BenchmarkDistribution distribution(std::vector<double> values)
{
    if (values.empty())
    {
        throw std::invalid_argument("benchmark duration sample is empty");
    }
    std::sort(values.begin(), values.end());
    BenchmarkDistribution result;
    result.minimum = values.front();
    result.maximum = values.back();
    result.median = values[values.size() / 2];
    const std::size_t p95Index = std::min(values.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(values.size()))) - 1);
    result.p95 = values[p95Index];
    result.mean = std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
    return result;
}

std::string compilerName()
{
#if defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GCC";
#elif defined(_MSC_VER)
    return "MSVC";
#else
    return "unknown";
#endif
}

std::string compilerVersion()
{
#if defined(__clang_version__)
    return __clang_version__;
#elif defined(__VERSION__)
    return __VERSION__;
#elif defined(_MSC_FULL_VER)
    return std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

std::string operatingSystem()
{
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "unknown";
#endif
}

} // namespace

void BenchmarkOptions::validate() const
{
    if (warmups == 0 || samples == 0)
    {
        throw std::invalid_argument("benchmark warmup and sample counts must be positive");
    }
}

void validateBenchmarkRecord(const BenchmarkRecord& record)
{
    if (record.schemaVersion != 1)
    {
        throw std::invalid_argument("unsupported numerical benchmark schema version");
    }
    if (record.benchmark.empty() || record.compiler.empty() || record.compilerVersion.empty()
        || record.build.empty() || record.isa.empty() || record.operatingSystem.empty()
        || record.hardwareThreads == 0 || record.dimensions.empty()
        || record.warmups == 0 || record.samples == 0 || record.durationUnit.empty()
        || record.gitCommit.empty())
    {
        throw std::invalid_argument("numerical benchmark record omits required metadata");
    }
    const double durations[] = {record.setupDuration, record.solveDuration.minimum,
                                record.solveDuration.median, record.solveDuration.p95,
                                record.solveDuration.maximum, record.solveDuration.mean};
    for (double duration : durations)
    {
        if (!std::isfinite(duration) || duration < 0.0)
        {
            throw std::invalid_argument("benchmark durations must be finite and nonnegative");
        }
    }
    if (!std::isfinite(record.referenceError) || record.referenceError < 0.0)
    {
        throw std::invalid_argument("benchmark reference error must be finite and nonnegative");
    }
}

std::string benchmarkRecordToJson(const BenchmarkRecord& record)
{
    validateBenchmarkRecord(record);
    std::ostringstream stream;
    stream << std::setprecision(17)
           << "{\"schema_version\":" << record.schemaVersion
           << ",\"benchmark\":\"" << escapeJson(record.benchmark)
           << "\",\"compiler\":\"" << escapeJson(record.compiler)
           << "\",\"compiler_version\":\"" << escapeJson(record.compilerVersion)
           << "\",\"build\":\"" << escapeJson(record.build)
           << "\",\"isa\":\"" << escapeJson(record.isa)
           << "\",\"os\":\"" << escapeJson(record.operatingSystem)
           << "\",\"hardware_threads\":" << record.hardwareThreads
           << ",\"dimensions\":{";
    bool first = true;
    for (const auto& dimension : record.dimensions)
    {
        if (!first)
        {
            stream << ',';
        }
        first = false;
        stream << '\"' << escapeJson(dimension.first) << "\":" << dimension.second;
    }
    stream << "},\"warmups\":" << record.warmups
           << ",\"samples\":" << record.samples
           << ",\"duration_unit\":\"" << escapeJson(record.durationUnit)
           << "\",\"setup_duration\":" << record.setupDuration
           << ",\"solve_min\":" << record.solveDuration.minimum
           << ",\"solve_median\":" << record.solveDuration.median
           << ",\"solve_p95\":" << record.solveDuration.p95
           << ",\"solve_max\":" << record.solveDuration.maximum
           << ",\"solve_mean\":" << record.solveDuration.mean
           << ",\"checksum\":" << record.checksum
           << ",\"reference_error\":" << record.referenceError
           << ",\"git_commit\":\"" << escapeJson(record.gitCommit) << "\"}";
    return stream.str();
}

BenchmarkRecord benchmarkRecordFromJson(const std::string& json)
{
    BenchmarkRecord result;
    result.schemaVersion = static_cast<std::size_t>(numberField(json, "schema_version"));
    if (result.schemaVersion != 1)
    {
        throw std::invalid_argument("unsupported numerical benchmark schema version");
    }
    result.benchmark = stringField(json, "benchmark");
    result.compiler = stringField(json, "compiler");
    result.compilerVersion = stringField(json, "compiler_version");
    result.build = stringField(json, "build");
    result.isa = stringField(json, "isa");
    result.operatingSystem = stringField(json, "os");
    result.hardwareThreads = static_cast<std::size_t>(numberField(json, "hardware_threads"));
    result.dimensions = dimensionsField(json);
    result.warmups = static_cast<std::size_t>(numberField(json, "warmups"));
    result.samples = static_cast<std::size_t>(numberField(json, "samples"));
    result.durationUnit = stringField(json, "duration_unit");
    result.setupDuration = numberField(json, "setup_duration");
    result.solveDuration.minimum = numberField(json, "solve_min");
    result.solveDuration.median = numberField(json, "solve_median");
    result.solveDuration.p95 = numberField(json, "solve_p95");
    result.solveDuration.maximum = numberField(json, "solve_max");
    result.solveDuration.mean = numberField(json, "solve_mean");
    result.checksum = unsignedField(json, "checksum");
    result.referenceError = numberField(json, "reference_error");
    result.gitCommit = stringField(json, "git_commit");
    validateBenchmarkRecord(result);
    return result;
}

std::uint64_t deterministicBenchmarkChecksum(const std::vector<double>& values)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (double value : values)
    {
        std::uint64_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

BenchmarkSelfTestResult runBenchmarkSelfTest(bool injectIncorrectResult)
{
    BenchmarkSelfTestResult result;
    std::vector<double> outputs;
    for (const std::string& name : kernelNames())
    {
        const double output = kernelWork(name);
        const double reference = kernelReference(name);
        bool correct = std::isfinite(output) && std::isfinite(reference)
            && std::abs(output - reference) <= kernelTolerance(name);
        if (injectIncorrectResult && name == "heston_mcs")
        {
            correct = false;
        }
        result.kernels[name] = correct;
        outputs.push_back(output);
    }
    result.success = std::all_of(result.kernels.begin(), result.kernels.end(),
                                 [](const auto& item) { return item.second; });
    result.checksum = deterministicBenchmarkChecksum(outputs);
    return result;
}

std::vector<BenchmarkRecord> runNumericalBenchmarks(const BenchmarkOptions& options)
{
    options.validate();
    const BenchmarkSelfTestResult selfTest = runBenchmarkSelfTest();
    if (!selfTest.success)
    {
        throw std::runtime_error("numerical benchmark correctness self-test failed before timing");
    }
    std::vector<BenchmarkRecord> records;
    records.reserve(kernelNames().size());
    for (const std::string& name : kernelNames())
    {
        const auto setupStart = std::chrono::steady_clock::now();
        const double reference = kernelReference(name);
        const auto setupEnd = std::chrono::steady_clock::now();
        const double correctnessOutput = kernelWork(name);
        if (!std::isfinite(correctnessOutput)
            || std::abs(correctnessOutput - reference) > kernelTolerance(name))
        {
            throw std::runtime_error("benchmark kernel failed its reference before timing: " + name);
        }
        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
        {
            (void)kernelWork(name);
        }
        std::vector<double> durations;
        std::vector<double> outputs;
        durations.reserve(options.samples);
        outputs.reserve(options.samples + 1);
        outputs.push_back(correctnessOutput);
        for (std::size_t sample = 0; sample < options.samples; ++sample)
        {
            const auto start = std::chrono::steady_clock::now();
            outputs.push_back(kernelWork(name));
            const auto end = std::chrono::steady_clock::now();
            durations.push_back(std::chrono::duration<double, std::nano>(end - start).count());
        }

        BenchmarkRecord record;
        record.benchmark = name;
        record.compiler = compilerName();
        record.compilerVersion = compilerVersion();
#ifdef NDEBUG
        record.build = "Release";
#else
        record.build = "Debug";
#endif
#ifdef DR3_ADVANCED_ISA
        record.isa = DR3_ADVANCED_ISA;
#else
        record.isa = "unspecified";
#endif
        record.operatingSystem = operatingSystem();
        record.hardwareThreads = std::max(1U, std::thread::hardware_concurrency());
        record.dimensions["logical_size"] = name == "heston_mcs" ? 3321 : 64;
        record.warmups = options.warmups;
        record.samples = options.samples;
        record.setupDuration = std::chrono::duration<double, std::nano>(
            setupEnd - setupStart).count();
        record.solveDuration = distribution(std::move(durations));
        record.checksum = deterministicBenchmarkChecksum(outputs);
        record.referenceError = std::abs(outputs.back() - reference);
#ifdef DR3_GIT_COMMIT
        record.gitCommit = DR3_GIT_COMMIT;
#else
        record.gitCommit = "unavailable";
#endif
        validateBenchmarkRecord(record);
        records.push_back(std::move(record));
    }
    return records;
}

} // namespace dr3::advanced
