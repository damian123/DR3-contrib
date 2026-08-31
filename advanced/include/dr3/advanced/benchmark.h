#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dr3::advanced
{

struct BenchmarkOptions
{
    std::size_t warmups{2};
    std::size_t samples{7};

    void validate() const;
};

struct BenchmarkDistribution
{
    double minimum{};
    double median{};
    double p95{};
    double maximum{};
    double mean{};
};

struct BenchmarkRecord
{
    std::size_t schemaVersion{1};
    std::string benchmark;
    std::string compiler;
    std::string compilerVersion;
    std::string build;
    std::string isa;
    std::string operatingSystem;
    std::size_t hardwareThreads{};
    std::map<std::string, std::size_t> dimensions;
    std::size_t warmups{};
    std::size_t samples{};
    std::string durationUnit{"nanoseconds"};
    double setupDuration{};
    BenchmarkDistribution solveDuration;
    std::uint64_t checksum{};
    double referenceError{};
    std::string gitCommit;
};

std::string benchmarkRecordToJson(const BenchmarkRecord& record);
BenchmarkRecord benchmarkRecordFromJson(const std::string& json);
void validateBenchmarkRecord(const BenchmarkRecord& record);

std::uint64_t deterministicBenchmarkChecksum(const std::vector<double>& values);

struct BenchmarkSelfTestResult
{
    bool success{};
    std::map<std::string, bool> kernels;
    std::uint64_t checksum{};
};

BenchmarkSelfTestResult runBenchmarkSelfTest(bool injectIncorrectResult = false);
std::vector<BenchmarkRecord> runNumericalBenchmarks(const BenchmarkOptions& options);

} // namespace dr3::advanced
