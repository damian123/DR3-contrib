#include "dr3/advanced/benchmark.h"

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
dr3::advanced::BenchmarkRecord sampleRecord()
{
    dr3::advanced::BenchmarkRecord record;
    record.benchmark = "pcr";
    record.compiler = "GCC";
    record.compilerVersion = "test";
    record.build = "Release";
    record.isa = "AVX2";
    record.operatingSystem = "Linux";
    record.hardwareThreads = 8;
    record.dimensions["logical_size"] = 257;
    record.dimensions["stages"] = 9;
    record.warmups = 2;
    record.samples = 7;
    record.setupDuration = 10.0;
    record.solveDuration = {1.0, 2.0, 3.0, 4.0, 2.5};
    record.checksum = 0xfedcba9876543210ULL;
    record.referenceError = 1.0e-13;
    record.gitCommit = "0123456789ab";
    return record;
}

TEST(BenchmarkOptions, RejectsZeroSamples)
{
    dr3::advanced::BenchmarkOptions options;
    options.samples = 0;
    EXPECT_THROW(options.validate(), std::invalid_argument);
}

TEST(BenchmarkOptions, RejectsZeroWarmups)
{
    dr3::advanced::BenchmarkOptions options;
    options.warmups = 0;
    EXPECT_THROW(options.validate(), std::invalid_argument);
}

TEST(BenchmarkJson, ContainsRequiredFields)
{
    const std::string json = dr3::advanced::benchmarkRecordToJson(sampleRecord());
    for (const std::string& field : {"schema_version", "benchmark", "compiler",
            "compiler_version", "build", "isa", "os", "hardware_threads",
            "dimensions", "samples", "duration_unit", "checksum",
            "reference_error", "git_commit"})
        EXPECT_NE(json.find("\"" + field + "\""), std::string::npos) << field;
}

TEST(BenchmarkJson, RoundTrip)
{
    const auto original = sampleRecord();
    const auto parsed = dr3::advanced::benchmarkRecordFromJson(
        dr3::advanced::benchmarkRecordToJson(original));
    EXPECT_EQ(parsed.schemaVersion, original.schemaVersion);
    EXPECT_EQ(parsed.benchmark, original.benchmark);
    EXPECT_EQ(parsed.dimensions, original.dimensions);
    EXPECT_EQ(parsed.checksum, original.checksum);
    EXPECT_DOUBLE_EQ(parsed.solveDuration.p95, original.solveDuration.p95);
    EXPECT_EQ(parsed.gitCommit, original.gitCommit);
}

TEST(BenchmarkJson, RejectsUnsupportedSchema)
{
    std::string json = dr3::advanced::benchmarkRecordToJson(sampleRecord());
    const auto position = json.find("\"schema_version\":1");
    ASSERT_NE(position, std::string::npos);
    json.replace(position, std::string("\"schema_version\":1").size(),
                 "\"schema_version\":2");
    EXPECT_THROW(dr3::advanced::benchmarkRecordFromJson(json), std::invalid_argument);
}

TEST(BenchmarkChecksum, IsDeterministic)
{
    const std::vector<double> values{1.0, -0.0, 3.25, 9.0};
    EXPECT_EQ(dr3::advanced::deterministicBenchmarkChecksum(values),
              dr3::advanced::deterministicBenchmarkChecksum(values));
}

TEST(BenchmarkSelfTest, ValidatesEveryKernel)
{
    const auto result = dr3::advanced::runBenchmarkSelfTest();
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.kernels.size(), 11U);
    for (const auto& kernel : result.kernels) EXPECT_TRUE(kernel.second) << kernel.first;
}

TEST(BenchmarkSelfTest, FailsOnIncorrectResult)
{
    const auto result = dr3::advanced::runBenchmarkSelfTest(true);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.kernels.at("heston_mcs"));
}

TEST(BenchmarkTimingMode, IsNotRegisteredAsCTest)
{
#ifdef DR3_ADVANCED_CTEST_FILE
    std::ifstream input(DR3_ADVANCED_CTEST_FILE, std::ios::binary);
    ASSERT_TRUE(input) << DR3_ADVANCED_CTEST_FILE;
    const std::string registrations((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
    EXPECT_EQ(registrations.find("dr3_numerical_benchmarks"), std::string::npos);
#else
    FAIL() << "CMake must expose the generated CTest registry to this test";
#endif
}

} // namespace
