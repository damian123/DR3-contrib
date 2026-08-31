#include "dr3/advanced/benchmark.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    try
    {
        dr3::advanced::BenchmarkOptions options;
        bool selfTestOnly = false;
        std::string outputPath = "dr3-numerical-benchmarks.json";
        for (int argument = 1; argument < argc; ++argument)
        {
            const std::string value = argv[argument];
            if (value == "--self-test")
            {
                selfTestOnly = true;
            }
            else if ((value == "--samples" || value == "--warmups" || value == "--output")
                     && argument + 1 < argc)
            {
                const std::string following = argv[++argument];
                if (value == "--samples")
                {
                    options.samples = static_cast<std::size_t>(std::stoull(following));
                }
                else if (value == "--warmups")
                {
                    options.warmups = static_cast<std::size_t>(std::stoull(following));
                }
                else
                {
                    outputPath = following;
                }
            }
            else
            {
                throw std::invalid_argument("unknown or incomplete benchmark option: " + value);
            }
        }

        const auto selfTest = dr3::advanced::runBenchmarkSelfTest();
        if (!selfTest.success)
        {
            std::cerr << "numerical benchmark self-test failed\n";
            return 2;
        }
        if (selfTestOnly)
        {
            std::cout << "validated " << selfTest.kernels.size()
                      << " numerical kernels; checksum=" << selfTest.checksum << '\n';
            return 0;
        }

        const auto records = dr3::advanced::runNumericalBenchmarks(options);
        std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("unable to open benchmark JSON output");
        }
        output << "{\"schema_version\":1,\"records\":[";
        for (std::size_t index = 0; index < records.size(); ++index)
        {
            if (index != 0)
            {
                output << ',';
            }
            output << dr3::advanced::benchmarkRecordToJson(records[index]);
        }
        output << "]}\n";
        std::cout << "wrote " << records.size() << " records to " << outputPath << '\n';
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "numerical benchmark error: " << exception.what() << '\n';
        return 1;
    }
}
