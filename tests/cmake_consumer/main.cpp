#include "VecX/dr3.h"
#include "dr3/advanced/benchmark.h"

#include <vector>

int main()
{
    if (DRC::VecD4D::VecXX::INS::size() != 4) return 1;

    const std::vector<double> values{1.0, 2.0, 3.0};
    return dr3::advanced::deterministicBenchmarkChecksum(values) == 0U ? 1 : 0;
}
