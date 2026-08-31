add_library(dr3_lattice STATIC
    american_pde.cpp
    american_trinomial.cpp
    barrier_trinomial.cpp
    batched_pde.cpp
    european_binomial.cpp
    european_pde.cpp
    european_trinomial.cpp
    parallel_executor.cpp
    tree_pricer_detail.cpp
    tridiagonal.cpp
)

target_link_libraries(dr3_lattice PUBLIC Vectorisation)
target_include_directories(dr3_lattice PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
dr3_apply_isa(dr3_lattice)

if(DR3_BUILD_EXAMPLES)
    add_executable(latticeExample lattice_example.cpp)
    target_link_libraries(latticeExample PRIVATE dr3_lattice)
endif()

if(DR3_BENCHMARKS)
    add_executable(NumericsBenchmark numerics_benchmark.cpp)
    target_link_libraries(NumericsBenchmark PRIVATE dr3_lattice)
    target_compile_definitions(NumericsBenchmark PRIVATE
        DR3_BENCHMARK_ISA="${DR3_ISA}")
    dr3_apply_isa(NumericsBenchmark)
endif()

if(DR3_BUILD_TESTS)
    add_executable(LatticeTest
        tests/TestPdeGrid.cpp
        tests/TestAmericanPde.cpp
        tests/TestBatchedPde.cpp
        tests/TestEuropeanPde.cpp
        tests/TestConcurrency.cpp
        tests/TestTreePricers.cpp
        tests/TestTridiagonal.cpp
    )
    target_link_libraries(LatticeTest PRIVATE dr3_lattice GTest::gtest GTest::gtest_main)
    dr3_apply_isa(LatticeTest)
    if(MSVC)
        target_compile_options(LatticeTest PRIVATE /EHsc)
    endif()
    add_test(NAME LatticeTest COMMAND $<TARGET_FILE:LatticeTest>)
    add_test(NAME Grid1D COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=Grid1D.*)
    add_test(NAME EuropeanPde COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=EuropeanPde.*)
    add_test(NAME AmericanPde COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=AmericanPde.*)
    add_test(NAME BatchedPde COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=BatchedPde.*)
    add_test(NAME PdeSensitivity COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=PdeSensitivity.*)
    add_test(NAME ConcurrencySafety
        COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=ConcurrencySafety.*)
    add_test(NAME ParallelExecutor
        COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=ParallelExecutor.*)
    add_test(NAME Tridiagonal COMMAND $<TARGET_FILE:LatticeTest> --gtest_filter=Tridiagonal.*)
endif()
