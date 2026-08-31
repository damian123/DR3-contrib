add_library(dr3_advanced STATIC
    src/adi.cpp
    src/benchmark.cpp
    src/heston.cpp
    src/pcr.cpp
)

target_compile_features(dr3_advanced PUBLIC cxx_std_17)
target_include_directories(dr3_advanced PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}"
)
target_link_libraries(dr3_advanced PUBLIC Vectorisation)
if(TARGET dr3_lattice)
    target_link_libraries(dr3_advanced PUBLIC dr3_lattice)
endif()
target_compile_definitions(dr3_advanced PRIVATE DR3_ADVANCED_ISA="${DR3_ISA}")
dr3_apply_isa(dr3_advanced)

if(MSVC)
    target_compile_options(dr3_advanced PRIVATE /W4 /EHsc)
else()
    target_compile_options(dr3_advanced PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(DR3_BUILD_TESTS)
    add_executable(AdvancedNumericsTest
        tests/TestAdi.cpp
        tests/TestBenchmark.cpp
        tests/TestCurvePlan.cpp
        tests/TestGridStencil.cpp
        tests/TestHeston.cpp
        tests/TestPcr.cpp
        tests/TestReverseAad.cpp
    )
    target_link_libraries(AdvancedNumericsTest PRIVATE
        dr3_advanced GTest::gtest GTest::gtest_main)
    target_compile_definitions(AdvancedNumericsTest PRIVATE
        DR3_ADVANCED_CTEST_FILE="${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake")
    dr3_apply_isa(AdvancedNumericsTest)
    if(DR3_ENABLE_SLOW_NUMERICAL_TESTS)
        target_compile_definitions(AdvancedNumericsTest PRIVATE
            DR3_ENABLE_SLOW_NUMERICAL_TESTS=1)
    endif()
    if(MSVC)
        target_compile_options(AdvancedNumericsTest PRIVATE /EHsc)
    endif()
    add_test(NAME AdvancedNumerics COMMAND $<TARGET_FILE:AdvancedNumericsTest>)
    add_test(NAME ReverseAAD
        COMMAND $<TARGET_FILE:AdvancedNumericsTest> --gtest_filter=ReverseAAD.*:SimdReverseAAD.*)
    add_test(NAME CurvePlan
        COMMAND $<TARGET_FILE:AdvancedNumericsTest> --gtest_filter=CurvePlan.*)
    add_test(NAME GridStencil2D
        COMMAND $<TARGET_FILE:AdvancedNumericsTest> --gtest_filter=Grid2D.*:Stencil2D.*)
    add_test(NAME Adi
        COMMAND $<TARGET_FILE:AdvancedNumericsTest> --gtest_filter=Adi.*)
    add_test(NAME Heston
        COMMAND $<TARGET_FILE:AdvancedNumericsTest> --gtest_filter=HestonReference.*:HestonPde.*)
    add_test(NAME Pcr
        COMMAND $<TARGET_FILE:AdvancedNumericsTest> --gtest_filter=Pcr.*)
    add_test(NAME BenchmarkTelemetry
        COMMAND $<TARGET_FILE:AdvancedNumericsTest> --gtest_filter=BenchmarkOptions.*:BenchmarkJson.*:BenchmarkChecksum.*:BenchmarkSelfTest.*:BenchmarkTimingMode.*)
endif()

if(DR3_BUILD_BENCHMARKS)
    add_executable(dr3_numerical_benchmarks src/benchmark_main.cpp)
    target_link_libraries(dr3_numerical_benchmarks PRIVATE dr3_advanced)
    dr3_apply_isa(dr3_numerical_benchmarks)
    execute_process(
        COMMAND git rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE DR3_ADVANCED_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT DR3_ADVANCED_GIT_COMMIT)
        set(DR3_ADVANCED_GIT_COMMIT "unavailable")
    endif()
    target_compile_definitions(dr3_numerical_benchmarks PRIVATE
        DR3_GIT_COMMIT="${DR3_ADVANCED_GIT_COMMIT}")
    # Full timing is intentionally not registered with CTest.
endif()
