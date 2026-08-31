if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# These translation units compile with GCC, Clang, and MSVC using the
# AVX2 DRC::VecD4D test namespace. Other VectorTest files currently
# depend on MSVC language extensions or missing headers.
set(DR3_VECTOR_TEST_PORTABLE_SOURCES
    TestBasicPortable.cpp
    TestConcurrentAllocator.cpp
    TestCurve.cpp
    TestAccumulator.cpp
    TestAllocator.cpp
    TestFilterSelect.cpp
    TestForwardAD.cpp
    TestSimilarity.cpp
    TestNeuralKernels.cpp
    testNamespace.cpp
    test_precise_accumulation.cpp
)

add_executable(VectorTest ${DR3_VECTOR_TEST_PORTABLE_SOURCES})

target_link_libraries(VectorTest PRIVATE Vectorisation GTest::gtest GTest::gtest_main)

target_include_directories(VectorTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

dr3_apply_isa(VectorTest)

if(MSVC)
    target_compile_options(VectorTest PRIVATE /W4 /EHsc)
else()
    target_compile_options(VectorTest PRIVATE -Wall -Wextra)
endif()

# Use add_test rather than gtest_discover_tests: CMake 4.x POST_BUILD
# discovery fails on Windows when the binary is in a multi-config output
# directory. The complete portable suite is a required gate.
add_test(NAME VectorTest COMMAND $<TARGET_FILE:VectorTest>)
add_test(NAME ParallelAllocator
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=ParallelAllocator.*)
add_test(NAME ConcurrencySafetyVector
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=ConcurrencySafety.*)
add_test(NAME ForwardAD
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=ForwardAD.*)
add_test(NAME Curve
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=Curve.*)
add_test(NAME CurveSensitivity
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=CurveSensitivity.*)
add_test(NAME Similarity
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=TestSimilarity.*)
add_test(NAME NeuralSoftmax
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=Softmax.*)
add_test(NAME NeuralLayerNorm
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=LayerNorm.*)
add_test(NAME NeuralRMSNorm
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=RMSNorm.*)
add_test(NAME NeuralActivation
    COMMAND $<TARGET_FILE:VectorTest> --gtest_filter=Activation.*)

# Keep the 128-bit VecD2D SSE4.2 path in its own translation unit. The
# primary VectorTest target continues to exercise the AVX2 VecD4D API.
add_executable(VectorTestSSE42 TestSSE42.cpp)
target_link_libraries(VectorTestSSE42 PRIVATE DR3::Vectorisation GTest::gtest GTest::gtest_main)
target_include_directories(VectorTestSSE42 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
dr3_apply_sse42(VectorTestSSE42)

if(MSVC)
    target_compile_options(VectorTestSSE42 PRIVATE /W4 /EHsc)
else()
    target_compile_options(VectorTestSSE42 PRIVATE -Wall -Wextra)
endif()

add_test(NAME VectorTestSSE42 COMMAND $<TARGET_FILE:VectorTestSSE42>)
