cmake_minimum_required(VERSION 3.17)

project(DR3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(DR3_BUILD_TESTS "Build VectorTest and register it with CTest" OFF)
option(DR3_BUILD_EXAMPLES "Build example programs" ON)
option(DR3_BENCHMARKS "Enable benchmark-oriented example output" OFF)
option(DR3_ENABLE_SLOW_NUMERICAL_TESTS "Enable expensive advanced numerical convergence tests" OFF)
option(DR3_BUILD_BENCHMARKS "Build reproducible advanced numerical benchmark telemetry" OFF)
option(DR3_FASTFLOAT_FIXTURES "Build the optional pinned fast_float CSV fixture test" OFF)
option(DR3_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer (GCC/Clang)" OFF)
option(DR3_ENABLE_TSAN "Enable ThreadSanitizer (GCC/Clang)" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    if(DR3_ENABLE_SANITIZERS OR DR3_ENABLE_TSAN)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "CMAKE_BUILD_TYPE was unset; defaulting to ${CMAKE_BUILD_TYPE}")
endif()

if(DR3_ENABLE_SANITIZERS AND DR3_ENABLE_TSAN)
    message(FATAL_ERROR "DR3_ENABLE_SANITIZERS and DR3_ENABLE_TSAN cannot be combined")
endif()

# VectorTest uses the AVX2 DRC::VecD4D namespace. AVX512 may be compiled
# but is not the default runtime test ISA.
set(DR3_ISA "AVX2" CACHE STRING "Instruction set for CMake builds: SSE2, SSE4.2, AVX2, or AVX512")
set_property(CACHE DR3_ISA PROPERTY STRINGS SSE2 SSE4.2 AVX2 AVX512)

function(dr3_apply_sse42 target)
    # MSVC has no /arch:SSE4.2 switch. /arch:AVX is its lowest supported
    # setting that includes SSE4.2; the definition selects the matching VCL path.
    if(MSVC)
        target_compile_options(${target} PRIVATE /arch:AVX)
        target_compile_definitions(${target} PRIVATE INSTRSET=6)
    else()
        target_compile_options(${target} PRIVATE -msse4.2)
    endif()
endfunction()

function(dr3_apply_isa target)
    if(MSVC)
        if(DR3_ISA STREQUAL "AVX512")
            target_compile_options(${target} PUBLIC /arch:AVX512)
        elseif(DR3_ISA STREQUAL "SSE2")
            target_compile_options(${target} PUBLIC /arch:SSE2)
        elseif(DR3_ISA STREQUAL "SSE4.2" OR DR3_ISA STREQUAL "SSE42")
            target_compile_options(${target} PUBLIC /arch:AVX)
            target_compile_definitions(${target} PUBLIC INSTRSET=6)
        else()
            target_compile_options(${target} PUBLIC /arch:AVX2)
        endif()
    else()
        if(DR3_ISA STREQUAL "AVX512")
            target_compile_options(${target} PUBLIC -mavx512f -mfma)
        elseif(DR3_ISA STREQUAL "SSE2")
            target_compile_options(${target} PUBLIC -msse2)
        elseif(DR3_ISA STREQUAL "SSE4.2" OR DR3_ISA STREQUAL "SSE42")
            target_compile_options(${target} PUBLIC -msse4.2)
        else()
            target_compile_options(${target} PUBLIC -mavx2 -mfma)
        endif()
    endif()
endfunction()

if(DR3_ENABLE_SANITIZERS)
    if(MSVC)
        message(WARNING "DR3_ENABLE_SANITIZERS is supported on GCC/Clang only")
    else()
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

if(DR3_ENABLE_TSAN)
    if(MSVC)
        message(FATAL_ERROR "DR3_ENABLE_TSAN is supported on GCC/Clang only")
    else()
        add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
        add_link_options(-fsanitize=thread)
    endif()
endif()

add_subdirectory(Vectorisation)

if(DR3_BUILD_TESTS)
    include(CTest)
    enable_testing()
endif()

if(DR3_BUILD_EXAMPLES)
    add_subdirectory(accumulateExample)
    add_subdirectory(cumNormalExample)
    add_subdirectory(inverseCumNormalExample)
    add_subdirectory(dancingAVX512)
    add_subdirectory(GettingStarted)
    add_subdirectory(scratch)
    add_subdirectory(VariadicReduction)
    add_subdirectory(portfolioGreeksExample)
    add_subdirectory(portfolioStressExample)
    add_subdirectory(monteCarloExample)
    add_subdirectory(ModelSensitivity)
    add_subdirectory(KMeansExample)
    add_subdirectory(EmbeddingSearch)
    add_subdirectory(SmallBatchInference)
endif()

if(DR3_BUILD_TESTS)
    add_subdirectory(VectorTest)
    add_subdirectory(AIKernels)
endif()

if(DR3_FASTFLOAT_FIXTURES)
    add_subdirectory(fastFloatFixture)
endif()

# The lattice core is reusable and is also a public dependency of the
# advanced library. Its example, test, and benchmark executables remain gated
# independently inside lattice/CMakeLists.txt.
add_subdirectory(lattice)

add_subdirectory(advanced)

# The fast-float compatibility command builds only VectorTest before running
# all registered tests. Keep the other required test executables reachable
# through that target as well.
if(DR3_BUILD_TESTS)
    add_dependencies(VectorTest VectorTestSSE42 DR3AIKernelTests LatticeTest AdvancedNumericsTest)
    if(DR3_FASTFLOAT_FIXTURES)
        add_dependencies(VectorTest FastFloatFixtureTest)
    endif()
endif()
