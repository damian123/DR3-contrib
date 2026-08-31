cmake_minimum_required(VERSION 3.17)

project(DR3PackageConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(DR3 0.1 CONFIG REQUIRED)

if(NOT TARGET DR3::Vectorisation)
    message(FATAL_ERROR "DR3 package did not export DR3::Vectorisation")
endif()

if(DEFINED DR3_EXPECTED_ISA AND
   NOT "${DR3_BUILT_ISA}" STREQUAL "${DR3_EXPECTED_ISA}")
    message(FATAL_ERROR
        "Expected a ${DR3_EXPECTED_ISA} package, found ${DR3_BUILT_ISA}")
endif()

get_target_property(_dr3_compile_options
    DR3::Vectorisation INTERFACE_COMPILE_OPTIONS)
get_target_property(_dr3_compile_features
    DR3::Vectorisation INTERFACE_COMPILE_FEATURES)

if(NOT "cxx_std_17" IN_LIST _dr3_compile_features)
    message(FATAL_ERROR "Installed DR3 target must require C++17")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if("-march=native" IN_LIST _dr3_compile_options)
        message(FATAL_ERROR "Installed DR3 target must not use -march=native")
    endif()

    if(DR3_BUILT_ISA STREQUAL "AVX2")
        if(NOT "-mavx2" IN_LIST _dr3_compile_options OR
           NOT "-mfma" IN_LIST _dr3_compile_options)
            message(FATAL_ERROR
                "Installed AVX2 target must export -mavx2 and -mfma")
        endif()
    endif()
elseif(MSVC AND DR3_BUILT_ISA STREQUAL "AVX2")
    if(NOT "/arch:AVX2" IN_LIST _dr3_compile_options)
        message(FATAL_ERROR
            "Installed AVX2 target must export /arch:AVX2")
    endif()
endif()

add_executable(dr3_package_consumer main.cpp)
target_link_libraries(dr3_package_consumer PRIVATE DR3::Vectorisation)

enable_testing()
add_test(NAME DR3PackageConsumer COMMAND $<TARGET_FILE:dr3_package_consumer>)
