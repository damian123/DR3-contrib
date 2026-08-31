cmake_minimum_required(VERSION 3.17)
project(DR3CMakeConsumer LANGUAGES CXX)

if(NOT DEFINED DR3_ROOT)
    message(FATAL_ERROR "Configure with -DDR3_ROOT=/path/to/DR3")
endif()

set(DR3_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DR3_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${DR3_ROOT}" dr3 EXCLUDE_FROM_ALL)

add_executable(dr3_cmake_consumer main.cpp)
target_link_libraries(dr3_cmake_consumer PRIVATE DR3::Vectorisation dr3_advanced)
