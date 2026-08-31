include(FetchContent)

# Keep this dependency entirely opt-in and immutable. The archive hash was
# verified against the upstream v8.2.10 release tarball.
set(FASTFLOAT_TEST OFF CACHE BOOL "" FORCE)
set(FASTFLOAT_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(FASTFLOAT_INSTALL OFF CACHE BOOL "" FORCE)
set(FASTFLOAT_CXX_STANDARD 17 CACHE STRING "" FORCE)
FetchContent_Declare(
    fast_float
    URL https://github.com/fastfloat/fast_float/archive/refs/tags/v8.2.10.tar.gz
    URL_HASH SHA256=76f958dd97b1cf4d8862d1f0986a47d4bdfa8845252bae15ef0f40de3b95961f
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(fast_float)

add_executable(FastFloatFixtureTest FastFloatFixtureTest.cpp)
target_link_libraries(FastFloatFixtureTest PRIVATE
    DR3::Vectorisation FastFloat::fast_float)
target_compile_definitions(FastFloatFixtureTest PRIVATE
    DR3_FASTFLOAT_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/options.csv")
dr3_apply_isa(FastFloatFixtureTest)

if(MSVC)
    target_compile_options(FastFloatFixtureTest PRIVATE /W4 /EHsc)
else()
    target_compile_options(FastFloatFixtureTest PRIVATE -Wall -Wextra)
endif()

if(DR3_BUILD_TESTS)
    add_test(NAME FastFloatFixture
        COMMAND $<TARGET_FILE:FastFloatFixtureTest> --self-test)
endif()
