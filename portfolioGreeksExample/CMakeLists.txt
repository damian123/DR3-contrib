add_executable(portfolioGreeksExample portfolioGreeksExample.cpp)
target_link_libraries(portfolioGreeksExample PRIVATE DR3::Vectorisation)
dr3_apply_isa(portfolioGreeksExample)
target_compile_definitions(portfolioGreeksExample PRIVATE
    DR3_BENCHMARKS=$<BOOL:${DR3_BENCHMARKS}>)

if(MSVC)
    target_compile_options(portfolioGreeksExample PRIVATE /W4 /EHsc)
else()
    target_compile_options(portfolioGreeksExample PRIVATE -Wall -Wextra)
endif()

if(DR3_BUILD_TESTS)
    add_test(NAME portfolioGreeksSelfTest
        COMMAND $<TARGET_FILE:portfolioGreeksExample> --self-test)
endif()
