add_executable(monteCarloExample monteCarloExample.cpp)
target_link_libraries(monteCarloExample PRIVATE DR3::Vectorisation)
dr3_apply_isa(monteCarloExample)
target_compile_definitions(monteCarloExample PRIVATE
    DR3_BENCHMARKS=$<BOOL:${DR3_BENCHMARKS}>)

if(MSVC)
    target_compile_options(monteCarloExample PRIVATE /W4 /EHsc)
else()
    target_compile_options(monteCarloExample PRIVATE -Wall -Wextra)
endif()

if(DR3_BUILD_TESTS)
    add_test(NAME monteCarloSelfTest
        COMMAND $<TARGET_FILE:monteCarloExample> --self-test)
endif()
