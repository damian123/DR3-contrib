add_executable(portfolioStressExample portfolioStressExample.cpp)
target_link_libraries(portfolioStressExample PRIVATE DR3::Vectorisation)
dr3_apply_isa(portfolioStressExample)
target_compile_definitions(portfolioStressExample PRIVATE
    DR3_BENCHMARKS=$<BOOL:${DR3_BENCHMARKS}>)

if(MSVC)
    target_compile_options(portfolioStressExample PRIVATE /W4 /EHsc)
else()
    target_compile_options(portfolioStressExample PRIVATE -Wall -Wextra)
endif()

if(DR3_BUILD_TESTS)
    add_test(NAME portfolioStressSelfTest
        COMMAND $<TARGET_FILE:portfolioStressExample> --self-test)
endif()
