add_executable(KMeansExample KMeansExample.cpp)
target_link_libraries(KMeansExample PRIVATE DR3::Vectorisation)
dr3_apply_isa(KMeansExample)

if(MSVC)
    target_compile_options(KMeansExample PRIVATE /W4 /EHsc)
else()
    target_compile_options(KMeansExample PRIVATE -Wall -Wextra)
endif()

if(DR3_BUILD_TESTS)
    add_test(NAME KMeans
        COMMAND $<TARGET_FILE:KMeansExample> --self-test)
endif()
