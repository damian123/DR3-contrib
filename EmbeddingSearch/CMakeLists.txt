add_executable(EmbeddingSearch EmbeddingSearch.cpp)

target_link_libraries(EmbeddingSearch PRIVATE DR3::Vectorisation)

target_include_directories(EmbeddingSearch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")

dr3_apply_isa(EmbeddingSearch)

if(MSVC)
    target_compile_options(EmbeddingSearch PRIVATE /EHsc /W4)
else()
    target_compile_options(EmbeddingSearch PRIVATE -Wall -Wextra)
endif()

if(DR3_BUILD_TESTS)
    add_test(NAME EmbeddingSearchSelfTest
        COMMAND $<TARGET_FILE:EmbeddingSearch> --self-test)
    set_tests_properties(EmbeddingSearchSelfTest PROPERTIES TIMEOUT 60)
endif()
