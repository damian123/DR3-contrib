add_executable(ModelSensitivity ModelSensitivity.cpp)
target_link_libraries(ModelSensitivity PRIVATE DR3::Vectorisation)
dr3_apply_isa(ModelSensitivity)

if(MSVC)
    target_compile_options(ModelSensitivity PRIVATE /W4 /EHsc)
else()
    target_compile_options(ModelSensitivity PRIVATE -Wall -Wextra)
endif()

if(DR3_BUILD_TESTS)
    add_test(NAME ModelSensitivity
        COMMAND $<TARGET_FILE:ModelSensitivity> --self-test)
endif()
