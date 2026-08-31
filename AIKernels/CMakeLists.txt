add_executable(DR3AIKernelTests AIKernelTests.cpp)
target_link_libraries(DR3AIKernelTests PRIVATE DR3::Vectorisation)
dr3_apply_isa(DR3AIKernelTests)

if(MSVC)
    target_compile_options(DR3AIKernelTests PRIVATE /W4 /EHsc)
else()
    target_compile_options(DR3AIKernelTests PRIVATE -Wall -Wextra)
endif()

add_test(NAME Quantization
    COMMAND $<TARGET_FILE:DR3AIKernelTests> --quantization)
add_test(NAME DotS8
    COMMAND $<TARGET_FILE:DR3AIKernelTests> --quantized-dot)
add_test(NAME Preprocess
    COMMAND $<TARGET_FILE:DR3AIKernelTests> --preprocess)
