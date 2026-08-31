add_library(Vectorisation STATIC
            VecX/alloc_policy.cpp
            VecX/vec.cpp)
add_library(DR3::Vectorisation ALIAS Vectorisation)

dr3_apply_isa(Vectorisation)

target_include_directories(Vectorisation PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:include>")
