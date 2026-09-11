cmake_minimum_required(VERSION 3.20)
project(mini_ort_micro LANGUAGES C)

option(MER_MICRO_BUILD_EXAMPLES "Build Mini Ort Micro examples" ON)

add_library(mini_ort_micro STATIC
    src/runtime.c
    src/tensor_view.c
    kernels/reference/fully_connected_f32.c
    kernels/reference/relu_f32.c
)
target_include_directories(mini_ort_micro PUBLIC include)
target_compile_features(mini_ort_micro PUBLIC c_std_11)
set_target_properties(mini_ort_micro PROPERTIES C_EXTENSIONS OFF)

if(MER_MICRO_BUILD_EXAMPLES)
  add_executable(mini_ort_micro_mlp examples/mlp_f32.c)
  target_link_libraries(mini_ort_micro_mlp PRIVATE mini_ort_micro)
endif()