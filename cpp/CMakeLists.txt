cmake_minimum_required(VERSION 3.20)
project(mini_edge_machine_learning_runtime LANGUAGES CXX)

add_library(mini_ort SHARED
    src/mini_ort/c_api.cc
    src/mini_ort/kernel.cc
    src/mini_ort/kernels/cpu/add.cc
    src/mini_ort/kernels/cpu/matmul.cc
    src/mini_ort/kernels/cpu/register.cc
    src/mini_ort/kernels/cpu/relu.cc
    src/mini_ort/model.cc
    src/mini_ort/session.cc
    src/mini_ort/tensor.cc
)
target_include_directories(mini_ort PUBLIC include)
target_compile_features(mini_ort PUBLIC cxx_std_20)

add_executable(mini_ort_mlp examples/mlp.cc)
target_compile_features(mini_ort_mlp PRIVATE cxx_std_20)
target_link_libraries(mini_ort_mlp PRIVATE mini_ort)