cmake_minimum_required(VERSION 3.20)
project(
    mini_edge_machine_learning_runtime 
    LANGUAGES CXX
)

set(
    MINI_ORT_LIBRARY_TYPE
    SHARED
    CACHE STRING "mini_ort library type"
)

set_property(
    CACHE MINI_ORT_LIBRARY_TYPE
    PROPERTY STRINGS SHARED STATIC
)

option(
    MINI_ORT_BUILD_EXAMPLES
    "Build native examples"
    ON
)

option(
    MINI_ORT_BUILD_BENCHMARKS
    "Build native benchmarks"
    ON
)

set(MINI_ORT_SOURCES
    src/mini_ort/allocator.cc
    src/mini_ort/c_api.cc
    src/mini_ort/execution_plan.cc
    src/mini_ort/kernel.cc
    src/mini_ort/kernels/cpu/add.cc
    src/mini_ort/kernels/cpu/matmul.cc
    src/mini_ort/kernels/cpu/register.cc
    src/mini_ort/kernels/cpu/relu.cc
    src/mini_ort/model.cc
    src/mini_ort/model_format.cc
    src/mini_ort/memory_plan.cc
    src/mini_ort/run_workspace.cc
    src/mini_ort/session.cc
    src/mini_ort/tensor.cc
    src/mini_ort/tensor_arena.cc
    src/mini_ort/tensor_storage.cc
    src/mini_ort/tensor_view.cc
)

add_library(
    mini_ort
    ${MINI_ORT_LIBRARY_TYPE}
    ${MINI_ORT_SOURCES}
)

target_include_directories(
    mini_ort 
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(
    mini_ort 
    PUBLIC 
        cxx_std_20
)

if (MINI_ORT_BUILD_EXAMPLES)

    add_executable(mini_ort_mlp examples/mlp.cc)
    target_compile_features(mini_ort_mlp PRIVATE cxx_std_20)
    target_link_libraries(mini_ort_mlp PRIVATE mini_ort)

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/run_model.cc")
        add_executable(mini_ort_run_model examples/run_model.cc)
        target_compile_features(mini_ort_run_model PRIVATE cxx_std_20)
        target_link_libraries(mini_ort_run_model PRIVATE mini_ort)

    endif()
endif()

if (
    MINI_ORT_BUILD_BENCHMARKS
    AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/inference_benchmark.cc"
)
    add_executable(mini_ort_benchmark benchmarks/inference_benchmark.cc)
    target_compile_features(mini_ort_benchmark PRIVATE cxx_std_20)
    target_link_libraries(mini_ort_benchmark PRIVATE mini_ort)
endif()

install(
    TARGETS mini_ort
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    DIRECTORY include/
    DESTINATION include
)