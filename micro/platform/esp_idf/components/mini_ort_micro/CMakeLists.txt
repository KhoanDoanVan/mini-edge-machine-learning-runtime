set(MER_MICRO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")

idf_component_register(
    SRCS
        "${MER_MICRO_ROOT}/src/runtime.c"
        "${MER_MICRO_ROOT}/src/tensor_view.c"
        "${MER_MICRO_ROOT}/kernels/reference/fully_connected_f32.c"
        "${MER_MICRO_ROOT}/kernels/reference/relu_f32.c"
    INCLUDE_DIRS "${MER_MICRO_ROOT}/include"
)