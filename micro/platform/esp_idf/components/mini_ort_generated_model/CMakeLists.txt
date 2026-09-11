set(MER_MICRO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(MER_GENERATED_MODEL_DIR "${MER_MICRO_ROOT}/generated/tiny_mlp_f32")

idf_component_register(
    SRCS "${MER_GENERATED_MODEL_DIR}/model.c"
    INCLUDE_DIRS "${MER_GENERATED_MODEL_DIR}/include"
    REQUIRES mini_ort_micro
)