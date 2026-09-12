cmake_minimum_required(VERSION 3.16)

set(
    EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../../micro/platform/esp_idf/components"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(mini_ort_generated_mlp)