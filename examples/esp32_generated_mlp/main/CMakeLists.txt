idf_component_register(
    SRCS "app_main.c"
    INCLUDE_DIRS "."
    REQUIRES
        mini_ort_generated_model
        mini_ort_platform
)