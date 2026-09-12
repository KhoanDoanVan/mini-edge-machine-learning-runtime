idf_component_register(
    SRCS "src/esp_memory_probe.c"
    INCLUDE_DIRS "include"
    PRIV_REQUIRES esp_common esp_system freertos heap log spi_flash
)