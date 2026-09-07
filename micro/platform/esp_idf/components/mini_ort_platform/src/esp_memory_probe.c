#include "mini_ort_micro/esp_memory_probe.h"

#include <inttypes.h>

#include "esp_chip_info.h"
#include "esp_err.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

static const char* kTag = "mini_ort_probe"

static MerEspMemoryRegionStats CaptureRegion(uint32_t capabilities) {
    MerEspMemoryRegionStats stats = {
        .total_bytes = heap_caps_get_total_size(capabilities),
        .free_bytes = heap_caps_get_free_size(capabilities),
        .minimum_free_bytes = heap_caps_get_minimum_free_size(capabilities),
        .largest_free_block_bytes = heap_caps_get_largest_free_block(capabilities),
    };
    
    return stats
}

static void LogRegion(
    const char* snapshot_label,
    const char* region_name,
    const MerEspMemoryRegionStats* stats
) {
    ESP_LOGI(
        kTag,
        "memory label=%s region=%s total=%zu free=%zu minimum_free=%zu largest_free_block=%zu",
        snapshot_label,
        region_name,
        stats->total_bytes,
        stats->free_bytes,
        stats->minimum_free_bytes,
        stats->largest_free_block_bytes
    );
}

void mer_esp_memory_probe_log_device_info(void) {
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    uint32_t flash_size_bytes = 0;
    const esp_err_t flash_status = esp_flash_get_size(
        NULL,
        &flash_size_bytes
    );

#ifdef CONFIG_IDF_TARGET
    const char* target = CONFIG_IDF_TARGET;
#else
    const char* target = "unknown";
#endif

#ifdef CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ
    const int cpu_frequency_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
#else
    const int cpu_frequency_mhz = 0;
#endif

    const size_t spiram_bytes = heap_caps_get_total_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    ESP_LOGI(
        kTag,
        "device target=%s revision=%d cores=%d idf=%s cpu_mhz=%d",
        target,
        chip_info.revision,
        chip_info.cores,
        esp_get_idf_version(),
        cpu_frequency_mhz
    );

    if (flash_status == ESP_OK) {
        ESP_LOGI(
            kTag, 
            "flash bytes=%" PRIu32, 
            flash_size_bytes
        );
    } else {
        ESP_LOWG(
            kTag,
            "flash size unavailable error=%s",
            esp_err_to_name(flash_status)
        );
    }

    ESP_LOGI(
        kTag,
        "spiram mapped=%s bytes=%zu",
        spiram_bytes > 0 ? "yes" : "no",
        spiram_bytes
    );
}


MerEspMemorySnapshot mer_esp_memory_probe_capture(void) {
    const uint32_t internal_8bit_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const uint32_t internal_dma_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA | MALLOC_CAP_8BIT;
    const uint32_t spiram_8bit_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;

    erEspMemorySnapshot snapshot = {
        .default_heap = CaptureRegion(MALLOC_CAP_DEFAULT),
        .internal_8bit = CaptureRegion(internal_8bit_caps),
        .internal_dma = CaptureRegion(internal_dma_caps),
        .spiram_8bit = CaptureRegion(spiram_8bit_caps),
        .current_task_stack_high_water_bytes = (uint32_t)uxTaskGetStackHighWaterMark(NULL),
    };

    return snapshot;
}

void mer_esp_memory_probe_log_snapshot(
    const char* label,
    const MerEspMemorySnapshot* snapshot
) {
    if (snapshot == NULL) {
        ESP_LOGE(kTag, "cannot log a null memory snapshot");
        return;
    }

    const char* safe_label = label != NULL ? label : "unnamed";

    LogRegion(safe_label, "default", &snapshot->default_heap);
    LogRegion(safe_label, "internal_8bit", &snapshot->internal_8bit);
    LogRegion(safe_label, "internal_dma", &snapshot->internal_dma);
    LogRegion(safe_label, "spiram_8bit", &snapshot->spiram_8bit);

    ESP_LOGI(
        kTag,
        "memory label=%s current_task_stack_high_water_bytes=%" PRIu32,
        safe_label,
        snapshot->current_task_stack_high_water_bytes
    );
}

void mer_esp_memory_probe_capture_and_log(const char* label) {
    const MerEspMemorySnapshot snapshot = mer_esp_memory_probe_capture();
    mer_esp_memory_probe_log_snapshot(
        label, 
        &snapshot
    );
}