#ifndef MINI_ORT_MICRO_ESP_MEMORY_PROBE_H_
#define MINI_ORT_MICRO_ESP_MEMORY_PROBE_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Measured capacity and availability for one heap region. */
typedef struct MerEspMemoryRegionStats {
    /** @brief Total bytes provided by the region. */
    size_t total_bytes;
    /** @brief Bytes currently available for allocation. */
    size_t free_bytes;
    /** @brief Lowest observed free-byte count since boot. */
    size_t minimum_free_bytes;
    /** @brief Size of the largest currently available contiguous block. */
    size_t largest_free_block_bytes;
} MerEspMemoryRegionStats;

/** @brief Snapshot of device memory regions and the current task stack. */
typedef struct MerEspMemorySnapshot {
    /** @brief Statistics for the default-capability heap. */
    MerEspMemoryRegionStats default_heap;
    /** @brief Statistics for internal 8-bit-capable memory. */
    MerEspMemoryRegionStats internal_8bit;
    /** @brief Statistics for internal DMA-capable memory. */
    MerEspMemoryRegionStats internal_dma;
    /** @brief Statistics for 8-bit-capable external SPIRAM. */
    MerEspMemoryRegionStats spiram_8bit;
    /** @brief Current task stack high-water mark in bytes. */
    uint32_t current_task_stack_high_water_bytes;
} MerEspMemorySnapshot;

/** @brief Log chip, SDK, CPU, flash, and SPIRAM information. */
void mer_esp_memory_probe_log_device_info(void);

/** @brief Capture current heap-region and task-stack statistics. */
MerEspMemorySnapshot mer_esp_memory_probe_capture(void);

/**
 * @brief Log all regions and task-stack statistics in a captured snapshot.
 * @param label Label included in each log entry; null uses a default label.
 * @param snapshot Snapshot to log; null is rejected and logged as an error.
 */
void mer_esp_memory_probe_log_snapshot(
    const char* label,
    const MerEspMemorySnapshot* snapshot
);

/** @brief Capture and immediately log a memory snapshot. */
void mer_esp_memory_probe_capture_and_log(const char* label);


#ifdef __cplusplus
} // extern "C"
#endif

#endif  // MINI_ORT_MICRO_ESP_MEMORY_PROBE_H_
