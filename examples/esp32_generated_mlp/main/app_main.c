#include <stdalign.h>
#include <stdint.h>

#include "esp_log.h"
#include "mini_ort_generated_model/model.h"
#include "mini_ort_micro/esp_memory_probe.h"
#include "mini_ort_micro/runtime.h"


static const char* kTag = "mini_ort_mlp";


alignas(MER_MODEL_ALIGNMENT) static uint8_t g_arena[MER_MODEL_ARENA_STORAGE_BYTES];

void app_main(void) {

    static const float input[MER_MODEL_INPUT_ELEMENTS] = {1.0F, 2.0F, 3.0F};

    float output[MER_MODEL_OUTPUT_ELEMENTS];

    mer_esp_memory_probe_log_device_info();

    mer_esp_memory_probe_capture_and_log("before_runtime_init");

    const MerMicroBufferRequirements requirements = mer_model_buffer_requirements();

    MerMicroContext context;
    MerMicroStatus status = mer_micro_init(
        &context,
        g_arena,
        MER_MODEL_ARENA_BYTES,
        NULL,
        MER_MODEL_SCRATCH_BYTES,
        &requirements
    );

    if (status == MER_MICRO_STATUS_OK) {
        status = mer_model_invoke_f32(
            &context,
            input,
            MER_MODEL_INPUT_ELEMENTS,
            output,
            MER_MODEL_OUTPUT_ELEMENTS
        );
    }

    if (status != MER_MICRO_STATUS_OK) {
        ESP_LOGE(kTag, "inference failed: %s", mer_micro_status_string(status));
        return;
    }

    ESP_LOGI(
        kTag,
        "output=[%.2f, %.2f] arena_bytes=%zu scratch_bytes=%zu",
        output[0],
        output[1],
        requirements.arena_bytes,
        requirements.scratch_bytes
    );

    mer_esp_memory_probe_capture_and_log("after_inference");

}