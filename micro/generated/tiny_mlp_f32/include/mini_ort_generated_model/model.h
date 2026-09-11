#ifndef MINI_ORT_GENERATED_MODEL_MODEL_H_
#define MINI_ORT_GENERATED_MODEL_MODEL_H_

#include <stddef.h>

#include "mini_ort_micro/runtime.h"
#include "mini_ort_micro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MER_MODEL_INPUT_ELEMENTS 3
#define MER_MODEL_OUTPUT_ELEMENTS 2
#define MER_MODEL_ARENA_BYTES 16
#define MER_MODEL_ARENA_STORAGE_BYTES 16
#define MER_MODEL_SCRATCH_BYTES 0
#define MER_MODEL_SCRATCH_STORAGE_BYTES 1
#define MER_MODEL_ALIGNMENT 16

MerMicroBufferRequirements mer_model_buffer_requirements(void);

MerMicroStatus mer_model_invoke_f32(
    MerMicroContext* context,
    const float* input,
    size_t input_elements,
    float* output,
    size_t output_elements);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MINI_ORT_GENERATED_MODEL_MODEL_H_
