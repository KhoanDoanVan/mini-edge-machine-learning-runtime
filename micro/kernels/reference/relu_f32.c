#include "mini_ort_micro/kernels.h"

#include <stdint.h>

static int RangesOverlap(
    const void* first,
    size_t first_bytes,
    const void* second,
    size_t second_bytes
) {
    const uintptr_t first_begin = (uintptr_t)first;
    const uintptr_t second_begin = (uintptr_t)second;

    if (first_bytes > UINTPTR_MAX - first_begin || second_bytes > UINTPTR_MAX - second_begin) {
        return 1;
    }

    const uintptr_t first_end = first_begin + first_bytes;
    const uintptr_t second_end = second_begin + second_bytes;

    return first_begin < second_end && second_begin < first_end;
}


MerMicroStatus mer_micro_relu_f32(
    const MerMicroConstTensorView* input,
    MerMicroTensorView* output
) {
    MerMicroStatus status = mer_micro_validate_const_tensor_view(input);

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    status = mer_micro_validate_tensor_view(output);

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    if (input->data_type != MER_MICRO_DATA_TYPE_FLOAT32 || output->data_type != MER_MICRO_DATA_TYPE_FLOAT32) {
        return MER_MICRO_STATUS_UNSUPPORTED_TYPE;
    }

    if (input->rank != output->rank || input->element_count != output->element_count) {
        return MER_MICRO_STATUS_SHAPE_MISMATCH;
    }

    for (uint32_t dimension = 0; dimension < input->rank; ++dimension) {
        if (input->dimensions[dimension] != output->dimensions[dimension]) {
            return MER_MICRO_STATUS_SHAPE_MISMATCH;s
        }
    }

    const size_t bytes = input->element_count * sizeof(float);

    if (input->data != output->data && RangesOverlap(
        input->data,
        bytes,
        output->data,
        bytes
    )) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    const float* input_data = (const float*)input->data;
    float* output_data = (float*)output->data;

    for (size_t element = 0; element < input->element_count; ++element) {
        const float value = input_data[element];
        output_data[element] = value > 0.0F ? value : 0.0F;
    }

    return MER_MICRO_STATUS_OK;
}