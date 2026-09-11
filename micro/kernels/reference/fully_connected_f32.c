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

static int IsFloat32Const(const MerMicroConstTensorView* view) {
    return view != NULL && view->data_type == MER_MICRO_DATA_TYPE_FLOAT32;
}

static int IsFloat32Mutable(const MerMicroTensorView* view) {
    return view != NULL && view->data_type == MER_MICRO_DATA_TYPE_FLOAT32;
}

MerMicroStatus mer_micro_fully_connected_f32(
    const MerMicroConstTensorView* input,
    const MerMicroConstTensorView* weights,
    const MerMicroConstTensorView* bias,
    MerMicroTensorView* output
) {
    MerMicroStatus status = mer_micro_validate_const_tensor_view(input);

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }
    status = mer_micro_validate_const_tensor_view(weights);

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }
    status = mer_micro_validate_tensor_view(output);

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    if (bias != NULL) {
        status = mer_micro_validate_const_tensor_view(bias);
        if (status != MER_MICRO_STATUS_OK) {
        return status;
        }
    }

    if (!IsFloat32Const(input) || !IsFloat32Const(weights) || !IsFloat32Mutable(output) || (bias != NULL && !IsFloat32Const(bias))) {
        return MER_MICRO_STATUS_UNSUPPORTED_TYPE;
    }

    if (
        input->rank != 1 || weights->rank != 2 || output->rank != 1 ||
        weights->dimensions[0] != input->dimensions[0] ||
        weights->dimensions[1] != output->dimensions[0] ||
      (bias != NULL && (bias->rank != 1 || bias->dimensions[0] != output->dimensions[0]))
    ) {
        return MER_MICRO_STATUS_SHAPE_MISMATCH;
    }

    const size_t input_bytes = input->element_count * sizeof(float);
    const size_t weight_bytes = weights->element_count * sizeof(float);
    const size_t output_bytes = output->element_count * sizeof(float);
    const size_t bias_bytes = bias != NULL ? bias->element_count * sizeof(float) : 0;

    if (
        RangesOverlap(
            output->data,
            output_bytes,
            input->data,
            input_bytes
        ) || RangesOverlap(
            output->data,
            output_bytes,
            weights->data,
            weight_bytes
        ) || (
            bias != NULL && RangesOverlap(
                output->data,
                output_bytes,
                bias->data,
                bias_bytes
            )
        )
    ) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    const size_t input_features = input->dimensions[0];
    const size_t output_features = output->dimensions[0];
    const float* input_data = (const float*)input->data;
    const float* weight_data = (const float*)weights->data;
    const float* bias_data = bias != NULL ? (const float*)bias->data : NULL;
    float* output_data = (float*)output->data;

    for (size_t column = 0; column < output_features; ++column) {
        output_data[column] = bias_data != NULL ? bias_data[column] : 0.0F;
    }

    // Input-major traversal matches row-major [input, output] .mer weights and
    // keeps both weights and output contiguous in the inner loop.
    for (size_t reduction = 0; reduction < input_features; ++reduction) {
        const float input_value = input_data[reduction];
        const size_t weight_row = reduction * output_features;

        for (size_t column = 0; column < output_features; ++column) {
            output_data[column] += input_value * weight_data[weight_row + column];
        }
    }

    return MER_MICRO_STATUS_OK;

}