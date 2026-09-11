#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>

#include "mini_ort_micro/kernels.h"
#include "mini_ort_micro/runtime.h"

static MerMicroStatus MakeConstF32(
    const float* data,
    const uint32_t* dimensions,
    uint32_t rank,
    MerMicroConstTensorView* output
) {
    return mer_micro_make_const_tensor_view(data, MER_MICRO_DATA_TYPE_FLOAT32, dimensions, rank, output);
}

static MerMicroStatus MakeF32(
    float* data,
    const uint32_t* dimensions,
    uint32_t rank,
    MerMicroTensorView* output
) {
  return mer_micro_make_tensor_view(data, MER_MICRO_DATA_TYPE_FLOAT32, dimensions, rank, output);
}

static int Check(MerMicroStatus status, const char* operation) {
    if (status == MER_MICRO_STATUS_OK) {
        return 1;
    }
    fprintf(stderr, "%s failed: %s\n", operation, mer_micro_status_string(status));
    return 0;
}

int main(void) {
    static const float input_data[3] = {1.0F, 2.0F, 3.0F};
    static const float first_weights[3 * 4] = {
        1.0F, 0.0F, -1.0F, 0.5F,
        0.0F, 1.0F, 1.0F, -1.0F,
        1.0F, 1.0F, 0.0F, 0.5F,
    };
    static const float first_bias[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    static const float second_weights[4 * 2] = {
        1.0F, 0.0F,
        0.0F, 1.0F,
        1.0F, 1.0F,
        -1.0F, 0.5F,
    };
    static const float second_bias[2] = {0.5F, -0.5F};

    alignas(16) uint8_t arena[4 * sizeof(float)];
    alignas(16) float output_data[2];

    const MerMicroBufferRequirements requirements = {
        .arena_bytes = sizeof(arena),
        .scratch_bytes = 0,
        .alignment = 16,
    };

    MerMicroContext context;
    if (!Check(
            mer_micro_init(
                &context, arena, sizeof(arena), NULL, 0, &requirements),
            "runtime init")
        ) {
        return 1;
    }

    void* hidden_data = NULL;
    if (!Check(
            mer_micro_arena_data(
                &context, 0, sizeof(arena), alignof(float), &hidden_data),
            "hidden arena resolve")) {
        return 1;
    }

    const uint32_t input_shape[1] = {3};
    const uint32_t hidden_shape[1] = {4};
    const uint32_t output_shape[1] = {2};
    const uint32_t first_weight_shape[2] = {3, 4};
    const uint32_t second_weight_shape[2] = {4, 2};

    MerMicroConstTensorView input;
    MerMicroConstTensorView first_weight;
    MerMicroConstTensorView first_bias_view;
    MerMicroTensorView hidden;
    MerMicroConstTensorView second_weight;
    MerMicroConstTensorView second_bias_view;
    MerMicroTensorView output;

    if (!Check(MakeConstF32(input_data, input_shape, 1, &input), "input view") ||
        !Check(
            MakeConstF32(first_weights, first_weight_shape, 2, &first_weight),
            "first weight view") ||
        !Check(
            MakeConstF32(first_bias, hidden_shape, 1, &first_bias_view),
            "first bias view") ||
        !Check(MakeF32((float*)hidden_data, hidden_shape, 1, &hidden),
                "hidden view") ||
        !Check(
            MakeConstF32(second_weights, second_weight_shape, 2, &second_weight),
            "second weight view") ||
        !Check(
            MakeConstF32(second_bias, output_shape, 1, &second_bias_view),
            "second bias view") ||
        !Check(MakeF32(output_data, output_shape, 1, &output), "output view")) {
        return 1;
    }

    if (!Check(
            mer_micro_fully_connected_f32(
                &input, &first_weight, &first_bias_view, &hidden),
            "first fully connected")) {
        return 1;
    }

    const MerMicroConstTensorView hidden_const = mer_micro_as_const_tensor_view(&hidden);
    if (!Check(mer_micro_relu_f32(&hidden_const, &hidden), "in-place relu")) {
        return 1;
    }

    if (!Check(
            mer_micro_fully_connected_f32(
                &hidden_const, &second_weight, &second_bias_view, &output),
            "second fully connected")) {
        return 1;
    }

    printf("output=[%.2f, %.2f] arena_bytes=%zu scratch_bytes=%zu\n",
            output_data[0],
            output_data[1],
            context.arena_bytes,
            context.scratch_bytes);

    return output_data[0] == 5.5F && output_data[1] == 5.5F ? 0 : 1;
}
