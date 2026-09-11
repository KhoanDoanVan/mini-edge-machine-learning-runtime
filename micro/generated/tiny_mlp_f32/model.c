#include "mini_ort_generated_model/model.h"

#include <stdint.h>

#include "mini_ort_micro/kernels.h"
#include "mini_ort_micro/tensor_view.h"

static const uint32_t kInputDimensions[1] = {3};

_Alignas(MER_MODEL_ALIGNMENT) static const float kLayer0Weights[12] = {

};

static const uint32_t kLayer0WeightDimensions[2] = {3, 4};

static const uint32_t kLayer0OutputDimensions[1] = {4};

_Alignas(MER_MODEL_ALIGNMENT) static const float kLayer0Bias[4] = {

};

static const uint32_t kLayer0BiasDimensions[1] = {4};

_Alignas(MER_MODEL_ALIGNMENT) static const float kLayer2Weights[8] = {

};

static const uint32_t kLayer2WeightDimensions[2] = {4, 2};

static const uint32_t kLayer2OutputDimensions[1] = {2};

_Alignas(MER_MODEL_ALIGNMENT) static const float kLayer2Bias[2] = {

};

static const uint32_t kLayer2BiasDimensions[1] = {2};

MerMicroBufferRequirements mer_model_buffer_requirements(void) {
  const MerMicroBufferRequirements requirements = {
      .arena_bytes = MER_MODEL_ARENA_BYTES,
      .scratch_bytes = MER_MODEL_SCRATCH_BYTES,
      .alignment = MER_MODEL_ALIGNMENT,
  };
  return requirements;
}

MerMicroStatus mer_model_invoke_f32(
    MerMicroContext* context,
    const float* input,
    size_t input_elements,
    float* output,
    size_t output_elements) {
  if (context == NULL || input == NULL || output == NULL) {
    return MER_MICRO_STATUS_INVALID_ARGUMENT;
  }
  if (input_elements != MER_MODEL_INPUT_ELEMENTS ||
      output_elements != MER_MODEL_OUTPUT_ELEMENTS) {
    return MER_MICRO_STATUS_SHAPE_MISMATCH;
  }

  MerMicroConstTensorView input_view;
  MerMicroStatus status = mer_micro_make_const_tensor_view(
      input,
      MER_MICRO_DATA_TYPE_FLOAT32,
      kInputDimensions,
      1,
      &input_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  void* layer_0_output_data = NULL;
  status = mer_micro_arena_data(
      context,
      0,
      16,
      _Alignof(float),
      &layer_0_output_data);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  MerMicroConstTensorView layer_0_weight_view;
  status = mer_micro_make_const_tensor_view(
      kLayer0Weights,
      MER_MICRO_DATA_TYPE_FLOAT32,
      kLayer0WeightDimensions,
      2,
      &layer_0_weight_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  MerMicroTensorView layer_0_output_view;
  status = mer_micro_make_tensor_view(
      layer_0_output_data,
      MER_MICRO_DATA_TYPE_FLOAT32,
      kLayer0OutputDimensions,
      1,
      &layer_0_output_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  MerMicroConstTensorView layer_0_bias_view;
  status = mer_micro_make_const_tensor_view(
      kLayer0Bias,
      MER_MICRO_DATA_TYPE_FLOAT32,
      kLayer0BiasDimensions,
      1,
      &layer_0_bias_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  status = mer_micro_fully_connected_f32(
      &input_view, &layer_0_weight_view, &layer_0_bias_view, &layer_0_output_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  const MerMicroConstTensorView layer_0_output_const =
      mer_micro_as_const_tensor_view(&layer_0_output_view);
  status = mer_micro_relu_f32(&layer_0_output_const, &layer_0_output_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  MerMicroConstTensorView layer_2_weight_view;
  status = mer_micro_make_const_tensor_view(
      kLayer2Weights,
      MER_MICRO_DATA_TYPE_FLOAT32,
      kLayer2WeightDimensions,
      2,
      &layer_2_weight_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  MerMicroTensorView layer_2_output_view;
  status = mer_micro_make_tensor_view(
      output,
      MER_MICRO_DATA_TYPE_FLOAT32,
      kLayer2OutputDimensions,
      1,
      &layer_2_output_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  MerMicroConstTensorView layer_2_bias_view;
  status = mer_micro_make_const_tensor_view(
      kLayer2Bias,
      MER_MICRO_DATA_TYPE_FLOAT32,
      kLayer2BiasDimensions,
      1,
      &layer_2_bias_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  status = mer_micro_fully_connected_f32(
      &layer_0_output_const, &layer_2_weight_view, &layer_2_bias_view, &layer_2_output_view);
  if (status != MER_MICRO_STATUS_OK) {
    return status;
  }
  return MER_MICRO_STATUS_OK;
}
