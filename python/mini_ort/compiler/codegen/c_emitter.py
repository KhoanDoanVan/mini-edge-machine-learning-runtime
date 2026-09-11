"""C11 source emission for the Micro/AOT runtime."""

from __future__ import annotations

from ..errors import ModelFormatError
from ..ir import LinearLayer, Model
from ..memory_planner import MemoryPlan
from .literals import aligned_float32_array



def generate_header(
    model: Model, 
    memory_plan: MemoryPlan
) -> str:
    arena_bytes = memory_plan.arena_bytes
    storage_bytes = max(1, arena_bytes)

    return f"""#ifndef MINI_ORT_GENERATED_MODEL_MODEL_H_
#define MINI_ORT_GENERATED_MODEL_MODEL_H_

#include <stddef.h>

#include "mini_ort_micro/runtime.h"
#include "mini_ort_micro/status.h"

#ifdef __cplusplus
extern "C" {{
#endif

#define MER_MODEL_INPUT_ELEMENTS {model.input_features}
#define MER_MODEL_OUTPUT_ELEMENTS {model.output_features}
#define MER_MODEL_ARENA_BYTES {arena_bytes}
#define MER_MODEL_ARENA_STORAGE_BYTES {storage_bytes}
#define MER_MODEL_SCRATCH_BYTES 0
#define MER_MODEL_SCRATCH_STORAGE_BYTES 1
#define MER_MODEL_ALIGNMENT {memory_plan.alignment}

MerMicroBufferRequirements mer_model_buffer_requirements(void);

MerMicroStatus mer_model_invoke_f32(
    MerMicroContext* context,
    const float* input,
    size_t input_elements,
    float* output,
    size_t output_elements);

#ifdef __cplusplus
}}  // extern "C"
#endif

#endif  // MINI_ORT_GENERATED_MODEL_MODEL_H_
"""

def generate_source(
    model: Model, 
    memory_plan: MemoryPlan
) -> str:
    destinations = memory_plan.destinations
    declarations: list[str] = []
    body: list[str] = []

    input_dims = "kInputDimensions"
    declarations.append(
        f"static const uint32_t {input_dims}[1] = {{{model.input_features}}};"
    )
    
    current_const = "input_view"
    current_mutable: str | None = None

    body.extend(
        [
            "  MerMicroConstTensorView input_view;",
            "  MerMicroStatus status = mer_micro_make_const_tensor_view(",
            "      input,",
            "      MER_MICRO_DATA_TYPE_FLOAT32,",
            f"      {input_dims},",
            "      1,",
            "      &input_view);",
            "  if (status != MER_MICRO_STATUS_OK) {",
            "    return status;",
            "  }",
        ]
    )

    for position, layer in enumerate(model.layers):

        # Linear
        if isinstance(layer, LinearLayer):
            prefix = f"layer_{layer.index}"
            weight_name = f"kLayer{layer.index}Weights"
            weight_dims = f"kLayer{layer.index}WeightDimensions"
            output_dims = f"kLayer{layer.index}OutputDimensions"

            declarations.extend(
                [
                    aligned_float32_array(weight_name, layer.weights),
                    f"static const uint32_t {weight_dims}[2] = "
                    f"{{{layer.in_features}, {layer.out_features}}};",
                    f"static const uint32_t {output_dims}[1] = "
                    f"{{{layer.out_features}}};",
                ]
            )

            bias_name = f"kLayer{layer.index}Bias"

            if layer.bias:
                declarations.append(
                    aligned_float32_array(bias_name, layer.bias)
                )

            destination = destinations[layer.index]
            output_pointer = "output"

            if destination.kind == "arena":
                output_pointer = f"{prefix}_output_data"
                body.extend(
                    [
                        f"  void* {output_pointer} = NULL;",
                        "  status = mer_micro_arena_data(",
                        "      context,",
                        f"      {destination.offset},",
                        f"      {destination.byte_count},",
                        "      _Alignof(float),",
                        f"      &{output_pointer});",
                        "  if (status != MER_MICRO_STATUS_OK) {",
                        "    return status;",
                        "  }",
                    ]
                )

            weight_view = f"{prefix}_weight_view"
            output_view = f"{prefix}_output_view"

            body.extend(
                [
                    f"  MerMicroConstTensorView {weight_view};",
                    "  status = mer_micro_make_const_tensor_view(",
                    f"      {weight_name},",
                    "      MER_MICRO_DATA_TYPE_FLOAT32,",
                    f"      {weight_dims},",
                    "      2,",
                    f"      &{weight_view});",
                    "  if (status != MER_MICRO_STATUS_OK) {",
                    "    return status;",
                    "  }",
                    f"  MerMicroTensorView {output_view};",
                    "  status = mer_micro_make_tensor_view(",
                    f"      {output_pointer},",
                    "      MER_MICRO_DATA_TYPE_FLOAT32,",
                    f"      {output_dims},",
                    "      1,",
                    f"      &{output_view});",
                    "  if (status != MER_MICRO_STATUS_OK) {",
                    "    return status;",
                    "  }",
                ]
            )

            bias_argument = "NULL"

            if layer.bias:
                bias_dims = f"kLayer{layer.index}BiasDimensions"
                bias_view = f"{prefix}_bias_view"
                declarations.append(
                    f"static const uint32_t {bias_dims}[1] = "
                    f"{{{layer.out_features}}};"
                )
                body.extend(
                    [
                        f"  MerMicroConstTensorView {bias_view};",
                        "  status = mer_micro_make_const_tensor_view(",
                        f"      {bias_name},",
                        "      MER_MICRO_DATA_TYPE_FLOAT32,",
                        f"      {bias_dims},",
                        "      1,",
                        f"      &{bias_view});",
                        "  if (status != MER_MICRO_STATUS_OK) {",
                        "    return status;",
                        "  }",
                    ]
                )
                bias_argument = f"&{bias_view}"

            body.extend(
                [
                    "  status = mer_micro_fully_connected_f32(",
                    f"      &{current_const}, &{weight_view}, "
                    f"{bias_argument}, &{output_view});",
                    "  if (status != MER_MICRO_STATUS_OK) {",
                    "    return status;",
                    "  }",
                ]
            )

            if position + 1 < len(model.layers):
                body.extend(
                    [
                        f"  const MerMicroConstTensorView "
                        f"{prefix}_output_const =",
                        f"      mer_micro_as_const_tensor_view(&{output_view});",
                    ]
                )

                current_const = f"{prefix}_output_const"

            current_mutable = output_view

        else:
            if current_mutable is None:
                raise ModelFormatError("ReLU has no mutable activation")

            body.extend(
                [
                    f"  status = mer_micro_relu_f32(&{current_const}, "
                    f"&{current_mutable});",
                    "  if (status != MER_MICRO_STATUS_OK) {",
                    "    return status;",
                    "  }",
                ]
            )

    declarations_text = "\n\n".join(declarations)
    body_text = "\n".join(body)

    return f"""#include "mini_ort_generated_model/model.h"

#include <stdint.h>

#include "mini_ort_micro/kernels.h"
#include "mini_ort_micro/tensor_view.h"

{declarations_text}

MerMicroBufferRequirements mer_model_buffer_requirements(void) {{
  const MerMicroBufferRequirements requirements = {{
      .arena_bytes = MER_MODEL_ARENA_BYTES,
      .scratch_bytes = MER_MODEL_SCRATCH_BYTES,
      .alignment = MER_MODEL_ALIGNMENT,
  }};
  return requirements;
}}

MerMicroStatus mer_model_invoke_f32(
    MerMicroContext* context,
    const float* input,
    size_t input_elements,
    float* output,
    size_t output_elements) {{
  if (context == NULL || input == NULL || output == NULL) {{
    return MER_MICRO_STATUS_INVALID_ARGUMENT;
  }}
  if (input_elements != MER_MODEL_INPUT_ELEMENTS ||
      output_elements != MER_MODEL_OUTPUT_ELEMENTS) {{
    return MER_MICRO_STATUS_SHAPE_MISMATCH;
  }}

{body_text}
  return MER_MICRO_STATUS_OK;
}}
"""