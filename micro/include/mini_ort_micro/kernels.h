#ifndef MINI_ORT_MICRO_KERNELS_H_
#define MINI_ORT_MICRO_KERNELS_H_

#include "mini_ort_micro/status.h"
#include "mini_ort_micro/tensor_view.h"


#ifdef __cplusplus
extern "C" {
#endif


MerMicroStatus mer_micro_fully_connected_f32(
    const MerMicroConstTensorView* input,
    const MerMicroConstTensorView* weights,
    const MerMicroConstTensorView* bias,
    MerMicroTensorView* output
);

MerMicroStatus mer_micro_relu_f32(
    const MerMicroConstTensorView* input,
    MerMicroTensorView* output
);


#ifdef __cplusplus
}   // extern "C"
#endif


#endif // MINI_ORT_MICRO_KERNELS_H_