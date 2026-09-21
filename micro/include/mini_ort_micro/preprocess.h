#ifndef MINI_ORT_MICRO_PREPROCESS_H_
#define MINI_ORT_MICRO_PREPROCESS_H_

#include <stdint.h>

#include "mini_ort_micro/status.h"
#include "mini_ort_micro/tensor_view.h"


#ifdef __cplusplus
extern "C" {
#endif


#define MER_MICRO_GRAYSCALE_FEATURE_COUNT 3

typedef enum MerMicroGrayscaleFeatureIndex {
    MER_MICRO_GRAYSCALE_FEATURE_MEAN = 0,
    MER_MICRO_GRAYSCALE_FEATURE_DARK_RATIO = 1,
    MER_MICRO_GRAYSCALE_FEATURE_BRIGHT_RATIO = 2,
} MerMicroGrayscaleFeatureIndex;

typedef struct MerMicroGrayscaleFeatureConfig {
    uint8_t dark_threshold;
    uint8_t bright_threshold;
} MerMicroGrayscaleFeatureConfig;

// Converts a contiguous rank-2 UINT8 grayscale image into a rank-1 FLOAT32
// feature tensor: normalized mean, dark-pixel ratio, and bright-pixel ratio.
// This operator allocates no memory and writes exactly three output elements.
MerMicroStatus mer_micro_grayscale_features_f32(
    const MerMicroConstTensorView* input,
    const MerMicroGrayscaleFeatureConfig* config,
    MerMicroTensorView* output
);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MINI_ORT_MICRO_PREPROCESS_H_