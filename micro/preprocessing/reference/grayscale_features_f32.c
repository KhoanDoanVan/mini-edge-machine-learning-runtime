#include "mini_ort_micro/preprocess.h"


#include <stddef.h>
#include <stdint.h>

MerMicroStatus mer_micro_grayscale_features_f32(
    const MerMicroConstTensorView* input,
    const MerMicroGrayscaleFeatureConfig* config,
    MerMicroTensorView* output
) {
    if (config == NULL) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    if (config->dark_threshold >= config->bright_threshold) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    MerMicroStatus status = mer_micro_validate_const_tensor_view(input);

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    status = mer_micro_validate_tensor_view(output);

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    if (input->data_type != MER_MICRO_DATA_TYPE_UINT8 
        || output->data_type != MER_MICRO_DATA_TYPE_FLOAT32
    ) {
        return MER_MICRO_STATUS_UNSUPPORTED_TYPE;
    }

    if (input->rank != 2 || output->rank != 1 
        || output->element_count != MER_MICRO_GRAYSCALE_FEATURE_COUNT
    ) {
        return MER_MICRO_STATUS_SHAPE_MISMATCH;
    }

    const uint8_t* pixels = (const uint8_t*)input->data;
    float brightness_sum = 0.0F;
    size_t dark_count = 0;
    size_t bright_count = 0;

    for (size_t index = 0; index < input->element_count; ++index) {
        const uint8_t pixel = pixels[index];
        brightness_sum += (float)pixel;
        dark_count += pixel <= config->dark_threshold;
        bright_count += pixel >= config->bright_threshold;
    }

    const float pixel_count = (float)input->element_count;
    float* features = (float*)output->data;
    features[MER_MICRO_GRAYSCALE_FEATURE_MEAN] = brightness_sum / (255.0F * pixel_count);
    features[MER_MICRO_GRAYSCALE_FEATURE_DARK_RATIO] = (float)dark_count / pixel_count;
    features[MER_MICRO_GRAYSCALE_FEATURE_BRIGHT_RATIO] = (float)bright_count / pixel_count;

    return MER_MICRO_STATUS_OK;
}