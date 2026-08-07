#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct MiniOrtStatus MiniOrtStatus;
typedef struct MiniOrtSession MiniOrtSession;
typedef struct MiniOrtValue MiniOrtValue;

typedef enum MiniOrtLayerType {
    MINI_ORT_LAYER_LINEAR = 1,
    MINI_ORT_LAYER_RELU = 2,
} MiniOrtLayerType;

typedef struct MiniOrtLayerDesc {
    MiniOrtLayerType type;
    size_t in_features;
    size_t out_features;
    const float* weight_data;
    size_t weight_count;
    const float* bias_data;
    size_t bias_count;
} MiniOrtLayerDesc;

const char* MiniOrtGetErrorMessage(const MiniOrtStatus* status);

void MiniOrtReleaseStatus(MiniOrtStatus* status);

MiniOrtStatus* MiniOrtCreateSession(const MiniOrtLayerDesc* layers, size_t layer_count, MiniOrtSession** output);

MiniOrtStatus* MiniOrtCreateSessionFromFile(
    const char* model_path,
    MiniOrtSession** output
);

void MiniOrtReleaseSession(MiniOrtSession* session);

MiniOrtSession* MiniOrtCreateFloatTensor(const float* data, size_t element_count, const int64_t* shape, size_t rank, MiniOrtValue** output);

const float* MiniOrtGetTensorData(const MiniOrtValue* value);

size_t MiniOrtGetTensorElementCount(const MiniOrtValue*  value);

const int64_t* MiniOrtGetTensorShape(const MiniOrtValue* value, size_t* rank);

void MiniOrtReleaseValue(MiniOrtValue* value);

MiniOrtStatus* MiniOrtRun(MiniOrtSession* session, const MiniOrtValue* input, MiniOrtValue** output);


#ifdef __cplusplus
}
#endif