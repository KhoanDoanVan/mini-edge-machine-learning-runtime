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

/** @brief Return the UTF-8 message stored in a status object. */
const char* MiniOrtGetErrorMessage(const MiniOrtStatus* status);

/** @brief Return the input feature count encoded by a session. */
size_t MiniOrtGetInputFeatureCount(const MiniOrtSession* session);

/** @brief Return the output feature count encoded by a session. */
size_t MiniOrtGetOutputFeatureCount(const MiniOrtSession* session);

/** @brief Release a status returned by a MiniOrt API call. */
void MiniOrtReleaseStatus(MiniOrtStatus* status);

/** @brief Create a session from caller-owned layer descriptors and weights. */
MiniOrtStatus* MiniOrtCreateSession(const MiniOrtLayerDesc* layers, size_t layer_count, MiniOrtSession** output);

/** @brief Create a session by loading a MER model from a filesystem path. */
MiniOrtStatus* MiniOrtCreateSessionFromFile(
    const char* model_path,
    MiniOrtSession** output
);

/** @brief Create a session by loading a MER model from an in-memory buffer. */
MiniOrtStatus* MiniOrtCreateSessionFromBuffer(
    const void* model_data,
    size_t model_size,
    MiniOrtSession** output
);

/** @brief Release a session and all resources it owns. */
void MiniOrtReleaseSession(MiniOrtSession* session);

/** @brief Create an owning float tensor from shape and payload data. */
MiniOrtStatus* MiniOrtCreateFloatTensor(const float* data, size_t element_count, const int64_t* shape, size_t rank, MiniOrtValue** output);

/** @brief Return a read-only pointer to a tensor's float payload. */
const float* MiniOrtGetTensorData(const MiniOrtValue* value);

/** @brief Return the number of scalar elements in a tensor value. */
size_t MiniOrtGetTensorElementCount(const MiniOrtValue*  value);

/** @brief Return tensor dimensions and write their rank to `rank`. */
const int64_t* MiniOrtGetTensorShape(const MiniOrtValue* value, size_t* rank);

/** @brief Release an owning tensor value. */
void MiniOrtReleaseValue(MiniOrtValue* value);

/** @brief Run inference and allocate an owning output value. */
MiniOrtStatus* MiniOrtRun(MiniOrtSession* session, const MiniOrtValue* input, MiniOrtValue** output);

/** @brief Run inference into an existing caller-owned output value. */
MiniOrtStatus* MiniOrtRunInto(MiniOrtSession* session, const MiniOrtValue* input, MiniOrtValue* output);


#ifdef __cplusplus
}
#endif
