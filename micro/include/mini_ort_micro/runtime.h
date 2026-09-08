#ifndef MINI_ORT_MICRO_RUNTIME_H_
#define MINI_ORT_MICRO_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#include "mini_ort_micro/status.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct MerMicroBufferRequirements {
    size_t arena_bytes;
    size_t scratch_bytes;
    size_t alignment;
} MerMicroBufferRequirements;


typedef struct MerMicroContext {
    uint8_t* arena;
    size_t arena_bytes;
    uint8_t* scratch;
    size_t scratch_bytes;
    size_t alignment;
} MerMicroContext;

MerMicroStatus mer_micro_init(
    MerMicroContext* context,
    void* arena,
    size_t arena_bytes,
    void* scratch,
    size_t scratch_bytes,
    const MerMicroBufferRequirements* requirements
);

MerMicroStatus mer_micro_scratch_data(
    const MerMicroContext* context,
    size_t offset,
    size_t bytes,
    size_t alignment,
    void** output
);

MerMicroStatus mer_micro_scratch_datat(
    const MerMicroContext* context,
    size_t offset,
    size_t bytes,
    size_t alignment,
    void**output
);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // MINI_ORT_MICRO_STATUS_H_