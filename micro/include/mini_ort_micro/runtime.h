#ifndef MINI_ORT_MICRO_RUNTIME_H_
#define MINI_ORT_MICRO_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#include "mini_ort_micro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Memory capacities required by a micro runtime instance. */
typedef struct MerMicroBufferRequirements {
    /** @brief Required bytes for persistent arena storage. */
    size_t arena_bytes;
    /** @brief Required bytes for temporary scratch storage. */
    size_t scratch_bytes;
    /** @brief Required power-of-two alignment for supplied buffers. */
    size_t alignment;
} MerMicroBufferRequirements;

/** @brief Caller-owned memory and layout state for one runtime instance. */
typedef struct MerMicroContext {
    /** @brief Base address of persistent arena storage. */
    uint8_t* arena;
    /** @brief Available bytes in persistent arena storage. */
    size_t arena_bytes;
    /** @brief Base address of temporary scratch storage. */
    uint8_t* scratch;
    /** @brief Available bytes in temporary scratch storage. */
    size_t scratch_bytes;
    /** @brief Alignment accepted for buffers resolved by this context. */
    size_t alignment;
} MerMicroContext;

/**
 * @brief Initialize a runtime context with caller-provided buffers.
 * @param context Context to initialize.
 * @param arena Persistent arena buffer and capacity.
 * @param arena_bytes Number of bytes available in `arena`.
 * @param scratch Temporary scratch buffer and capacity.
 * @param scratch_bytes Number of bytes available in `scratch`.
 * @param requirements Required capacities and alignment.
 * @return A status describing validation success or failure.
 */
MerMicroStatus mer_micro_init(
    MerMicroContext* context,
    void* arena,
    size_t arena_bytes,
    void* scratch,
    size_t scratch_bytes,
    const MerMicroBufferRequirements* requirements
);

/**
 * @brief Resolve an aligned range within the scratch buffer.
 * @param context Initialized runtime context.
 * @param offset Byte offset from the start of scratch storage.
 * @param bytes Number of bytes requested.
 * @param alignment Required power-of-two alignment for the returned address.
 * @param output Receives the resolved address on success.
 * @return A status describing validation success or failure.
 */
MerMicroStatus mer_micro_scratch_data(
    const MerMicroContext* context,
    size_t offset,
    size_t bytes,
    size_t alignment,
    void** output
);

/** @brief Resolve an aligned range within the scratch buffer. */
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
