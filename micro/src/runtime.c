#include "mini_ort_micro/runtime.h"

#include <stdint.h>

static int IsPowerOfTwo(size_t value) {
    /* Buffer alignment is validated as a power of two for mask-based checks. */
    return value != 0 && (value & (value - 1)) == 0;
}

/** @brief Return whether `pointer` satisfies the requested power-of-two alignment. */
static int IsAligned(
    const void* pointer,
    size_t alignment
) {
    return ((uintptr_t)pointer & (alignment - 1)) == 0;
}


static MerMicroStatus ValidateBuffer(
    const void* buffer,
    size_t supplied_bytes,
    size_t required_bytes,
    size_t alignment,
    MerMicroStatus too_small_status
) {
    /* Validate capacity before dereferencing the optional buffer pointer. */
    if (supplied_bytes < required_bytes) {
        return too_small_status;
    }
    if (supplied_bytes > 0 && buffer == NULL) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }
    if (required_bytes > 0 && !IsAligned(buffer, alignment)) {
        return MER_MICRO_STATUS_MISALIGNED_BUFFER;
    }
    return MER_MICRO_STATUS_OK;
}

static MerMicroStatus ResolveBuffer(
    const uint8_t* buffer,
    size_t buffer_bytes,
    size_t offset,
    size_t bytes,
    size_t alignment,
    void** output
) {
    /* Resolve a subrange only after checking output, bounds, and alignment. */
    if (output == NULL) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    *output = NULL;

    if (buffer == NULL || bytes == 0 || !IsPowerOfTwo(alignment)) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    if (offset > buffer_bytes || bytes > buffer_bytes - offset) {
        return MER_MICRO_STATUS_OUT_OF_RANGE;
    }

    const uint8_t* resolved = buffer + offset;

    if (!IsAligned(resolved, alignment)) {
        return MER_MICRO_STATUS_MISALIGNED_BUFFER;
    }

    *output = (void*)resolved;
    return MER_MICRO_STATUS_OK;
}

const char* mer_micro_status_string(MerMicroStatus status) {
    /* Keep diagnostics allocation-free and safe to call on embedded targets. */
    switch (status) {
        case MER_MICRO_STATUS_OK:
            return "ok";
        case MER_MICRO_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case MER_MICRO_STATUS_ARENA_TOO_SMALL:
            return "arena too small";
        case MER_MICRO_STATUS_SCRATCH_TOO_SMALL:
            return "scratch too small";
        case MER_MICRO_STATUS_MISALIGNED_BUFFER:
            return "misaligned buffer";
        case MER_MICRO_STATUS_SHAPE_MISMATCH:
            return "shape mismatch";
        case MER_MICRO_STATUS_UNSUPPORTED_TYPE:
            return "unsupported data type";
        case MER_MICRO_STATUS_OUT_OF_RANGE:
            return "buffer range is out of bounds";
    }
    return "unknown status";
}


MerMicroStatus mer_micro_init(
    MerMicroContext* context,
    void* arena,
    size_t arena_bytes,
    void* scratch,
    size_t scratch_bytes,
    const MerMicroBufferRequirements* requirements
) {
    /* Initialize the context only after both caller-owned buffers validate. */
    if (context == NULL || requirements == NULL || !IsPowerOfTwo(requirements->alignment)) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    context->arena = NULL;
    context->arena_bytes = 0;
    context->scratch = NULL;
    context->scratch_bytes = 0;
    context->alignment = 0;

    MerMicroStatus status = ValidateBuffer(
        arena,
        arena_bytes,
        requirements->arena_bytes,
        requirements->alignment,
        MER_MICRO_STATUS_ARENA_TOO_SMALL
    );

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    status = ValidateBuffer(
        scratch,
        scratch_bytes,
        requirements->scratch_bytes,
        requirements->alignment,
        MER_MICRO_STATUS_SCRATCH_TOO_SMALL
    );

    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    context->arena = (uint8_t*)arena;
    context->arena_bytes = arena_bytes;
    context->scratch = (uint8_t*)scratch;
    context->scratch_bytes = scratch_bytes;
    context->alignment = requirements->alignment;
    return MER_MICRO_STATUS_OK;
}

MerMicroStatus mer_micro_arena_data(
    const MerMicroContext* context,
    size_t offset,
    size_t bytes,
    size_t alignment,
    void** output
) {
    /* Resolve a bounded aligned range from persistent arena storage. */
    if (context == NULL) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }
    return ResolveBuffer(
        context->arena, 
        context->arena_bytes, 
        offset, 
        bytes, 
        alignment, 
        output
    );
}

MerMicroStatus mer_micro_scratch_data(
    const MerMicroContext* context,
    size_t offset,
    size_t bytes,
    size_t alignment,
    void** output
) {
    /* Resolve a bounded aligned range from temporary scratch storage. */
    if (context == NULL) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }
    return ResolveBuffer(
        context->scratch,
        context->scratch_bytes,
        offset,
        bytes,
        alignment,
        output
    );
}
