#ifndef MINI_ORT_MICRO_STATUS_H_
#define MINI_ORT_MICRO_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Result codes returned by the micro runtime APIs. */
typedef enum MerMicroStatus {
    /** @brief The operation completed successfully. */
    MER_MICRO_STATUS_OK = 0,
    /** @brief An argument was null, invalid, or otherwise unsupported. */
    MER_MICRO_STATUS_INVALID_ARGUMENT = 1,
    /** @brief The supplied arena does not have enough capacity. */
    MER_MICRO_STATUS_ARENA_TOO_SMALL = 2,
    /** @brief The supplied scratch buffer does not have enough capacity. */
    MER_MICRO_STATUS_SCRATCH_TOO_SMALL = 3,
    /** @brief A supplied buffer or resolved range has incorrect alignment. */
    MER_MICRO_STATUS_MISALIGNED_BUFFER = 4,
    /** @brief Tensor dimensions do not match the stored element count. */
    MER_MICRO_STATUS_SHAPE_MISMATCH = 5,
    /** @brief The requested data type is not supported. */
    MER_MICRO_STATUS_UNSUPPORTED_TYPE = 6,
    /** @brief A requested range or size exceeds representable or valid bounds. */
    MER_MICRO_STATUS_OUT_OF_RANGE = 7,
} MerMicroStatus;

/** @brief Return a stable human-readable message for a status code. */
const char* mer_micro_status_string(MerMicroStatus status);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // MINI_ORT_MICRO_STATUS_H_
