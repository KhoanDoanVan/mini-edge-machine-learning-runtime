#ifndef MINI_ORT_MICRO_STATUS_H_
#define MINI_ORT_MICRO_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MerMicroStatus {
    MER_MICRO_STATUS_OK = 0,
    MER_MICRO_STATUS_INVALID_ARGUMENT = 1,
    MER_MICRO_STATUS_ARENA_TOO_SMALL = 2,
    MER_MICRO_STATUS_SCRATCH_TOO_SMALL = 3,
    MER_MICRO_STATUS_MISALIGNED_BUFFER = 4,
    MER_MICRO_STATUS_SHAPE_MISMATCH = 5,
    MER_MICRO_STATUS_UNSUPPORTED_TYPE = 6,
    MER_MICRO_STATUS_OUT_OF_RANGE = 7,
} MerMicroStatus;

const char* mer_micro_status_string(MerMicroStatus status);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // MINI_ORT_MICRO_STATUS_H_