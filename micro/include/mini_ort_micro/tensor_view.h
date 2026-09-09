#ifndef MINI_ORT_MICRO_TENSOR_VIEW_H_
#define MINI_ORT_MICRO_TENSOR_VIEW_H_

#include <stddef.h>
#include <stdint.h>

#include "mini_ort_micro/status.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @brief Maximum number of dimensions supported by a micro tensor view. */
#define MER_MICRO_MAX_RANK 4

/** @brief Scalar element types supported by micro tensor views. */
typedef enum MerMicroDataType {
    /** @brief IEEE 754 single-precision floating-point values. */
    MER_MICRO_DATA_TYPE_FLOAT32 = 1,
    /** @brief Signed 8-bit integer values. */
    MER_MICRO_DATA_TYPE_INT8 = 2,
    /** @brief Signed 32-bit integer values. */
    MER_MICRO_DATA_TYPE_INT32 = 3,
} MerMicroDataType;

/** @brief Non-owning read-only view over a typed tensor buffer. */
typedef struct MerMicroConstTensorView {
    /** @brief Address of the first scalar element. */
    const void* data;
    /** @brief Number of scalar elements referenced by the view. */
    size_t element_count;
    /** @brief Extent of each dimension, up to `rank` entries. */
    uint32_t dimensions[MER_MICRO_MAX_RANK];
    /** @brief Number of active dimensions in `dimensions`. */
    uint32_t rank;
    /** @brief Scalar type stored at `data`. */
    MerMicroDataType data_type;
} MerMicroConstTensorView;

/** @brief Non-owning mutable view over a typed tensor buffer. */
typedef struct MerMicroTensorView {
    /** @brief Address of the first scalar element. */
    void* data;
    /** @brief Number of scalar elements referenced by the view. */
    size_t element_count;
    /** @brief Extent of each dimension, up to `rank` entries. */
    uint32_t dimensions[MER_MICRO_MAX_RANK];
    /** @brief Number of active dimensions in `dimensions`. */
    uint32_t rank;
    /** @brief Scalar type stored at `data`. */
    MerMicroDataType data_type;
} MerMicroTensorView;

/** @brief Return the size in bytes of one scalar of `data_type`. */
size_t mer_micro_data_type_size(MerMicroDataType data_type);

/**
 * @brief Construct a read-only tensor view from a buffer and shape.
 * @param data Caller-owned tensor buffer.
 * @param data_type Scalar type stored in `data`.
 * @param dimensions Tensor extents.
 * @param rank Number of dimensions, from one through `MER_MICRO_MAX_RANK`.
 * @param output View to initialize.
 * @return A status describing validation success or failure.
 */
MerMicroStatus mer_micro_make_const_tensor_view(
    const void* data,
    MerMicroDataType data_type,
    const uint32_t* dimensions,
    uint32_t rank,
    MerMicroConstTensorView* output
);

/**
 * @brief Construct a mutable tensor view from a buffer and shape.
 * @param data Caller-owned writable tensor buffer.
 * @param data_type Scalar type stored in `data`.
 * @param dimensions Tensor extents.
 * @param rank Number of dimensions, from one through `MER_MICRO_MAX_RANK`.
 * @param output View to initialize.
 * @return A status describing validation success or failure.
 */
MerMicroStatus mer_micro_make_tensor_view(
    void* data,
    MerMicroDataType data_type,
    const uint32_t* dimensions,
    uint32_t rank,
    MerMicroTensorView* output
);

/** @brief Validate a read-only tensor view's type, shape, and element count. */
MerMicroStatus mer_micro_validate_const_tensor_view(
    const MerMicroConstTensorView* view
);

/** @brief Validate a mutable tensor view's type, shape, and element count. */
MerMicroStatus mer_micro_validate_tensor_view(
    const MerMicroTensorView* view
);

/** @brief Convert a mutable tensor view to a read-only view without copying data. */
MerMicroConstTensorView mer_micro_as_const_tensor_view(
    const MerMicroTensorView* view
);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // MINI_ORT_MICRO_TENSOR_VIEW_H_
