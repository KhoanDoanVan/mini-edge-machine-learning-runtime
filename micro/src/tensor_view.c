#include "mini_ort_micro/tensor_view.h"

#include <stdint.h>
#include <string.h>


_Static_assert(
    sizeof(float) == 4,
    "FLOAT32 tensors require 32-bit float"
);

static MerMicroStatus ValidateByteSize(
    size_t element_count,
    MerMicroDataType data_type
) {
  /* Reject unsupported types and prevent element-to-byte multiplication overflow. */
  const size_t element_size = mer_micro_data_type_size(data_type);
  if (element_size == 0) {
    return MER_MICRO_STATUS_UNSUPPORTED_TYPE;
  }
  return element_count <= SIZE_MAX / element_size ? MER_MICRO_STATUS_OK : MER_MICRO_STATUS_OUT_OF_RANGE;
}

static MerMicroStatus CalculateElementCount(
    const uint32_t* dimensions,
    uint32_t rank,
    size_t* element_count
) {
    /* Compute the product while checking rank, zero extents, and overflow. */
    if (dimensions == NULL || element_count == NULL || rank == 0 || rank > MER_MICRO_MAX_RANK) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    size_t count = 1;
    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
        const size_t extent = dimensions[dimension];
        if (extent == 0 || count > SIZE_MAX / extent) {
            return MER_MICRO_STATUS_OUT_OF_RANGE;
        }
        count *= extent;
    }
    *element_count = count;
  
    return MER_MICRO_STATUS_OK;
}

size_t mer_micro_data_type_size(
    MerMicroDataType data_type
) {
    /* Return zero for unknown types so callers can report a validation error. */
    switch (data_type) {
        case MER_MICRO_DATA_TYPE_FLOAT32:
            return sizeof(float);
        case MER_MICRO_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case MER_MICRO_DATA_TYPE_INT32:
            return sizeof(int32_t);
    }

    return 0;
}


MerMicroStatus mer_micro_make_const_tensor_view(
    const void* data,
    MerMicroDataType data_type,
    const uint32_t* dimensions,
    uint32_t rank,
    MerMicroConstTensorView* output
) {
    /* Build a validated read-only view without taking ownership of `data`. */
    if (data == NULL || output == NULL || mer_micro_data_type_size(data_type) == 0) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    size_t element_count = 0;

    const MerMicroStatus status = CalculateElementCount(dimensions, rank, &element_count);
  
    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }

    const MerMicroStatus byte_size_status = ValidateByteSize(element_count, data_type);
    if (byte_size_status != MER_MICRO_STATUS_OK) {
        return byte_size_status;
    }

    memset(output, 0, sizeof(*output));
    output->data = data;
    output->element_count = element_count;
    memcpy(output->dimensions, dimensions, rank * sizeof(dimensions[0]));
    output->rank = rank;
    output->data_type = data_type;

    return MER_MICRO_STATUS_OK;
}

MerMicroStatus mer_micro_make_tensor_view(
    void* data,
    MerMicroDataType data_type,
    const uint32_t* dimensions,
    uint32_t rank,
    MerMicroTensorView* output
) {
    /* Build a validated mutable view without taking ownership of `data`. */
    if (data == NULL || output == NULL || mer_micro_data_type_size(data_type) == 0) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    size_t element_count = 0;
  
    const MerMicroStatus status = CalculateElementCount(dimensions, rank, &element_count);
    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }
    const MerMicroStatus byte_size_status = ValidateByteSize(element_count, data_type);
    if (byte_size_status != MER_MICRO_STATUS_OK) {
        return byte_size_status;
    }

    memset(output, 0, sizeof(*output));
    output->data = data;
    output->element_count = element_count;
    memcpy(output->dimensions, dimensions, rank * sizeof(dimensions[0]));
    output->rank = rank;
    output->data_type = data_type;
    return MER_MICRO_STATUS_OK;
}

MerMicroStatus mer_micro_validate_const_tensor_view(
    const MerMicroConstTensorView* view
) {
    /* Recompute the shape product and compare it with the stored element count. */
    if (view == NULL || view->data == NULL || mer_micro_data_type_size(view->data_type) == 0) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }

    size_t expected_count = 0;
    const MerMicroStatus status = CalculateElementCount(view->dimensions, view->rank, &expected_count);
    if (status != MER_MICRO_STATUS_OK) {
        return status;
    }
    const MerMicroStatus byte_size_status = ValidateByteSize(expected_count, view->data_type);
    if (byte_size_status != MER_MICRO_STATUS_OK) {
        return byte_size_status;
    }
    return view->element_count == expected_count ? MER_MICRO_STATUS_OK : MER_MICRO_STATUS_SHAPE_MISMATCH;
}


MerMicroStatus mer_micro_validate_tensor_view(
    const MerMicroTensorView* view
) {
    /* Validate mutable views through the same rules as read-only views. */
    if (view == NULL) {
        return MER_MICRO_STATUS_INVALID_ARGUMENT;
    }
    const MerMicroConstTensorView const_view = mer_micro_as_const_tensor_view(view);
    return mer_micro_validate_const_tensor_view(&const_view);
}


MerMicroConstTensorView mer_micro_as_const_tensor_view(
    const MerMicroTensorView* view
) {
    /* Copy view metadata while preserving the original buffer address. */
    MerMicroConstTensorView result;
    memset(&result, 0, sizeof(result));
    if (view == NULL) {
        return result;
    }

    result.data = view->data;
    result.element_count = view->element_count;
    memcpy(result.dimensions, view->dimensions, sizeof(result.dimensions));
    result.rank = view->rank;
    result.data_type = view->data_type;
    return result;
}
