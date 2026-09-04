#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "mini_ort/allocator.h"

namespace mini_ort {

    /**
     * @brief Process-wide statistics for TensorStorage payload management.
     *
     * These counters describe float payload buffers owned by TensorStorage.
     * Shape metadata and allocations performed by unrelated standard-library
     * containers are intentionally excluded.
     */
    struct TensorMemoryStats final {
        std::uint64_t buffer_allocations;
        std::uint64_t allocated_bytes;
        std::uint64_t tensor_copies;
        std::uint64_t copied_bytes;
        std::uint64_t live_bytes;
        std::uint64_t peak_live_bytes;
    };

    // These counters describe float payload buffers owned by TensorStorage. Shape
    // metadata and allocations made elsewhere by the standard library are excluded.

    /**
     * @brief Reset all TensorStorage memory counters to zero.
     *
     * This function is intended for test and benchmark setup. It does not
     * release live allocations or alter existing TensorStorage objects.
     */
    void ResetTensorMemoryStats() noexcept;

    /**
     * @brief Take a snapshot of TensorStorage memory counters.
     * @return Current allocation, copy, live-byte, and peak-byte totals.
     */
    [[nodiscard]] TensorMemoryStats GetTensorMemoryStats() noexcept;

    class TensorStorage final {

        public:
            /**
             * @brief Allocate storage for a fixed number of float elements.
             *
             * The payload is owned by this object and initialized according to
             * the allocator implementation's allocation semantics.
             *
             * @param element_count Number of float elements to reserve.
             * @param allocator Allocator retained until destruction.
             * @throws std::bad_alloc when the allocation cannot be satisfied.
             */
            explicit TensorStorage(
                std::size_t element_count,
                Allocator& allocator = DefaultAllocator()
            );
            /**
             * @brief Construct storage by copying a floating-point vector.
             * @param data Source values; the storage does not alias this vector.
             * @param allocator Allocator retained until destruction.
             */
            explicit TensorStorage(
                std::vector<float> data,
                Allocator& allocator = DefaultAllocator()
            );

            /** @brief Deep-copy another storage payload. */
            TensorStorage(const TensorStorage& other);
            /** @brief Replace this payload with a deep copy of `other`. */
            TensorStorage& operator=(const TensorStorage& other);
            /** @brief Transfer payload ownership without copying elements. */
            TensorStorage(TensorStorage&& other) noexcept;
            /** @brief Release this payload and take ownership from `other`. */
            TensorStorage& operator=(TensorStorage&& other) noexcept;
            /** @brief Release the payload through its originating allocator. */
            ~TensorStorage();

            /** @brief Return the number of float elements in the payload. */
            [[nodiscard]] std::size_t size() const noexcept;
            /** @brief Return whether the payload contains zero elements. */
            [[nodiscard]] bool empty() const noexcept;
            /** @brief Return a read-only span over the owned payload. */
            [[nodiscard]] std::span<const float> data() const noexcept;
            /** @brief Return a writable span over the owned payload. */
            [[nodiscard]] std::span<float> mutable_data() noexcept;

        private:

            void Release() noexcept;

            Allocator* allocator_;

            // std::vector<float> data_;
            float* data_;

            std::size_t element_count_;
            std::uint64_t tracked_bytes_ = 0;
    };

}
