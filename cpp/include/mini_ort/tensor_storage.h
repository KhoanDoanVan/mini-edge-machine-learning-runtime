#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mini_ort {

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

    void ResetTensorMemoryStats() noexcept;
    [[nodiscard]] TensorMemoryStats GetTensorMemoryStats() noexcept;

    class TensorStorage final {

        public:
            explicit TensorStorage(std::size_t element_count);
            explicit TensorStorage(std::vector<float> data);

            TensorStorage(const TensorStorage& other);
            TensorStorage& operator=(const TensorStorage& other);
            TensorStorage(TensorStorage&& other) noexcept;
            TensorStorage& operator=(TensorStorage&& other) noexcept;
            ~TensorStorage();

            [[nodiscard]] std::size_t size() const noexcept;
            [[nodiscard]] bool empty() const noexcept;
            [[nodiscard]] std::span<const float> data() const noexcept;
            [[nodiscard]] std::span<float> mutable_data() noexcept;

        private:
            std::vector<float> data_;
            std::uint64_t tracked_bytes_ = 0;
    }

}