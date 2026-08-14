#pragma once

#include <cstddef>
#include <span>
#include <cstdint>
#include <vector>


namespace mini_ort {

    using Shape = std::vector<std::int64_t>;

    struct TensorMemoryStats final {
        std::uint64_t buffer_allocations;
        std::uint64_t allocated_bytes;
        std::uint64_t tensor_copies;
        std::uint64_t copied_bytes;
        std::uint64_t live_bytes;
        std::uint64_t peak_live_bytes;
    }

    // These counters describe float payload buffers owned by Tensor. Shape metadata
    // and allocations made by the C++ standard library are intentionally excluded.
    void ResetTensorMemoryStats() noexcept;

    [[nodiscard]] TensorMemoryStats GetTensorMemoryStats() noexcept;

    class Tensor final {
        public:

            Tensor(Shape shape, std::vector<float> data);
            Tensor(const Tensor& other);
            Tensor& operator = (const Tensor& other);
            Tensor(Tensor&& other) noexcept;
            Tensor& operator = (Tensor&& other) noexcept;
            ~Tensor();

            [[nodiscard]] std::size_t size() const noexcept;
            [[nodiscard]] bool empty() const noexcept;
            [[nodiscard]] const Shape& shape() const noexcept;
            [[nodiscard]] std::span<const float> data() const noexcept;
            [[nodiscard]] std::span<float> mutable_data() noexcept;

        private:

            Shape shape_;
            std::vector<float> data_;
            std::uint64_t tracked_bytes_ = 0;
    };

}