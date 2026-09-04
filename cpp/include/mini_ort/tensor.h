#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "mini_ort/tensor_storage.h"
#include "mini_ort/tensor_view.h"


// old version:
// namespace mini_ort {

//     using Shape = std::vector<std::int64_t>;

//     struct TensorMemoryStats final {
//         std::uint64_t buffer_allocations;
//         std::uint64_t allocated_bytes;
//         std::uint64_t tensor_copies;
//         std::uint64_t copied_bytes;
//         std::uint64_t live_bytes;
//         std::uint64_t peak_live_bytes;
//     }

//     // These counters describe float payload buffers owned by Tensor. Shape metadata
//     // and allocations made by the C++ standard library are intentionally excluded.
//     void ResetTensorMemoryStats() noexcept;

//     [[nodiscard]] TensorMemoryStats GetTensorMemoryStats() noexcept;

//     class Tensor final {
//         public:

//             Tensor(Shape shape, std::vector<float> data);
//             Tensor(const Tensor& other);
//             Tensor& operator = (const Tensor& other);
//             Tensor(Tensor&& other) noexcept;
//             Tensor& operator = (Tensor&& other) noexcept;
//             ~Tensor();

//             [[nodiscard]] std::size_t size() const noexcept;
//             [[nodiscard]] bool empty() const noexcept;
//             [[nodiscard]] const Shape& shape() const noexcept;
//             [[nodiscard]] std::span<const float> data() const noexcept;
//             [[nodiscard]] std::span<float> mutable_data() noexcept;

//         private:

//             Shape shape_;
//             std::vector<float> data_;
//             std::uint64_t tracked_bytes_ = 0;
//     };

// }

// new version
namespace mini_ort {

    class Tensor final {

        public:
            /**
             * @brief Construct a zero-initialized tensor for a shape.
             * @param shape Dimensions expressed as signed element counts.
             * @throws std::invalid_argument for invalid dimensions.
             * @throws std::bad_alloc when payload storage cannot be allocated.
             */
            explicit Tensor(Shape shape);

            /**
             * @brief Construct a tensor by copying supplied values.
             * @param shape Tensor dimensions; its element count must equal data size.
             * @param data Initial floating-point payload.
             * @throws std::invalid_argument when shape and data sizes disagree.
             */
            Tensor(Shape shape, std::vector<float> data);

            /** @brief Deep-copy shape metadata and payload. */
            Tensor(const Tensor& other) = default;
            /** @brief Replace this tensor with a deep copy of `other`. */
            Tensor& operator=(const Tensor& other) = default;
            /** @brief Transfer shape metadata and payload ownership. */
            Tensor(Tensor&& other) noexcept = default;
            /** @brief Release this tensor and take ownership from `other`. */
            Tensor& operator=(Tensor&& other) noexcept = default;
            /** @brief Release the tensor payload and metadata. */
            ~Tensor() = default;

            /** @brief Return the number of scalar elements. */
            [[nodiscard]] std::size_t size() const noexcept;
            /** @brief Return whether this tensor contains zero elements. */
            [[nodiscard]] bool empty() const noexcept;
            /** @brief Return the tensor dimensions. */
            [[nodiscard]] const Shape& shape() const noexcept;
            /** @brief Return a read-only span over the payload. */
            [[nodiscard]] std::span<const float> data() const noexcept;
            /** @brief Return a writable span over the payload. */
            [[nodiscard]] std::span<float> mutable_data() noexcept;
            /** @brief Create a non-owning read-only view without copying. */
            [[nodiscard]] ConstTensorView view() const;
            /** @brief Create a non-owning mutable view without copying. */
            [[nodiscard]] TensorView mutable_view();
            /** @brief Return the underlying read-only storage object. */
            [[nodiscard]] const TensorStorage& storage() const noexcept;
            /** @brief Return the underlying mutable storage object. */
            [[nodiscard]] TensorStorage& mutable_storage() noexcept;

        private:
            Shape shape_;
            TensorStorage storage_;

    };

}
