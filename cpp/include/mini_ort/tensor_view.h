#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>



namespace mini_ort {

    using Shape = std::vector<std::int64_t>;
    using ShapeView = std::span<const std::int64_t>;

    class ConstTensorView final {

        public:
            /**
             * @brief Construct a non-owning read-only tensor view.
             * @param shape Dimensions; must outlive this view.
             * @param data Scalar payload; its element count must match @p shape.
             */
            ConstTensorView(
                ShapeView shape,
                std::span<const float> data
            );

            /** @brief Return the dimensions referenced by this view. */
            [[nodiscard]] ShapeView shape() const noexcept;
            /** @brief Return the number of scalar elements. */
            [[nodiscard]] std::size_t size() const noexcept;
            /** @brief Return whether the payload contains no elements. */
            [[nodiscard]] bool empty() const noexcept;
            /** @brief Return the immutable scalar payload. */
            [[nodiscard]] std::span<const float> data() const noexcept;

        private:
            ShapeView shape_;
            std::span<const float> data_;

    };


    class TensorView final {

        public:
            /**
             * @brief Construct a non-owning mutable tensor view.
             * @param shape Dimensions; must outlive this view.
             * @param data Writable payload whose size matches @p shape.
             */
            TensorView(
                ShapeView shape,
                std::span<float> data
            );

            /** @brief Return the dimensions referenced by this view. */
            [[nodiscard]] ShapeView shape() const noexcept;
            /** @brief Return the number of scalar elements. */
            [[nodiscard]] std::size_t size() const noexcept;
            /** @brief Return whether the payload contains no elements. */
            [[nodiscard]] bool empty() const noexcept;
            /** @brief Return a read-only view of the payload. */
            [[nodiscard]] std::span<const float> data() const noexcept;
            /** @brief Return the writable scalar payload. */
            [[nodiscard]] std::span<float> mutable_data() const noexcept;
            /** @brief Convert this view to a read-only view without copying. */
            [[nodiscard]] ConstTensorView as_const() const;

        private:
            ShapeView shape_;
            std::span<float> data_;

    };

}
