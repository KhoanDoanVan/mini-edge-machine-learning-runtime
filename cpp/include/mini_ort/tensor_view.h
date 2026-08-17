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
            ConstTensorView(
                ShapeView shape,
                std::span<const float> data
            );

            [[nodiscard]] ShapeView shape() const noexcept;
            [[nodiscard]] std::size_t size() const noexcept;
            [[nodiscard]] bool empty() const noexcept;
            [[nodiscard]] std::span<const float> data() const noexcept;

        private:
            ShapeView shape_;
            std::span<const float> data_;

    };


    class TensorView final {

        public:
            TensorView(
                ShapeView shape,
                std::span<float> data
            );

            [[nodiscard]] ShapeView shape() const noexcept;
            [[nodiscard]] std::size_t size() const noexcept;
            [[nodiscard]] bool empty() const noexcept;
            [[nodiscard]] std::span<const float> data() const noexcept;
            [[nodiscard]] std::span<float> mutable_data() const noexcept;
            [[nodiscard]] ConstTensorView as_const() const;

        private:
            ShapeView shape_;
            std::span<float> data_;

    };

}