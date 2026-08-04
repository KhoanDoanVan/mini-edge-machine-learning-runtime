#pragma once

#include <cstddef>
#include <span>
#include <cstdint>
#include <vector>


namespace mini_ort {

    using Shape = std::vector<std::int64_t>;

    class Tensor final {
        public:

        Tensor(Shape shape, std::vector<float> data);

        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] const Shape& shape() const noexcept;
        [[nodiscard]] std::span<const float> data() const noexcept;
        [[nodiscard]] std::span<float> mutable_data() noexcept;

        private:

        Shape shape_;
        std::vector<float> data_;
    };

}