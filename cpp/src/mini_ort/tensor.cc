#include "mini_ort/tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>


namespace mini_ort {

    namespace {
        std::size_t ElementCount(const Shape& shape) {

            for (const auto dimension : shape) {
                if (dimension < 0) {
                    throw std::invalid_argument("tensor dimensions must be non-negative");
                }
            }

            std::size_t count = 1;
            
            for (const auto dimension : shape) {
                const auto size = static_cast<std::size_t>(dimension);

                if (size == 0) {
                    count = 0;
                    continue;
                }

                if (count > std::numeric_limits<std::size_t>::max() / size) {
                    throw std::overflow_error("tensor element count overflow size_t");
                }

                count *= size;
            }

            return count;
        }
    }

    Tensor::Tensor(Shape shape, std::vector<float> data) : shape_(std::move(shape)), data_(std::move(data)) {
        const auto expected_size = ElementCount(shape_);

        if (data_.size() != expected_size) {
            throw std::invalid_argument("tensor data size does not match its shape.");
        }
    }

    std::size_t Tensor::size() const noexcept { return data_.size(); }

    bool Tensor::empty() const noexcept { return data_.empty(); }

    const Shape& Tensor::shape() const noexcept { return shape_; }

    std::span<const float> Tensor::data() const noexcept { return data_; }

    std::span<float> Tensor::mutable_data() noexcept { return data_; }

}