#include "mini_ort/tensor_view.h"

#include <limits>
#include <stdexcept>

namespace mini_ort {

    namespace {

        std::size_t ElementCount(const ShapeView shape) {

            std::size_t count = 1;

            for (const auto dimension : shape) {
                if (dimension < 0 ) {
                    throw std::invalid_argument("tensor dimensions must be non-negative");
                }

                const auto size = static_cast<std::size_t>(dimension);
                
                if (size == 0) {
                    count = 0;
                    continue;
                }

                if (count > std::numeric_limits<std::size_t>::max() / size) {
                    throw std::overflow_error("tensor element count overflows size_t");
                }

                count *= size;
            }

            return count;
        }

        void Validate(
            const ShapeView shape,
            const std::size_t data_size
        ) {
            if (ElementCount(shape) != data_size) {
                throw std::invalid_argument("tensor view data size does not match its shape");
            }
        }

    }


    ConstTensorView::ConstTensorView(
        const ShapeView shape,
        const std::span<const float> data
    ) : shape_(shape), data_(data) {
        Validate(
            shape_,
            data_.size()
        );
    }

    ShapeView ConstTensorView::shape() const noexcept { return shape_; }

    std::size_t ConstTensorView::size() const noexcept { return data_.size(); }

    bool ConstTensorView::empty() const noexcept { return data_.empty(); }

    std::span<const float> ConstTensorView::data() const noexcept { return data_; }

    TensorView::TensorView(
        const ShapeView shape,
        const std::span<float> data
    ) : shape_(shape), data_(data) {
        Validate(
            shape_,
            data_.size()
        );
    }

    ShapeView TensorView::shape() const noexcept { return shape_; }

    std::size_t TensorView::size() const noexcept { return data_.size(); }

    bool TensorView::empty() const noexcept { return data_.empty(); }

    std::span<const float> TensorView::data() const noexcept { return data_; }

    std::span<float> TensorView::mutable_data() const noexcept { return data_; }

    ConstTensorView TensorView::as_const() const {
        return ConstTensorView(
            shape_,
            data_
        );
    }

}