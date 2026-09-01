#include "mini_ort/kernels/cpu.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_ort::cpu {
    
    void Add(
        const ConstTensorView lhs, 
        const ConstTensorView rhs,
        const TensorView output
    ) {

        if (!std::ranges::equal(lhs.shape(), output.shape())) {
            throw std::invalid_argument("Add output shape must match the left input");
        }

        // std::vector<float> output(lhs.size());
        const auto lhs_data = lhs.data();
        const auto rhs_data = rhs.data();
        const auto output_data = output.mutable_data();

        if (std::ranges::equal(lhs.shape(), rhs.shape())) {
            for (std::size_t index = 0; index < lhs.size(); ++index) {
                output_data[index] = lhs_data[index] + rhs_data[index];
            }
        } else if (rhs.shape().empty()) {
            const float scalar = rhs_data[0];
            for (std::size_t index = 0; index < lhs.size(); ++index) {
                output_data[index] = lhs_data[index] + scalar;
            }
        } else if (rhs.shape().size() == 1 && !lhs.shape().empty() && rhs.shape()[0] == lhs.shape().back()) {
            const auto width = static_cast<std::size_t>(rhs.shape()[0]);
            for (std::size_t index = 0; index < lhs.size(); ++index) {
                output_data[index] = lhs_data[index] + rhs_data[index % width];
            }
        } else {
            throw std::invalid_argument("Add expects equal shapes, a scaler, or a last-axis bias");
        }

        // return Tensor(
        //     Shape(
        //         lhs.shape().begin(),
        //         lhs.shape().end()
        //     ),
        //     std::move(output)
        // );

    }

}