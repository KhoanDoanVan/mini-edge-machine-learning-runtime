#include "mini_ort/kernels/cpu.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_ort::cpu {
    
    Tensor Add(
        const Tensor& lhs, 
        const Tensor& rhs
    ) {
        std::vector<float> output(lhs.size());
        const auto lhs_data = lhs.data();
        const auto rhs_data = rhs.data();

        if (lhs.shape() == rhs.shape()) {
            for (std::size_t index = 0; index < lhs.size(); ++index) {
                output[index] = lhs_data[index] + rhs_data[index];
            }
        } else if (rhs.shape().empty()) {
            const float scalar = rhs_data[0];
            for (std::size_t index = 0; index < lhs.size(); ++index) {
                output[index] = lhs_data[index] + scalar;
            }
        } else if (rhs.shape().size() == 1 && !lhs.shape().empty() && rhs.shape()[0] == lhs.shape().back()) {
            const auto width = static_cast<std::size_t>(rhs.shape()[0]);
            for (std::size_t index = 0; index < lhs.size(); ++index) {
                output[index] = lhs_data[index] + rhs_data[index % width];
            }
        } else {
            throw std::invalid_argument("Add expects equal shapes, a scaler, or a last-axis bias");
        }

        return Tensor(
            lhs.shape(),
            std::move(output)
        );

    }

}