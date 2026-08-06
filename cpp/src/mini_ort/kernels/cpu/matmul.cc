#include "mini_ort/kernels/cpu.h"

#include <stdexcept>
#include <utility>
#include <vector>


namespace mini_ort::cpu {

    Tensor MatMul(
        const Tensor& lhs,
        const Tensor& rhs
    ) {

        if (lhs.shape().size() != 2 || rhs.shape().size() != 2) {
            throw std::invalid_argument("MatMul expects rank-2 tensors");
        }

        const auto rows = static_cast<std::size_t>(lhs.shape()[0]);
        const auto inner = static_cast<std::size_t>(lhs.shape()[1]);
        const auto rhs_inner = static_cast<std::size_t>(rhs.shape()[0]);
        const auto columns = static_cast<std::size_t>(rhs.shape()[1]);

        if (inner != rhs_inner) {
            throw std::invalid_argument("MatMul inner dimensions must match");
        }

        // output owns a heap allocation containing all MatMul results.
        std::vector<float> output(
            rows * columns,
            0.0F
        );
        const auto lhs_data = lhs.data();
        const auto rhs_data = rhs.data();

        // i-k-j keeps rhs and output access contiguous in the innermost loop.
        for (std::size_t row = 0; row < rows; ++row) {
            const auto lhs_row = row * inner;
            const auto output_row = row * columns;

            for (std::size_t reduction = 0; reduction < inner; ++reduction) {
                const float lhs_value = lhs_data[lhs_row + reduction];
                const auto rhs_row = reduction * columns;

                for (std::size_t column = 0; column < columns; ++column) {
                    output[output_row + column] += lhs_value * rhs_data[rhs_row + column];
                }
            }
        }

        return Tensor(
            {
                static_cast<std::int64_t>(rows),
                static_cast<std::int64_t>(columns)
            },
            // ownership of output’s internal memory instead of copying it
            std::move(output)
        );

    }

}