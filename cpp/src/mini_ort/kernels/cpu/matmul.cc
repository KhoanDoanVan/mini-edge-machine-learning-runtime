#include "mini_ort/kernels/cpu.h"

#include <stdexcept>
#include <utility>
#include <vector>


namespace mini_ort::cpu {

    namespace {
        constexpr std::size_t kRowTile = 16;
        constexpr std::size_t kReductionTile = 64;
        constexpr std::size_t kColumnTile = 64;
        constexpr std::size_t kMinimumTiledRows = 8;
        constexpr std::size_t kMinimumTiledInner = 256;
        constexpr std::size_t kMinimumTiledColumns = 64;
        constexpr std::size_t kMinimumTiledOperations = 512 * 1024;


        struct MatMulShape final {
            std::size_t rows;
            std::size_t inner;
            std::size_t columns;
        };

        MatMulShape Validate(
            const ConstTensorView lhs,
            const ConstTensorView rhs,
            const TensorView output
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

            if (output.shape().size() != 2 || output.shape()[0] != static_cast<std::int64_t>(rows) || output.shape()[1] != static_cast<std::int64_t>(columns)) {
                throw std::invalid_argument("MatMul output shape is invalid");
            }

            if (output.data().data() == lhs.data().data() || output.data().data() == rhs.data().data()) {
                throw std::invalid_argument("MatMul does not support aliased output");
            }

            return MatMulShape{
                rows, 
                inner, 
                columns
            };
        }

        bool HasAtLeasetOperations(
            const std::size_t rows, 
            const std::size_t inner,
            const std::size_t columns,
            const std::size_t minimum
        ) noexcept {
            if (rows == 0 || inner == 0 || columns == 0) {
                return false;
            }
            if (rows > std::numeric_limits<std::size_t>::max() / inner) {
                return true;
            }
            const auto row_inner = rows * inner;
            if (row_inner > std::numeric_limits<std::size_t>::max() / columns) {
                return true;
            }

            return row_inner * columns >= minimum;
        }

        void ComputeScalar(
            const ConstTensorView lhs, 
            const ConstTensorView rhs,
            const TensorView output, 
            const MatMulShape shape
        ) {
            const auto lhs_data = lhs.data();
            const auto rhs_data = rhs.data();
            const auto output_data = output.mutable_data();

            std::fill(output_data.begin(), output_data.end(), 0.0F);

            // i-k-j keeps RHS and output access contiguous in the innermost loop.
            for (std::size_t row = 0; row < shape.rows; ++row) {

                const auto lhs_row = row * shape.inner;
                const auto output_row = row * shape.columns;

                for (std::size_t reduction = 0; reduction < shape.inner; ++reduction) {

                    const float lhs_value = lhs_data[lhs_row + reduction];
                    const auto rhs_row = reduction * shape.columns;

                    for (std::size_t column = 0; column < shape.columns; ++column) {

                        output_data[output_row + column] += lhs_value * rhs_data[rhs_row + column];

                    }

                }

            }
        }

        void ComputeTiled(
            const ConstTensorView lhs, 
            const ConstTensorView rhs,
            const TensorView output, 
            const MatMulShape shape
        ) {
            const auto lhs_data = lhs.data();
            const auto rhs_data = rhs.data();
            const auto output_data = output.mutable_data();

            std::fill(output_data.begin(), output_data.end(), 0.0F);

            // Work on cache-sized submatrices. Each RHS tile is reused by multiple rows
            // before the traversal advances, while the innermost column access remains
            // contiguous for both RHS and output.

            for (std::size_t row_begin = 0; row_begin < shape.rows; row_begin += kRowTile) {

                const auto row_end = std::min(row_begin + kRowTile, shape.rows);

                for (std::size_t reduction_begin = 0; reduction_begin < shape.inner; reduction_begin += kReductionTile) {
                    const auto reduction_end = std::min(reduction_begin + kReductionTile, shape.inner);


                    for (std::size_t column_begin = 0; column_begin < shape.columns; column_begin += kColumnTile) {
                        const auto column_end = std::min(column_begin + kColumnTile, shape.columns);

                        for (std::size_t reduction = reduction_begin; reduction < reduction_end; ++reduction) {
                            const auto rhs_row = reduction * shape.columns;

                            for (std::size_t row = row_begin; row < row_end; ++row) {
                                const auto lhs_row = row * shape.inner;
                                const auto output_row = row * shape.columns;
                                const float lhs_value = lhs_data[lhs_row + reduction];

                                for (std::size_t column = column_begin; column < column_end; ++column) {
                                    output_data[output_row + column] += lhs_value * rhs_data[rhs_row + column];
                                }
                            }
                        }
                    }

                }

            }
        }

    } // namespace

    MatMulImplementation SelectMatMulImplementation(
        const std::size_t rows, 
        const std::size_t inner,
        const std::size_t columns
    ) noexcept {
        // A row tile is what enables RHS reuse. Batch-1 and very narrow problems
        // stay scalar because blocking overhead cannot pay for itself there.
        if (rows < kMinimumTiledRows || inner < kMinimumTiledInner || columns < kMinimumTiledColumns || !HasAtLeasetOperations(rows, inner, columns, kMinimumTiledOperations)) {
            return MatMulImplementation::kScalar;
        }

        return MatMulImplementation::kTiled;
    }

    const char* MatMulImplementationName(const MatMulImplementation implementation) noexcept {
        switch (implementation) {
            case MatMulImplementation::kScalar:
            return "scalar";
            case MatMulImplementation::kTiled:
            return "tiled";
        }
        return "unknown";
    }

    void MatMul(
        const ConstTensorView lhs,
        const ConstTensorView rhs,
        const TensorView output
    ) {

        const auto shape = Validate(lhs, rhs, output);

        if (SelectMatMulImplementation(shape.rows, shape.inner, shape.columns) == MatMulImplementation::kTiled) {
            ComputeTiled(lhs, rhs, output, shape);
        } else {
            ComputeScalar(lhs, rhs, output, shape);
        }

    }

    void MatMulScalar(
        const ConstTensorView lhs, 
        const ConstTensorView rhs,
        const TensorView output
    ) {
        const auto shape = Validate(lhs, rhs, output);
        ComputeScalar(lhs, rhs, output, shape);
    }

    void MatMulTiled(
        const ConstTensorView lhs, 
        const ConstTensorView rhs,
        const TensorView output
    ) {
        const auto shape = Validate(lhs, rhs, output);
        ComputeTiled(lhs, rhs, output, shape);
    }

}