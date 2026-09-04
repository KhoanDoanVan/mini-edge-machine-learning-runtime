#pragma once

#include "mini_ort/kernel.h"
#include "mini_ort/tensor.h"
#include "mini_ort/tensor_view.h"


namespace mini_ort::cpu {

    enum class MatMulImplementation : std::uint8_t {
        kScalar,
        kTiled
    };

    /** @brief Select scalar or tiled multiplication for the given dimensions. */
    [[nodiscard]] MatMulImplementation SelectMatMulImplementation (
        std::size_t rows,
        std::size_t inner,
        std::size_t columns
    ) noexcept;
    /** @brief Return a stable diagnostic name for a multiplication strategy. */
    [[nodiscard]] const char* MatMulImplementationName (MatMulImplementation implementation) noexcept;

    /** @brief Multiply two rank-2 float tensors into a preallocated output. */
    void MatMul(ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    /** @brief Execute the portable scalar matrix-multiplication kernel. */
    void MatMulScaler (ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    /** @brief Execute the cache-blocked matrix-multiplication kernel. */
    void MatMulTiled (ConstTensorView lhs, ConstTensorView rhs, TensorView output);

    /** @brief Add two broadcast-compatible tensors into `output`. */
    void Add(ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    /** @brief Apply element-wise ReLU to `input` and write `output`. */
    void Relu(ConstTensorView input, TensorView output);

    /** @brief Register all CPU reference kernels in `registry`. */
    void RegisterKernels(KernelRegistry& registry);
    
}
