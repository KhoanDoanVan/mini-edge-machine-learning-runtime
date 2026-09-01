#pragma once

#include "mini_ort/kernel.h"
#include "mini_ort/tensor.h"
#include "mini_ort/tensor_view.h"


namespace mini_ort::cpu {

    enum class MatMulImplementation : std::uint8_t {
        kScalar,
        kTiled
    };

    [[nodiscard]] MatMulImplementation SelectMatMulImplementation (
        std::size_t rows,
        std::size_t inner,
        std::size_t columns
    ) noexcept;
    [[nodiscard]] const char* MatMulImplementationName (MatMulImplementation implementation) noexcept;

    void MatMul(ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    void MatMulScaler (ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    void MatMulTiled (ConstTensorView lhs, ConstTensorView rhs, TensorView output);

    void Add(ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    void Relu(ConstTensorView input, TensorView output);

    void RegisterKernels(KernelRegistry& registry);
    
}