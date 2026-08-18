#pragma once

#include "mini_ort/kernel.h"
#include "mini_ort/tensor.h"
#include "mini_ort/tensor_view.h"


namespace mini_ort::cpu {

    Tensor MatMul(ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    Tensor Add(ConstTensorView lhs, ConstTensorView rhs, TensorView output);
    Tensor Relu(ConstTensorView input, TensorView output);

    void RegisterKernels(KernelRegistry& registry);
    
}