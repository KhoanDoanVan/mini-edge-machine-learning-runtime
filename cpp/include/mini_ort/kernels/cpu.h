#pragma once

#include "mini_ort/kernel.h"
#include "mini_ort/tensor.h"
#include "mini_ort/tensor_view.h"


namespace mini_ort::cpu {

    Tensor MatMul(ConstTensorView lhs, ConstTensorView rhs);
    Tensor Add(ConstTensorView lhs, ConstTensorView rhs);
    Tensor Relu(ConstTensorView input);

    void RegisterKernels(KernelRegistry& registry);
    
}