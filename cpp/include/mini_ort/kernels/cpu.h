#pragma once

#include "mini_ort/kernel.h"
#include "mini_ort/tensor.h"


namespace mini_ort::cpu {

    Tensor MatMul(const Tensor& lhs, const Tensor& rhs);
    Tensor Add(const Tensor& lhs, const Tensor& rhs);
    Tensor Relu(const Tensor& input);

    void RegisterKernels(KernelRegistry& registry);
    
}