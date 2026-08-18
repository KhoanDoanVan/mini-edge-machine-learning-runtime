#pragma once

#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_ort/tensor.h"

namespace mini_ort {
    
    class Kernel {

        public:

        virtual ~Kernel() = default;
        virtual void Compute(
            std::span<const ConstTensorView> inputs,
            TensorView output
        ) const = 0;

    };

    class KernelRegistry final {

        public:

        void Registry(std::string op_type, std::unique_ptr<Kernel> kernel);
        [[nodiscard]] const Kernel& Resolve(const std::string& op_type) const;

        private:

        std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels_;

    };

}