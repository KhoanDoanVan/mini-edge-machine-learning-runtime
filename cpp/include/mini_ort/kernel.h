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
        /**
         * @brief Execute the kernel for one set of input tensors.
         * @param inputs Read-only tensors in operator-defined order.
         * @param output Preallocated destination tensor view.
         * @throws std::invalid_argument when shapes or arity are unsupported.
         */
        virtual void Compute(
            std::span<const ConstTensorView> inputs,
            TensorView output
        ) const = 0;

    };

    class KernelRegistry final {

        public:

        /** @brief Register ownership of a kernel under an operator name. */
        void Registry(std::string op_type, std::unique_ptr<Kernel> kernel);
        /**
         * @brief Resolve a registered operator implementation.
         * @param op_type Operator name encoded by the model.
         * @return Registered kernel reference.
         * @throws std::out_of_range when no implementation is registered.
         */
        [[nodiscard]] const Kernel& Resolve(const std::string& op_type) const;

        private:

        std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels_;

    };

}
