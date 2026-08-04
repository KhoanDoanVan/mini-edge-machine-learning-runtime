#include "mini_ort/kernel.h"

#include <stdexcept>
#include <utility>

namespace mini_ort {

    void KernelRegistry::Registry(
        std::string op_type,
        std::unique_ptr<Kernel> kernel
    ) {
        if (op_type.empty()) {
            throw std::invalid_argument("kernel op type must not be empty");
        }

        if (kernel == nullptr) {
            throw std::invalid_argument("kernel must not be null");
        }

        const auto [position, inserted] = kernels_.emplace(std::move(op_type), std::move(kernel));

        if (!inserted) {
            throw std::invalid_argument("kernel is already registered: " + position->first);
        }
    }

    const Kernel& KernelRegistry::Resolve(const std::string& op_type) const {
        const auto position = kernels_.find(op_type);

        if (position == kernels_.end()) {
            throw std::out_of_range("no kernel registered for: " + op_type);
        }

        return *position->second;
    }
    
}