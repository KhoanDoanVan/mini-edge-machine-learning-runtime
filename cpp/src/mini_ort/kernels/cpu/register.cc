#include "mini_ort/kernels/cpu.h"

#include <memory>
#include <stdexcept>
#include <utility>


namespace mini_ort::cpu {

    namespace {

        void RequireInputCount(
            const std::span<const ConstTensorView> inputs,
            const std::size_t expected,
            const char* op_type
        ) {
            if (inputs.size() != expected) {
                throw std::invalid_argument(std::string(op_type) + " received an invalid input count");
            }
        }

        class MatMulKernel final : public Kernel {

            public:

                void Compute(
                    const std::span<const ConstTensorView> inputs,
                    const TensorView output
                ) const override {
                    RequireInputCount(
                        inputs,
                        2,
                        "MatMul"
                    );
                    MatMul(
                        inputs[0],
                        inputs[1],
                        output
                    );
                }

        };

        class AddKernel final : public Kernel {

            public:

                void Compute(
                    const std::span<const ConstTensorView> inputs,
                    const TensorView output
                ) const override {
                    RequireInputCount(
                        inputs,
                        2,
                        "Add"
                    );
                    Add(
                        inputs[0],
                        inputs[1],
                        output
                    );
                }

        };

        class ReluKernel final : public Kernel {

            public:
            
                void Compute(
                    const std::span<const ConstTensorView> inputs,
                    const TensorView output
                ) const override {
                    RequireInputCount(
                        inputs,
                        1,
                        "Relu"
                    );
                    Relu(
                        inputs[0],
                        output
                    );
                }

        };

    }

    void RegisterKernels(
        KernelRegistry& registry
    ) {
        registry.Registry(
            "MatMul",
            std::make_unique<MatMulKernel>()
        );
        registry.Registry(
            "Add",
            std::make_unique<AddKernel>()
        );
        registry.Registry(
            "Relu",
            std::make_unique<ReluKernel>()
        );
    }

}