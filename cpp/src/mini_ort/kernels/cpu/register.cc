#include "mini_ort/kernels/cpu.h"

#include <memory>
#include <stdexcept>
#include <utility>


namespace mini_ort::cpu {

    namespace {

        void RequireInputs(
            std::span<const Tensor* const> inputs,
            const std::size_t expected,
            const char* op_type
        ) {
            if (inputs.size() != expected) {
                throw std::invalid_argument(std::string(op_type) + " received an invalid input count");
            }

            for (const Tensor* input : inputs) {
                if (input == nullptr) {
                    throw std::invalid_argument(std::string(op_type) + " received a null input");
                }
            }
        }

        class MatMulKernel final : public Kernel {

            public:

                std::vector<Tensor> Compute(
                    const std::span<const Tensor* const> inputs
                ) const override {
                    RequireInputs(
                        input,
                        2,
                        "MatMul"
                    );
                    std::vector<Tensor> outputs;
                    outputs.emplace_back(
                        MatMul(
                            *input[0],
                            *input[1]
                        )
                    );

                    return outputs;
                }

        };

        class AddKernel final : public Kernel {

            public:

                std::vector<Tensor> Compute(
                    const std::span<const Tensor* const> inputs
                ) const override {
                    RequireInputs(
                        inputs,
                        2,
                        "MatMul"
                    );
                    std::vector<Tensor> outputs;
                    outputs.emplace_back(
                        MatMul(
                            *input[0],
                            *input[1]
                        )
                    );
                    return outputs;
                }

        };

        class ReluKernel final : public Kernel {

            public:
            
                std::vector<Tensor> Compute(
                    const std::span<const Tensor* const> inputs
                ) const override {
                    RequireInputs(
                        inputs,
                        1,
                        "Relu"
                    );
                    std::vector<Tensor> outputs;
                    outputs.emplace_back(
                        Relu(
                            *inputs[0]
                        )
                    );

                    return outputs;
                }

        };

    }

    void RegisterKernels(
        KernelRegistry& registry
    ) {
        registry.Register(
            "MatMul",
            std::make_unique<MatMulKernel>()
        );
        registry.Register(
            "Add",
            std::make_unique<AddKernel>()
        );
        registry.Register(
            "Relu",
            std::make_unique<ReluKernel>()
        );
    }

}