#include "mini_ort/execution_plan.h"

#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <variant>

namespace mini_ort {

    namespace {

        ValueRef Activation (const ValueId value) noexcept {
            return ValueRef {
                value,
                nullptr
            };
        }

        ValueRef Initializer (const Tensor& tensor) noexcept {
            return ValueRef {
                ValueId::kInput,
                &tensor
            };
        } 

        std::size_t LinearCount (const SequentialModel& model) noexcept {
            std::size_t count = 0;

            for (const auto& layer : model.layers()) {
                if (std::holds_alternative<LinearLayer>(layer)) {
                    ++count;
                }
            }

            return count;
        }

        std::size_t ScratchIndex (const ValueId value) {
            if (value == ValueId::kScratch0) {
                return 0;
            }

            if (value == ValueId::kScratch1) {
                return 1;
            }

            throw std::logic_error("value is not a scratch slot");
        }

        bool IsScratch (const ValueId value) noexcept { return value == ValueId::kScratch0 || value == ValueId::kScratch1; }

        ConstTensorView ResolveView(
            const ValueRef value,
            const ConstTensorView input,
            const TensorView output,
            std::array<std::optional<TensorView>, RunWorkspace::kSlotCount>& scratch
        ) {
            if (value.initializer != nullptr) {
                return value.initializer->view();
            }

            switch (value.value) {

                case ValueId::kInput:
                    return input;

                case ValueId::kOutput:
                    return output.as_const();

                // kScratch0 & kScratch1 are same block
                case ValueId::kScratch0:
                case ValueId::kScratch1:
                    {
                        const auto index = ScratchIndex(value.value);

                        if (!scratch[index].has_value()) {
                            throw std::logic_error("execution plan read an unavailable scratch slot");
                        }

                        return scratch[index]->as_const();
                    }
            }

            throw std::logic_error("execution plan contains an invalid value");
        }


        TensorView ResolveOutput(
            const Instruction& instruction,
            const ShapeView shape,
            const TensorView output,
            RunWorkspace& workspace,
            std::array<std::optional<TensorView>, RunWorkspace::kSlotCount>& scratch
        ) {
            if (instruction.output == ValueId::kOutput) {
                return output;
            }

            if (!IsScratch(instruction.output)) {
                throw std::logic_error("execution plan cannot write to the model input");
            }

            const auto index = ScratchIndex(instruction.output);

            if (instruction.in_place) {
                if (!scratch[index].has_value()) {
                    throw std::logic_error("in-place instruction has not scratch buffer");
                }

                return *scratch[index];
            }

            scratch[index] = workspace.Acquire(index, shape);

            return *scratch[index];
        }


        Instruction UnaryInstruction(
            const Kernel& kernel,
            const ValueId input,
            const ValueId output
        ) noexcept {
            return Instruction {
                &kernel,
                {
                    Activation(input),
                    Activation(ValueId::kInput)
                },
                1,
                output,
                OutputShapePolicy::kSameAsFirstInput,
                0,
                input == output
            };
        }

        Instruction BinaryInstruction(
            const Kernel& kernel,
            const ValueRef lhs,
            const ValueRef rhs,
            const ValueId output,
            const OutputShapePolicy output_shape,
            const std::int64_t output_features
        ) noexcept {
            return Instruction {
                &kernel,
                {
                    lhs,
                    rhs
                },
                2,
                output,
                output_shape,
                output_features,
                lhs.initializer == nullptr && lhs.value == output
            };
        }

    }
    
}