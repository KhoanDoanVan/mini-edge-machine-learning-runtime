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

        ConstTensorView ResolveValue(
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

    } // namespace


    ExecutionPlan::ExecutionPlan(
        const SequentialModel& model,
        const KernelRegistry& registry
    ) {
        instructions_.reserve(model.layers().size() * 2);
        const auto& matmul = registry.Resolve("MatMul");
        const auto& add = registry.Resolve("Add");
        const auto& relu = registry.Resolve("Relu");

        std::size_t remaining_linears = LinearCount(model);
        std::size_t next_scratch = 0;
        ValueId current = ValueId::kInput;

        const auto acquire_scratch = [&]() {
            const auto value = next_scratch == 0 ? ValueId::kScratch0 : ValueId::kScratch1;
            next_scratch = (next_scratch + 1) % RunWorkspace::kSlotCount;
            return value;
        };

        for (const auto& layer : model.layers()) {

            if (const auto& linear = std::get_if<LinearLayer>(&layer)) {
                --remaining_linears;

                const auto destination = remaining_linears == 0 ? ValueId::kOutput : acquire_scratch();

                instructions_.push_back(
                    BinaryInstruction(
                        matmul,
                        Activation(current),
                        Initializer(linear->weight),
                        destination,
                        OutputShapePolicy::kMatrix,
                        linear->weight.shape()[1]
                    )
                );
            } else {

                if (current == ValueId::kInput) {
                    current = remaining_linears == 0 ? ValueId::kOutput : acquire_scratch();

                    instructions_.push_back(
                        UnaryInstruction(
                            relu,
                            ValueId::kInput,
                            current
                        )
                    );
                } else {
                    instructions_.push_back(
                        UnaryInstruction(
                            relu,
                            current,
                            current
                        )
                    );
                }

            }

        }
    }


    void ExecutionPlan::Execute(
        const ConstTensorView input,
        const TensorView output,
        RunWorkspace& workspace
    ) const {

        std::array<std::optional<TensorView>, RunWorkspace::kSlotCount> scratch;

        for (const auto& instruction : instructions_) {
            const auto first = ResolveValue(
                instruction.inputs[1],
                input,
                output,
                scratch
            );

            std::optional<ConstTensorView> second;

            if (instruction.input_count == 2) {
                second.emplace(
                    ResolveValue(
                        instruction.inputs[1],
                        input,
                        output,
                        scratch
                    )
                );
            }

            std::array<std::int64_t, 2> matrix_shape{};

            ShapeView output_shape = first.shape();

            if (instruction.output_shape == OutputShapePolicy::kMatrix) {
                if (first.shape().size() != 2) {
                    throw std::invalid_argument("MatMul expects a rank-2 activation");
                }

                matrix_shape = {
                    first.shape()[0],
                    instruction.output_features
                };

                output_shape = ShapeView(
                    matrix_shape.data(),
                    matrix_shape.size()
                );
            }

            auto destination = ResolveOutput(
                instruction,
                output_shape,
                output,
                workspace,
                scratch
            );

            if (instruction.input_count == 1) {
                const std::array inputs{first};
                instruction.kernel->Compute(
                    std::span(inputs),
                    destination
                );
            } else {
                const std::array inputs{first, *second};
                instruction.kernel->Compute(
                    std::span(inputs),
                    destination
                );
            }
        }

    }

    std::size_t ExecutionPlan::size() const noexcept {
        return instructions_.size();
    }

    const std::vector<Instruction>& ExecutionPlan::instructions() const noexcept {
        return instructions_;
    }
    
}