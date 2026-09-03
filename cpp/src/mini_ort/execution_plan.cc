#include "mini_ort/execution_plan.h"

#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <variant>
#include <utility>
#include <algorithm>

namespace mini_ort {

    namespace {

        ValueRef Input() noexcept {
            return ValueRef {
                ValueKind::kInput,
                0,
                nullptr
            };
        }

        ValueRef Output() noexcept {
            return ValueRef {
                ValueKind::kOutput,
                0,
                nullptr
            };
        }

        ValueRef Temporary(const std::size_t temporary_id) noexcept {
            return ValueRef {
                ValueKind::kTemporary,
                temporary_id,
                nullptr
            };
        }

        ValueRef Initializer(const Tensor& tensor) noexcept {
            return ValueRef {
                ValueKind::kInitializer,
                0,
                &tensor
            };
        }

        // ValueRef Activation (const ValueId value) noexcept {
        //     return ValueRef {
        //         value,
        //         nullptr
        //     };
        // }

        // ValueRef Initializer (const Tensor& tensor) noexcept {
        //     return ValueRef {
        //         ValueId::kInput,
        //         &tensor
        //     };
        // } 

        std::size_t LinearCount (const SequentialModel& model) noexcept {
            std::size_t count = 0;

            for (const auto& layer : model.layers()) {
                if (std::holds_alternative<LinearLayer>(layer)) {
                    ++count;
                }
            }

            return count;
        }

        // std::size_t ScratchIndex (const ValueId value) {
        //     if (value == ValueId::kScratch0) {
        //         return 0;
        //     }

        //     if (value == ValueId::kScratch1) {
        //         return 1;
        //     }

        //     throw std::logic_error("value is not a scratch slot");
        // }

        std::size_t FirstLinearInputFeatures(const SequentialModel& model) noexcept {
            for (const auto& layer : model.layers()) {
                if (const auto* linear = std::get_if<LinearLayer>(&layer)) {
                    return linear->in_features;
                }
            }
            
            return 0;
        }

        // bool IsScratch (const ValueId value) noexcept { return value == ValueId::kScratch0 || value == ValueId::kScratch1; }

        void MarkRead(
            const ValueRef value,
            const std::size_t instruction_index,
            std::vector<ActivationLifetime>& lifetimes
        ) {
            if (value.kind != ValueKind::kTemporary) {
                return;
            }

            if (value.temporary_id >= lifetimes.size()) {
                throw std::logic_error("instruction reads an unknown temporary");
            }

            lifetimes[value.temporary_id].last_instruction = instruction_index;
        }

        ConstTensorView ResolveValue(
            const ValueRef value,
            const ConstTensorView input,
            const TensorView output,
            // std::array<std::optional<TensorView>, RunWorkspace::kSlotCount>& scratch
            const MemoryPlan& memory_plan,
            const RunWorkspace& workspace
        ) {
            switch (value.kind) {
                case ValueKind::kInput:
                    return input;
                case ValueKind::kOutput:
                    return output.as_const();
                case ValueKind::kTemporary:
                    return workspace.Get(
                        memory_plan.slot_for(value.temporary_id)
                    );
                case ValueKind::kInitializer:
                    if (value.initializer == nullptr) {
                        throw std::logic_error("initializer value has no Tensor");
                    }
                    return value.initializer->view();
            }

            throw std::logic_error("execution plan contains an invalid value");
        }


        TensorView ResolveOutput(
            const Instruction& instruction,
            const ShapeView shape,
            const TensorView output,
            RunWorkspace& workspace,
            // std::array<std::optional<TensorView>, RunWorkspace::kSlotCount>& scratch
            const MemoryPlan& memory_plan
        ) {
            const auto value = instruction.output;

            if (value.kind == ValueKind::kOutput) {
                return output;
            }

            if (value.kind != ValueKind::kTemporary) {
                throw std::logic_error("instruction has an invalid output value");
            }

            const auto slot = memory_plan.slot_for(value.temporary_id);

            if (instruction.in_place) {
                auto existing = workspace.GetMutable(slot);

                if (existing.shape().size() != shape.size() || !std::equal(
                    existing.shape().begin(),
                    existing.shape().end(),
                    shape.begin()
                )) {
                    throw std::logic_error("in-place instruction changed activation shape");
                }

                return existing;
            }

            return workspace.Acquire(
                slot,
                shape
            );
        }


        // Instruction UnaryInstruction(
        //     const Kernel& kernel,
        //     const ValueId input,
        //     const ValueId output
        // ) noexcept {
        //     return Instruction {
        //         &kernel,
        //         {
        //             Activation(input),
        //             Activation(ValueId::kInput)
        //         },
        //         1,
        //         output,
        //         OutputShapePolicy::kSameAsFirstInput,
        //         0,
        //         input == output
        //     };
        // }

        // Instruction BinaryInstruction(
        //     const Kernel& kernel,
        //     const ValueRef lhs,
        //     const ValueRef rhs,
        //     const ValueId output,
        //     const OutputShapePolicy output_shape,
        //     const std::int64_t output_features
        // ) noexcept {
        //     return Instruction {
        //         &kernel,
        //         {
        //             lhs,
        //             rhs
        //         },
        //         2,
        //         output,
        //         output_shape,
        //         output_features,
        //         lhs.initializer == nullptr && lhs.value == output
        //     };
        // }



    } // namespace

    struct ExecutionPlan::Compilation final {
        std::vector<Instruction> instructions;
        std::vector<ActivationLifetime> lifetimes;
    };

    ExecutionPlan::ExecutionPlan(Compilation compilation) : instructions_(std::move(compilation.instructions)),
        memory_plan_(std::move(compilation.lifetimes)) {}

    ExecutionPlan::ExecutionPlan(
        const SequentialModel& model,
        const KernelRegistry& registry
    ) : ExecutionPlan(
        Compile(model, registry)
    ) {
        // instructions_.reserve(model.layers().size() * 2);
        // const auto& matmul = registry.Resolve("MatMul");
        // const auto& add = registry.Resolve("Add");
        // const auto& relu = registry.Resolve("Relu");

        // std::size_t remaining_linears = LinearCount(model);
        // std::size_t next_scratch = 0;
        // ValueId current = ValueId::kInput;

        // const auto acquire_scratch = [&]() {
        //     const auto value = next_scratch == 0 ? ValueId::kScratch0 : ValueId::kScratch1;
        //     next_scratch = (next_scratch + 1) % RunWorkspace::kSlotCount;
        //     return value;
        // };

        // for (const auto& layer : model.layers()) {

        //     if (const auto& linear = std::get_if<LinearLayer>(&layer)) {
        //         --remaining_linears;

        //         const auto destination = remaining_linears == 0 ? ValueId::kOutput : acquire_scratch();

        //         instructions_.push_back(
        //             BinaryInstruction(
        //                 matmul,
        //                 Activation(current),
        //                 Initializer(linear->weight),
        //                 destination,
        //                 OutputShapePolicy::kMatrix,
        //                 linear->weight.shape()[1]
        //             )
        //         );
        //     } else {

        //         if (current == ValueId::kInput) {
        //             current = remaining_linears == 0 ? ValueId::kOutput : acquire_scratch();

        //             instructions_.push_back(
        //                 UnaryInstruction(
        //                     relu,
        //                     ValueId::kInput,
        //                     current
        //                 )
        //             );
        //         } else {
        //             instructions_.push_back(
        //                 UnaryInstruction(
        //                     relu,
        //                     current,
        //                     current
        //                 )
        //             );
        //         }

        //     }

        // }
    }

    ExecutionPlan::Compilation ExecutionPlan::Compile(
        const SequentialModel& model,
        const KernelRegistry& registry
    ) {
        Compilation compilation;

        compilation.instructions.reserve(
            model.layers().size() * 2
        );
        compilation.lifetimes.reserve(
            model.layers().size()
        );

        const auto& matmul = registry.Resolve("MatMul");
        const auto& add = registry.Resolve("Add");
        const auto& relu = registry.Resolve("Relu");

        std::size_t remaining_linears = LinearCount(model);
        std::size_t current_features = FirstLinearInputFeatures(model);
        ValueRef current = Input();

        const auto new_temporary = [&](
            const std::size_t elements_per_batch
        ) {
            const auto temporary_id = compilation.lifetimes.size();
            const auto definition = compilation.instructions.size();

            compilation.lifetimes.push_back(
                ActivationLifetime {
                    definition,
                    definition,
                    elements_per_batch,
                    0
                }
            );

            return Temporary(temporary_id);
        };

        const auto append = [&](
            Instruction instruction
        ) {
            const auto instruction_index = compilation.instructions.size();

            for (std::size_t input_index = 0; input_index < instruction.input_count; ++input_index) {
                MarkRead(
                    instruction.inputs[input_index],
                    instruction_index,
                    compilation.lifetimes
                );
            }

            compilation.instructions.push_back(
                std::move(instruction)
            );
        };

        for (const auto& layer : model.layers()) {
            if (const auto* linear = std::get_if<LinearLayer>(&layer)) {
                --remaining_linears;

                const auto destination = remaining_linears == 0 ? Output() : new_temporary(linear->out_features);

                append(
                    Instruction {
                        &matmul,
                        {
                            current,
                            Initializer(linear->weight)
                        },
                        2,
                        destination,
                        OutputShapePolicy::kMatrix,
                        static_cast<std::int64_t>(linear->out_features),
                        false
                    }
                );

                current = destination;
                current_features = linear->out_features;

                if (linear->bias.has_value()) {
                    append(
                        Instruction {
                            &add,
                            {
                                current,
                                Initializer(*linear->bias)
                            },
                            2,
                            current,
                            OutputShapePolicy::kSameAsFirstInput,
                            0,
                            true
                        }
                    );
                }
            } else {
                if (current.kind == ValueKind::kInput) {
                    const auto destination = remaining_linears == 0 ? Output() : new_temporary(current_features);

                    append(
                        Instruction {
                            &relu,
                            {
                                current,
                                Input()
                            },
                            1,
                            destination,
                            OutputShapePolicy::kSameAsFirstInput,
                            0,
                            false
                        }
                    );

                    current = destination;
                } else {
                    append(
                        Instruction {
                            &relu,
                            {
                                current,
                                Input()
                            },
                            1,
                            current,
                            OutputShapePolicy::kSameAsFirstInput,
                            0,
                            true
                        }
                    );
                }
            }
        }

        return compilation;
    }
        


    void ExecutionPlan::Execute(
        const ConstTensorView input,
        const TensorView output,
        RunWorkspace& workspace
    ) const {

        // std::array<std::optional<TensorView>, RunWorkspace::kSlotCount> scratch;

        std::size_t batch_size = 0;

        if (!memory_plan_.slot_elements_per_batch().empty()) {
            if (input.shape().size() != 2 || input.shape()[0] < 0) {
                throw std::invalid_argument("planned temporary activations require a rank-2 input");
            }
            batch_size = static_cast<std::size_t>(input.shape()[0]);
        }


        workspace.Prepare(
            batch_size,
            memory_plan_.slot_elements_per_batch()
        );

        for (const auto& instruction : instructions_) {
            const auto first = ResolveValue(
                instruction.inputs[1],
                input,
                output,
                memory_plan_,
                // scratch
                workspace
            );

            std::optional<ConstTensorView> second;

            if (instruction.input_count == 2) {
                second.emplace(
                    ResolveValue(
                        instruction.inputs[1],
                        input,
                        output,
                        memory_plan_,
                        // scratch
                        workspace
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
                // scratch
                memory_plan_
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
    
    const MemoryPlan& ExecutionPlan::memory_plan() const noexcept {
        return memory_plan_;
    }
}