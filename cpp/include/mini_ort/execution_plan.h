#pragma once


#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mini_ort/kernel.h"
#include "mini_ort/model.h"
#include "mini_ort/run_workspace.h"
#include "mini_ort/tensor_view.h"
#include "mini_ort/memory_plan.h"

namespace mini_ort {

    // enum class ValueId : std::uint8_t {
    //     kInput,
    //     kScratch0,
    //     kScratch1,
    //     kOutput
    // };

    enum class ValueKind : std::uint8_t {
        kInput,
        kTemporary,
        kOutput,
        kInitializer
    };

    enum class OutputShapePolicy :std::uint8_t {
        kSameAsFirstInput,
        kMatrix
    };

    struct ValueRef final {
        // ValueId value = ValueId::kInput;
        ValueKind kind = ValueKind::kInput;
        std::size_t temporary_id = 0;
        const Tensor* initializer = nullptr;
    };

    struct Instruction final {
        const Kernel* kernel;
        std::array<ValueRef, 2> inputs;
        std::size_t input_count;
        ValueRef output;
        OutputShapePolicy output_shape;
        std::int64_t output_features;
        bool in_place;
    };

    class ExecutionPlan final {
            
        public:

            ExecutionPlan(
                const SequentialModel& model,
                const KernelRegistry& registry
            );  

            void Execute(
                ConstTensorView input,
                TensorView output,
                RunWorkspace& workspace
            ) const;

            [[nodiscard]] std::size_t size() const noexcept;
            [[nodiscard]] const std::vector<Instruction>& instructions() const noexcept;
            [[nodiscard]] const MemoryPlan& memory_plan() const noexcept;

        private:

            struct Compilation;

            explicit ExecutionPlan(Compilation compilation);
            [[nodiscard]] static Compilation Compile(
                const SequentialModel& model,
                const KernelRegistry& registry
            );

            std::vector<Instruction> instructions_;
            MemoryPlan memory_plan_;
    };

}