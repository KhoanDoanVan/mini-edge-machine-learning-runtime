#pragma once


#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mini_ort/kernel.h"
#include "mini_ort/model.h"
#include "mini_ort/run_workspace.h"
#include "mini_ort/tensor_view.h"

namespace mini_ort {

    enum class ValueId : std::uint8_t {
        kInput,
        kScratch0,
        kScratch1,
        kOutput
    };

    enum class OutputShapePolicy :std::uint8_t {
        kSameAsFirstInput,
        kMatrix
    };

    struct ValueRef final {
        ValueId value = ValueId::kInput;
        const Tensor* initializer = nullptr;
    };

    struct Instruction final {
        const Kernel* kernel;
        std::array<ValueRef, 2> inputs;
        std::size_t input_count;
        ValueId output;
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

        private:

            std::vector<Instruction> instructions_;
    };

}