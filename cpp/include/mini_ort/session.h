#pragma once

#include <filesystem>
#include "mini_ort/kernel.h"
#include "mini_ort/model.h"
#include "mini_ort/tensor.h"
#include "mini_ort/run_workspace.h"
#include "mini_ort/execution_plan.h"

namespace mini_ort {

    class InferenceSession final {
        
        public:

        explicit InferenceSession(SequentialModel model);
        explicit InferenceSession(const std::filesystem::path& model_path);
        InferenceSession(const InferenceSession&) = delete;
        InferenceSession& operator = (const InferenceSession&) = delete;
        // InferenceSession(InferenceSession&&) noexcept = default;
        // InferenceSession& operator = (InferenceSession&&) noexcept = default;
        InferenceSession(InferenceSession&& other);
        InferenceSession& operator = (InferenceSession&& other);

        // Run reuses internal scratch buffers and must not be called concurrently on
        // the same session.
        [[nodiscard]] Tensor Run(const Tensor& input) const;
        void RunInto(
            ConstTensorView input,
            TensorView output
        ) const;
        [[nodiscard]] Shape OutputShape(ConstTensorView input) const;
        [[nodiscard]] std::size_t InputFeatures() const noexcept;
        [[nodiscard]] std::size_t OutputFeatures() const noexcept;
        [[nodiscard]] std::size_t InstructionCount() const noexcept;

        
        private:

        SequentialModel model_;
        KernelRegistry registry_;
        mutable RunWorkspace workspace_;
        ExecutionPlan plan_;
    };

}