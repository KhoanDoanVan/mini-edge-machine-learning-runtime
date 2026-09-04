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

        /** @brief Construct a session from an already validated model. */
        explicit InferenceSession(SequentialModel model);
        /** @brief Load and construct a session from a host filesystem path. */
        explicit InferenceSession(const std::filesystem::path& model_path);
        InferenceSession(const InferenceSession&) = delete;
        InferenceSession& operator = (const InferenceSession&) = delete;
        // InferenceSession(InferenceSession&&) noexcept = default;
        // InferenceSession& operator = (InferenceSession&&) noexcept = default;
        /** @brief Transfer session ownership without copying model state. */
        InferenceSession(InferenceSession&& other);
        /** @brief Replace this session by transferring state from `other`. */
        InferenceSession& operator = (InferenceSession&& other);

        // Run reuses internal scratch buffers and must not be called concurrently on
        // the same session.
        /** @brief Execute inference and allocate an owning output tensor. */
        [[nodiscard]] Tensor Run(const Tensor& input) const;
        /** @brief Execute inference into caller-provided output storage. */
        void RunInto(
            ConstTensorView input,
            TensorView output
        ) const;
        /** @brief Infer the concrete output shape for an input view. */
        [[nodiscard]] Shape OutputShape(ConstTensorView input) const;
        /** @brief Return the model input feature count. */
        [[nodiscard]] std::size_t InputFeatures() const noexcept;
        /** @brief Return the model output feature count. */
        [[nodiscard]] std::size_t OutputFeatures() const noexcept;
        /** @brief Return compiled instruction count. */
        [[nodiscard]] std::size_t InstructionCount() const noexcept;
        /** @brief Return logical temporary tensor count. */
        [[nodiscard]] std::size_t TemporaryCount() const noexcept;
        /** @brief Return physical arena slot count. */
        [[nodiscard]] std::size_t ArenaSlotCount() const noexcept;
        [[nodiscard]] std::size_t ArenaLayoutBytes() const noexcept;
        [[nodiscard]] std::size_t ArenaCapacityBytes() const noexcept;

        
        private:

        SequentialModel model_;
        KernelRegistry registry_;
        mutable RunWorkspace workspace_;
        ExecutionPlan plan_;
    };

}
