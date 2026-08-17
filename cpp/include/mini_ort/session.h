#pragma once

#include <filesystem>
#include "mini_ort/kernel.h"
#include "mini_ort/model.h"
#include "mini_ort/tensor.h"

namespace mini_ort {

    class InferenceSession final {
        
        public:

        explicit InferenceSession(SequentialModel model);
        explicit InferenceSession(const std::filesystem::path& model_path);
        InferenceSession(const InferenceSession&) = delete;
        InferenceSession& operator = (const InferenceSession&) = delete;
        InferenceSession(InferenceSession&&) noexcept = default;
        InferenceSession& operator = (InferenceSession&&) noexcept = default;

        [[nodiscard]] Tensor Run(const Tensor& input) const;

        private:

        SequentialModel model_;
        KernelRegistry registry_;
        
    };

}