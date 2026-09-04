#pragma once

#include <cstddef>
#include <optional>
#include <variant>
#include <vector>

#include "mini_ort/tensor.h"

namespace mini_ort {

    struct LinearLayer {
        std::size_t in_features;
        std::size_t out_features;
        Tensor weight;
        std::optional<Tensor> bias;
    };

    struct ReluLayer {};

    using Layer = std::variant<LinearLayer, ReluLayer>;

    class SequentialModel final {

        public:

        /**
         * @brief Construct and validate a sequential model.
         * @param layers Ordered layer definitions to execute.
         * @throws std::invalid_argument when layer dimensions are inconsistent.
         */
        explicit SequentialModel(std::vector<Layer> layers);

        /** @brief Return the validated layers in execution order. */
        [[nodiscard]] const std::vector<Layer>& layers() const noexcept;

        private:

        std::vector<Layer> layers_;

    };

}
