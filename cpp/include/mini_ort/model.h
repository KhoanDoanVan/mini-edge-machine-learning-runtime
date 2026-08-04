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

        explicit SequentialModel(std::vector<Layer> layers);

        [[nodiscard]] const std::vector<Layer>& layers() const noexcept;

        private:

        std::vector<Layer> layers_;

    };

}