#include "mini_ort/model.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>


namespace mini_ort {

    SequentialModel::SequentialModel(std::vector<Layer> layers) : layers_(std::move(layers)) {
        if (layers_.empty()) {
            throw std::invalid_argument("SequentialModel requires at least one layer");
        }

        std::optional<std::size_t> current_features;

        for (const auto& layer : layers_) {
            if (const auto* linear = std::get_if<LinearLayer>(&layer)) {

                if (linear->in_features == 0 || linear->out_features == 0) {
                    throw std::invalid_argument("Linear feature counts must be positive");
                }
                
                const Shape expected_weight {
                    static_cast<std::int64_t>(linear->in_features),
                    static_cast<std::int64_t>(linear->out_features)
                };

                if (linear->weight.shape() != expected_weight) {
                    throw std::invalid_argument("Linear weight shape is invalid");
                }

                if (linear->bias.has_value() && linear->bias->shape() != Shape{static_cast<std::int64_t>(linear->out_features)}) {
                    throw std::invalid_argument("Linear bias shape is invalid");
                }

                if (current_features.has_value() && *current_features != linear->in_features) {
                    throw std::invalid_argument("adjacent Linear layers have incompatible feature counts");
                }

                current_features = linear->out_features;
            }
        }
    }

    const std::vector<Layer>& SequentialModel::layers() const noexcept {
        return layers_;
    }

}