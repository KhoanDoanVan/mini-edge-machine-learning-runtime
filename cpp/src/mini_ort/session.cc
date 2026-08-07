#include "mini_ort/session.h"

#include <initializer_list>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>


#include "mini_ort/kernels/cpu.h"
#include "mini_ort/model_format.h"


namespace mini_ort {

    namespace {

        Tensor Execute(
            const KernelRegistry& registry,
            const std::string& op_type,
            const std::initializer_list<const Tensor*> inputs
        ) {
            const std::vector<const Tensor*> input_list(inputs);

            auto outputs = registry.Resolve(op_type).Compute(
                std::span<const Tensor* const>(
                    input_list.data(),
                    input_list.size()
                )
            );

            if (outputs.size() != 1) {
                throw std::runtime_error(op_type + " did not return one ouput");
            }

            return std::move(outputs.front());
        }

    }

    InferenceSession::InferenceSession(SequentialModel model) : model_(std::move(model)) {
        cpu::RegisterKernels(registry_);
    }

    InferenceSession::InferenceSession(const std::filesystem::path& model_path) : InferenceSession(LoadModel(model_path)) {}


    Tensor InferenceSession::Run(const Tensor& input) const {
        const Tensor* current = &input;
        std::optional<Tensor> owned_value;


        for (const auto& layer : model_.layers()) {
            
            if (const auto* linear = std::get_if<LinearLayer>(&layer)) {

                auto linear_output = Execute(
                    registry_,
                    "MatMul",
                    {
                        current,
                        &linear->weight
                    }
                );

                if (linear->bias.has_value()) {
                    owned_value = Execute(
                        registry_,
                        "Add",
                        {
                            &linear_output,
                            &*linear->bias
                        }
                    );
                } else {
                    owned_value = std::move(linear_output);
                }

            } else {
                owned_value = Execute(
                    registry_,
                    "Relu",
                    {
                        current
                    }
                );
            }

            current = &*owned_value;
        }

        return std::move(*owned_value);
    }

}