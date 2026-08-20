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

        void Execute(
            const KernelRegistry& registry,
            const std::string& op_type,
            const std::initializer_list<ConstTensorView> inputs,
            const TensorView output
        ) {
            // const std::vector<const Tensor*> input_list(inputs);

            // auto outputs = registry.Resolve(op_type).Compute(
            //     std::span<const Tensor* const>(
            //         input_list.data(),
            //         input_list.size()
            //     )
            // );

            // if (outputs.size() != 1) {
            //     throw std::runtime_error(op_type + " did not return one ouput");
            // }

            // return std::move(outputs.front());
            registry.Resolve(op_type).Compute(
                std::span<const ConstTensorView>(
                    inputs.begin(),
                    inputs.size()
                ),
                output
            );
        }


        Shape MatMulOutputShape(
            const ConstTensorView lhs,
            const ConstTensorView rhs
        ) {
            if (lhs.shape().size() != 2 || rhs.shape().size() != 2) {
                throw std::invalid_argument("MatMul expects rank-2 tensors");
            }

            if (lhs.shape()[1] != rhs.shape()[0]) {
                throw std::invalid_argument("MatMul inner dimensions must match");
            }

            return Shape {
                lhs.shape()[0],
                rhs.shape()[1]
            };
        }

        std::size_t LinearCount(const SequentialModel& model) noexcept {
            std::size_t count = 0;

            for (const auto& layer : model.layers()) {
                if (std::holds_alternative<LinearLayer>(layer)) {
                    ++count;
                }
            }

            return count;
        }

    } // namespace

    InferenceSession::InferenceSession(SequentialModel model) : model_(std::move(model)) {
        cpu::RegisterKernels(registry_);
    }

    InferenceSession::InferenceSession(const std::filesystem::path& model_path) : InferenceSession(LoadModel(model_path)) {}


    Tensor InferenceSession::Run(const Tensor& input) const {

        auto current = input.view();
        std::optional<TensorView> writable_current;
        std::optional<Tensor> result;

        std::size_t remaining_linears = LinearCount(model_);
        std::size_t next_scratch_slot = 0;

        // const Tensor* current = &input;
        // std::optional<Tensor> owned_value;

        const auto acquire_scratch = [&](const ShapeView shape) {
            const auto slot = next_scratch_slot;
            next_scratch_slot = (next_scratch_slot + 1) % RunWorkspace::kSlotCount;
            return workspace_.Acquire(
                slot,
                shape
            );
        };
        

        for (const auto& layer : model_.layers()) {
            
            if (const auto* linear = std::get_if<LinearLayer>(&layer)) {

                // const auto current_view = current->view();
                const auto weight_view = linear->weight.view();
                const auto output_shape = MatMulOutputShape(
                    current,
                    weight_view
                );
                --remaining_linears;

                std::optional<TensorView> destination;

                if (remaining_linears == 0) {
                    result.emplace(output_shape);
                    destination.emplace(result->mutable_view());
                } else {
                    destination.emplace(
                        acquire_scratch(output_shape)
                    );
                }

                // Tensor linear_output(
                //     MatMulOutputShape(
                //         current_view,
                //         weight_view
                //     )
                // );

                Execute(
                    registry_,
                    "MatMul",
                    {
                        // current_view,
                        current,
                        weight_view
                    },
                    *destination
                );

                if (linear->bias.has_value()) {
                    Execute(
                        registry_,
                        "Add",
                        {
                            destination.as_count(),
                            linear->bias->view()
                        },
                        *destination
                    );
                }

                // owned_value = std::move(linear_output);

                // auto linear_output = Execute(
                //     registry_,
                //     "MatMul",
                //     {
                //         current,
                //         &linear->weight
                //     }
                // );

                // if (linear->bias.has_value()) {
                //     owned_value = Execute(
                //         registry_,
                //         "Add",
                //         {
                //             &linear_output,
                //             &*linear->bias
                //         }
                //     );
                // } else {
                //     owned_value = std::move(linear_output);
                // }

                current = destination->ac_const();
                writable_current = *destination;

            } else {
                // owned_value = Execute(
                //     registry_,
                //     "Relu",
                //     {
                //         current
                //     }
                // );
                // if (owned_value.has_value()) {
                //     Execute(
                //         registry_,
                //         "Relu",
                //         {
                //             owned_value->view()
                //         },
                //         owned_value->mutable_view()
                //     );
                // } else {
                //     Tensor relu_output(current->shape());
                //     Execute(
                //         registry_,
                //         "Relu",
                //         {
                //             current->view()
                //         },
                //         relu_output.mutable_view()
                //     );
                //     owned_value = std::move(relu_output);
                // }
                if (writable_current.has_value()) {
                    Execute(
                        registry_,
                        "Relu",
                        {
                            current
                        },
                        *writable_current
                    );
                } else {
                    std::optional<TensorView> destination;

                    if (remaining_linears == 0) {
                        result.emplace(
                            Shape(
                                current.shape().begin(),
                                current.shape().end()
                            )
                        );
                        destination.emplace(
                            result->mutable_view()
                        );
                    } else {
                        destination.emplace(
                            acquire_scratch(
                                current.shape()
                            )
                        );
                    }

                    Execute(
                        registry_,
                        "Relu",
                        {
                            current
                        },
                        *destination
                    );

                    current = destination->as_const();

                    writable_current = *destination;
                }
            }

            // current = &*owned_value;
        }

        // return std::move(*owned_value);
        return std::move(*result);
    }

} // namespace