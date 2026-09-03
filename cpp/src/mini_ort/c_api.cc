#include "mini_ort/c_api.h"

#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_ort/model.h"
#include "mini_ort/session.h"
#include "mini_ort/tensor.h"
#include "mini_ort/model_format.h"


struct MiniOrtStatus {
    std::string message;
};

struct MiniOrtSession {
    explicit MiniOrtSession(mini_ort::SequentialModel model) : session(std::move(model)) {}

    mini_ort::InferenceSession session;
};


struct MiniOrtValue {
    explicit MiniOrtValue(mini_ort::Tensor value) : tensor(std::move(value)) {}

    mini_ort::Tensor tensor;
};

namespace {

    template <typename Function>
    MiniOrtStatus* Guard(Function&& function) {
        try {
            std::forward<Function>(function)();
            return nullptr;
        } catch (const std::exception& error) {
            return new MiniOrtStatus{
                error.what()
            };
        } catch (...) {
            return new MiniOrtStatus{
                "unknown native runtime error"
            };
        }
    }

    std::int64_t CheckedDimension(const std::size_t dimension) {
        if (dimension > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("layer dimension is too large");
        }
        return static_cast<std::int64_t>(dimension);
    }

    std::vector<float> CopyFloats(
        const float* data,
        const std::size_t count
    ) {
        if (count == 0) {
            return {};
        }

        if (data == nullptr) {
            throw std::invalid_argument("non-empty tensor data must not be null");
        }

        return std::vector<float>(data, data + count);
    }

} // namespace


const char* MiniOrtGetErrorMessage(const MiniOrtStatus* status) {
    return status == nullptr ? "" : status->message.c_str();
}


void MiniOrtReleaseStatus(MiniOrtStatus* status) {
    delete status;
}

MiniOrtStatus* MiniOrtCreateSession(
    const MiniOrtLayerDesc* layers,
    const size_t layer_count,
    MiniOrtSession** output
) {
    if (output != nullptr) {
        *output = nullptr;
    }

    return Guard(
        [&] {
            if (output == nullptr) {
                throw std::invalid_argument("session output pointer must not be null");
            }

            if (layers == nullptr && layer_count != 0) {
                throw std::invalid_argument("layer array must not be null");
            }

            std::vector<mini_ort::Layer> native_layers;

            native_layers.reserve(layer_count);

            for (std::size_t index = 0; index < layer_count; ++index) {

                const auto& layer = layers[index];

                if (layer.type == MINI_ORT_LAYER_LINEAR) {
                    mini_ort::Tensor weight(
                        {
                            CheckedDimension(layer.in_features),
                            CheckedDimension(layer.out_features)
                        },
                        CopyFloats(
                            layer.weight_data,
                            layer.weight_count
                        )
                    );

                    std::optional<mini_ort::Tensor> bias;

                    if (layer.bias_data != nullptr || layer.bias_count != 0) {
                        bias.emplace(
                            mini_ort::Shape{
                                CheckedDimension(layer.out_features)
                            },
                            CopyFloats(
                                layer.bias_data,
                                layer.bias_count
                            )
                        );
                    }

                    native_layers.emplace_back(
                        mini_ort::LinearLayer{
                            layer.in_features,
                            layer.out_features,
                            std::move(weight),
                            std::move(bias)
                        }
                    );
                
                } else if (layer.type == MINI_ORT_LAYER_RELU) {
                    native_layers.emplace_back(
                        mini_ort::ReluLayer{}
                    );
                } else {
                    throw std::invalid_argument("unsuppported layer type");
                }

            }

            *output = new MiniOrtSession(
                mini_ort::SequentialModel(
                    std::move(native_layers)
                )
            );
        }
    );
}


MiniOrtStatus* MiniOrtCreateSessionFromFile(
    const char* model_path,
    MiniOrtSession** output
) {
    if (output != nullptr) {
        *output = nullptr;
    }

    return Guard(
        [&] {
            if (model_path == nullptr || output == nullptr) {
                throw std::invalid_argument("model path and output must not be null");
            }
            *output = new MiniOrtSession(mini_ort::LoadModel(model_path));
        }
    );
}

size_t MiniOrtGetInputFeatureCount(const MiniOrtSession* session) {
    return session == nullptr ? 0 : session->session.InputFeatures();
}

size_t MiniOrtGetOutputFeatureCount(const MiniOrtSession* session) {
    return session == nullptr ? 0 : session->session.OutputFeatures();
}


void MiniOrtReleaseSession(MiniOrtSession* session) {
    delete session;
}


MiniOrtStatus* MiniOrtCreateFloatTensor(
    const float* data,
    const size_t element_count,
    const int64_t* shape,
    const size_t rank,
    MiniOrtValue** output
) {
    if (output != nullptr) {
        *output = nullptr;
    }

    return Guard(
        [&] {
            if (output == nullptr) {
                throw std::invalid_argument("tensor output pointer must not be null");
            }

            if (shape == nullptr && rank != 0) {
                throw std::invalid_argument("tensor shape must not be null");
            }

            mini_ort::Shape native_shape;

            if (rank != 0) {
                native_shape.assign(
                    shape,
                    shape + rank
                );
            }

            *output = new MiniOrtValue(
                mini_ort::Tensor(
                    std::move(native_shape),
                    CopyFloats(
                        data,
                        element_count
                    )
                )
            );
        }
    );
}


const float* MiniOrtGetTensorData(const MiniOrtValue* value) {
    return value == nullptr ? nullptr : value->tensor.data().data();
}

size_t MiniOrtGetTensorElementCount(const MiniOrtValue* value) {
    return value == nullptr ? 0 : value->tensor.size();
}

const int64_t* MiniOrtGetTensorShape(
    const MiniOrtValue* value,
    size_t* rank
) {
    if (rank != nullptr) {
        *rank = value == nullptr ? 0 : value->tensor.shape().size();
    }
    return value == nullptr ? nullptr : value->tensor.shape().data();
}

void MiniOrtReleaseValue(MiniOrtValue* value) {
    delete value;
}

MiniOrtStatus* MiniOrtRun(
    MiniOrtSession* session,
    const MiniOrtValue* input,
    MiniOrtValue** output
) {
    if (output != nullptr) {
        *output = nullptr;
    }

    return Guard(
        [&] {
            if (session == nullptr || input == nullptr || output == nullptr) {
                throw std::invalid_argument("session, input, and output must not be null");
            }
            *output = new MiniOrtValue(
                session->session.Run(input->tensor)
            );
        }
    );
}


MiniOrtStatus* MiniOrtRunInto(
    MiniOrtSession* session,
    const MiniOrtValue* input,
    MiniOrtValue* output
) {
    return Guard([&] {
        if (session == nullptr || input == nullptr || output == nullptr) {
            throw std::invalid_argument("session, input, and output must not be null");
        }

        session->session.RunInto(
            input->tensor.view(),
            output->tensor.mutable_view()
        );
    });
}