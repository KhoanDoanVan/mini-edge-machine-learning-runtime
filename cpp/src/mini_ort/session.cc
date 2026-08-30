#include "mini_ort/session.h"

// #include <initializer_list>
// #include <optional>
// #include <string>
#include <stdexcept>
#include <utility>
// #include <vector>


#include "mini_ort/kernels/cpu.h"
#include "mini_ort/model_format.h"


namespace mini_ort {

    namespace {

        KernelRegistry CreateCpuRegistry() {
            KernelRegistry registry;
            cpu::RegisterKernels(registry);
            return registry;
        }

    } // namespace

    InferenceSession::InferenceSession(SequentialModel model) : model_(std::move(model)), registry_(CreateCpuRegistry()), plan_(model_, registry_) {}

    InferenceSession::InferenceSession(
        const std::filesystem::path& model_path
    ) : InferenceSession(
        LoadModel(model_path)
    ) {}

    InferenceSession::InferenceSession(InferenceSession&& other) : model_(std::move(other.model_)), registry_(std::move(other.registry_)), plan_(model_, registry_), workspace_(std::move(other.workspace_)) {}

    InferenceSession& InferenceSession::operator=(InferenceSession&& other) {
        if (this == &other) {
            return *this;
        }

        model_ = std::move(other.model_);   
        registry_ = std::move(other.registry_);
        plan_ = ExecutionPlan(
            model_,
            registry_
        );
        workspace_ = std::move(other.workspace_);

        return *this;
    }

    Tensor InferenceSession::Run(const Tensor& input) const {
        Tensor output(
            OutputShape (
                input.view()
            )
        );
        RunInto(
            input.view(),
            output.mutable_view()
        );
        return output;
    }

    Shape InferenceSession::OutputShape(const ConstTensorView input) const {
        Shape output_shape(
            input.shape().begin(),
            input.shape().end()
        );

        for (const auto& layer : model_.layers()) {

            if (const auto* linear = std::get_if<LinearLayer>(&layer)) {
                if (output_shape.size() != 2 || output_shape[1] != static_cast<std::int64_t>(linear->in_features)) {
                    throw std::invalid_argument("input shape is incompatible with a Linear layer");
                }
                output_shape[1] = static_cast<std::int64_t>(linear->out_features);
            }
        }
        return output_shape;
    }

    std::size_t InferenceSession::InputFeatures() const noexcept {
        for (const auto& layer : model_.layers()) {
            if (const auto* linear = std::get_if<LinearLayer>(&layer)) {
                return linear->in_features;
            }
        }
        return 0;
    }

    std::size_t InferenceSession::OutputFeatures() const noexcept {
        for (auto layer = model_.layers().rbegin(); layer != model_.layers().rend(); ++layer) {
            if (const auto* linear = std::get_if<LinearLayer>(&*layer)) {
                return linear->out_features;
            }
        }
        return 0;
    }

    std::size_t InferenceSession::InstructionCount() const noexcept {
        return plan_.size();
    }

    std::size_t InferenceSession::TemporaryCount() const noexcept {
        return plan_.memory_plan().temporary_count();
    }

    std::size_t InferenceSession::ArenaSlotCount() const noexcept {
        return plan_.memory_plan().slot_count();
    }

    std::size_t InferenceSession::ArenaLayoutBytes() const noexcept {
        return workspace_.arena_layout_bytes();
    }

    std::size_t InferenceSession::ArenaCapacityBytes() const noexcept {
        return workspace_.arena_capacity_bytes();
    }


    void InferenceSession::RunInto(
        const ConstTensorView input,
        const TensorView output
    ) const {
        plan_.Execute(
            input,
            output,
            workspace_
        );
    }

} // namespace