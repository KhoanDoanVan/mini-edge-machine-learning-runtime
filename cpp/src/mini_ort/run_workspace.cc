#include "mini_ort/run_workspace.h"

#include <limits>
#include <stdexcept>
// #include <utility>

namespace mini_ort {

    namespace {

        std::size_t ElementCount(const ShapeView shape) {
            std::size_t count = 1;

            for (const auto dimension : shape) {
                if (dimension < 0) {
                    throw std::invalid_argument("workspace dimensions must be non-negative");
                }

                const auto size = static_cast<std::size_t>(dimension);

                if (size == 0) {
                    count = 0;
                    continue;
                }

                if (count > std::numeric_limits<std::size_t>::max() / size) {
                    throw std::overflow_error("workspace element count overflows size_t");
                }

                count *= size;
            }

            return count;
        }


    }

    RunWorkspace::RunWorkspace(Allocator& allocator) noexcept : arena_(allocator) {}

    void RunWorkspace::Prepare(
        const std::size_t batch_size,
        const std::span<const std::size_t> slot_elements_per_batch
    ) {
        concrete_slot_elements_.resize(slot_elements_per_batch.size());
        slots_.resize(slot_elements_per_batch.size());

        for (std::size_t slot = 0; slot < slot_elements_per_batch.size(); ++slot) {
            const auto elements_per_batch = slot_elements_per_batch[slot];
            if (elements_per_batch != 0 && batch_size > std::numeric_limits<std::size_t>::max() / elements_per_batch) {
                throw std::overflow_error("workspace slot size overflows size_t");
            }
            concrete_slot_elements_[slot] = batch_size * elements_per_batch;
            slots_[slot].available = false;
        }
        arena_.Configure(concrete_slot_elements_);
    }

    TensorView RunWorkspace::Acquire(
        const std::size_t slot_index,
        const ShapeView shape
    ) {

        if (slot_index >= slots_.size()) {
            throw std::out_of_range("workspace slot is out of range");
        }

        const auto required_elements = ElementCount(shape);

        auto& slot = slots_[slot_index];

        if (required_elements > concrete_slot_elements_[slot_index]) {
            throw std::invalid_argument("activation exceeds its planned arena slot");
        }

        slot.shape.assign(
            shape.begin(),
            shape.end()
        );

        // if (!slot.storage.has_value() || slot.storage->size() < required_elements) {
        //     TensorStorage replacement(
        //         required_elements,
        //         *allocator_
        //     );
        //     slot.storage = std::move(replacement);
        // }
        slot.active_elements = required_elements;
        slot.available = true;

        return TensorView(
            slot.shape,
            // slot.storage->mutable_data().first(required_elements)
            arena_.Slice(
                slot_index,
                required_elements
            )
        );

    }

    ConstTensorView RunWorkspace::Get(const std::size_t slot_index) const {
        if (slot_index >= slots_.size()) {
            throw std::out_of_range("workspace slot is out of range");
        }

        const auto& slot = slots_[slot_index];

        if (!slot.available) {
            throw std::logic_error("execution plan read an unavailable arena slot");
        }

        return ConstTensorView(
            slot.shape,
            arena_.Slice(
                slot_index,
                slot.active_elements
            )
        );
    }

    TensorView RunWorkspace::GetMutable(const std::size_t slot_index) {
        if (slot_index >= slots_.size()) {
            throw std::out_of_range("workspace slot is out of range");
        }

        const auto& slot = slots_[slot_index];

        if (!slot.available) {
            throw std::logic_error("execution plan wrote an unavailable arena slot");
        }

        return TensorView(
            slot.shape,
            arena_.Slice(
                slot_index,
                slot.active_elements
            )
        );
    }

    std::size_t RunWorkspace::arena_layout_bytes() const noexcept {
        return arena_.layout_bytes();
    }

    std::size_t RunWorkspace::arena_capacity_bytes() const noexcept {
        return arena_.capacity_bytes();
    }

}