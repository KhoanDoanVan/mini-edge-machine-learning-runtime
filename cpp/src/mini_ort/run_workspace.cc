#include "mini_ort/run_workspace.h"

#include <limits>
#include <stdexcept>
#include <utility>

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

    RunWorkspace::RunWorkspace(Allocator& allocator) noexcept : allocator_(&allocator) {}

    TensorView RunWorkspace::Acquire(
        const std::size_t slot_index,
        const ShapeView shape
    ) {

        if (slot_index >= slots_.size()) {
            throw std::out_of_range("workspace slot is out of range");
        }

        const auto required_elements = ElementCount(shape);

        auto& slot = slots_[slot_index];

        slot.shape.assign(
            shape.begin(),
            shape.end()
        );

        if (!slot.storage.has_value() || slot.storage->size() < required_elements) {
            TensorStorage replacement(
                required_elements,
                *allocator_
            );
            slot.storage = std::move(replacement);
        }

        return TensorView(
            slot.shape,
            slot.storage->mutable_data().first(required_elements)
        );

    }

}