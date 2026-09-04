#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace mini_ort {

    struct ActivationLifetime final {
        std::size_t first_instruction;
        std::size_t last_instruction;
        std::size_t elements_per_batch;
        std::size_t arena_slot;
    };

    // Converts logical temporary lifetimes into resuable physical arena slots.
    // Two activations may share a slot only when their live intervals do not
    // overlap. Sizes remain expressed per batch so one plan supports dynamic batch.
    class MemoryPlan final {

        public:
            /**
             * @brief Pack non-overlapping activation lifetimes into arena slots.
             * @param lifetimes Logical temporary lifetimes indexed by temporary id.
             */
            explicit MemoryPlan(std::vector<ActivationLifetime> lifetimes);

            /** @brief Return the physical slot assigned to a temporary id. */
            [[nodiscard]] std::size_t slot_for(std::size_t temporary_id) const;
            /** @brief Return each slot's required elements per batch. */
            [[nodiscard]] std::span<const std::size_t> slot_elements_per_batch() const noexcept;
            /** @brief Return the original logical activation lifetimes. */
            [[nodiscard]] const std::vector<ActivationLifetime>& lifetimes() const noexcept;
            /** @brief Return the number of logical temporaries. */
            [[nodiscard]] std::size_t temporary_count() const noexcept;
            /** @brief Return the number of physical arena slots. */
            [[nodiscard]] std::size_t slot_count() const noexcept;

        private:
            std::vector<ActivationLifetime> lifetimes_;
            std::vector<std::size_t> slot_elements_per_batch_;

    };

}
