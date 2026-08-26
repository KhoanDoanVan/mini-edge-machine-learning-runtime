#include "mini_ort/memory_plan.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>


namespace mini_ort {

    MemoryPlan::MemoryPlan(
        std::vector<ActivationLifetime> lifetimes
    ) : lifetimes_(std::move(lifetimes)) {

        struct SlotState final {
            std::size_t last_instruction;
            std::size_t elements_per_batch;
        };

        std::vector<std::size_t> order(lifetimes_.size());
        std::iota(
            order.begin(),
            order.end(),
            0
        );
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](const auto lhs, const auto rhs) {
                return lifetimes_[lhs].first_instruction < lifetimes_[rhs].first_instruction;
            }
        );

        std::vector<SlotState> slots;

        for (const auto temporary_id : order) {
            auto& lifetime = lifetimes_[temporary_id];

            if (lifetime.first_instruction > lifetime.last_instruction) {
                throw std::invalid_argument("activation lifetime is invalid");
            }

            if (lifetime.elements_per_batch == 0) {
                throw std::invalid_argument("temporary activation cannot be empty");
            }

            std::size_t best_slot = slots.size();
            std::size_t best_capacity = std::numeric_limits<std::size_t>::max();

            for (std::size_t slot = 0; slot < slots.size(); ++slot) {
                // Strictly less is important: an instruction cannot overwrite an input
                // activation while that same instruction is still reading it.
                if (slots[slot].last_instruction >= lifetime.first_instruction) {
                    continue;
                }

                const auto capacity = std::max(
                    slots[slot].elements_per_batch,
                    lifetime.elements_per_batch
                );

                if (capacity < best_capacity) {
                    best_slot = slot;
                    best_capacity = capacity;
                }
            }

            if (best_slot == slots.size()) {
                best_slot = slots.size();
                slots.push_back(
                    SlotState {
                        lifetime.last_instruction,
                        lifetime.elements_per_batch
                    }
                );
            } else {
                slots[best_slot].last_instruction = lifetime.last_instruction;
                slots[best_slot].elements_per_batch = best_capacity;
            }

            lifetime.arena_slot = best_slot;
        }

        slot_elements_per_batch_.reserve(slots.size());

        for (const auto& slot : slots) {
            slot_elements_per_batch_.push_back(
                slot.elements_per_batch
            );
        }
    }

    std::size_t MemoryPlan::slot_for(
        const std::size_t temporary_id
    ) const {
        if (temporary_id >= lifetimes_.size()) {
            throw std::out_of_range("temporary activation is out of range");
        }
        return lifetimes_[temporary_id].arena_slot;
    }

    std::span<const std::size_t> MemoryPlan::slot_elements_per_batch() const noexcept {
        return slot_elements_per_batch_;
    }

    const std::vector<ActivationLifetime>& MemoryPlan::lifetimes() const noexcept {
        return lifetimes_;
    }

    std::size_t MemoryPlan::temporary_count() const noexcept {
        return lifetimes_.size();
    }

    std::size_t MemoryPlan::slot_count() const noexcept {
        return slot_elements_per_batch_.size();
    }


}