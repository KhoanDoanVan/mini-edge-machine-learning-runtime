#pragma once

#include <span>
#include <cstddef>
#include <vector>

#include "mini_ort/allocator.h"
// #include "mini_ort/tensor_storage.h"
#include "mini_ort/tensor_view.h"
#include "mini_ort/tensor_arena.h"


namespace mini_ort {

    class RunWorkspace final {

        public:

            // static constexpr std::size_t kSlotCount = 2;

            /**
             * @brief Construct reusable per-session execution storage.
             * @param allocator Allocator used when the arena grows.
             */
            explicit RunWorkspace(
                Allocator& allocator = DefaultAllocator()
            ) noexcept;

            /**
             * @brief Materialize slot capacities for a concrete batch size.
             * @param batch_size Runtime batch dimension.
             * @param slot_elements_per_batch Plan capacities before batching.
             */
            void Prepare(
                std::size_t batch_size,
                std::span<const std::size_t> slot_elements_per_batch
            );

            /** @brief Acquire writable storage for a planned slot and shape. */
            [[nodiscard]] TensorView Acquire(
                std::size_t slot,
                ShapeView shape
            );
            /** @brief Return an existing slot as a read-only view. */
            [[nodiscard]] ConstTensorView Get(std::size_t slot) const;
            /** @brief Return an existing slot as a writable view. */
            [[nodiscard]] TensorView GetMutable(std::size_t slot);
            /** @brief Return the planned arena layout in bytes. */
            [[nodiscard]] std::size_t arena_layout_bytes() const noexcept;
            /** @brief Return the allocated arena capacity in bytes. */
            [[nodiscard]] std::size_t arena_capacity_bytes() const noexcept;

        private:

            struct Slot final {
                Shape shape;
                // std::optional<TensorStorage> storage;
                std::size_t active_elements = 0;
                bool available = false;
            };

            // Allocator* allocator_;
            TensorArena arena_;
            // std::array<Slot> slots_;
            std::vector<Slot> slots_;
            std::vector<std::size_t> concrete_slot_elements_;

    };

}
