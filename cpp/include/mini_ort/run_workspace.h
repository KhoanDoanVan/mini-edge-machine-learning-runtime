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

            explicit RunWorkspace(
                Allocator& allocator = DefaultAllocator()
            ) noexcept;

            void Prepare(
                std::size_t batch_size,
                std::span<const std::size_t> slot_elements_per_batch
            );

            [[nodiscard]] TensorView Acquire(
                std::size_t slot,
                ShapeView shape
            );
            [[nodiscard]] ConstTensorView Get(std::size_t slot) const;
            [[nodiscard]] TensorView GetMutable(std::size_t slot);
            [[nodiscard]] std::size_t arena_layout_bytes() const noexcept;
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