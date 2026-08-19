#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "mini_ort/allocator.h"
#include "mini_ort/tensor_storage.h"
#include "mini_ort/tensor_view.h"


namespace mini_ort {

    class RunWorkspace final {

        public:

            static constexpr std::size_t kSlotCount = 2;

            explicit RunWorkspace(
                Allocator& allocator = DefaultAllocator()
            ) noexcept;

            [[nodiscard]] TensorView Acquire(
                std::size_t slot,
                ShapeView shape
            );

        private:

            struct Slot final {
                Shape shape;
                std::optional<TensorStorage> storage;
            };

            Allocator* allocator_;
            std::array<Slot, kSlotCount> slots_;

    };

}