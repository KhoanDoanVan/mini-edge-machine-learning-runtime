#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "mini_ort/allocator.h"
#include "mini_ort/tensor_storage.h"

namespace mini_ort {
// Owns one aligned allocation and divides it into non-overlapping regions.
// MemoryPlan decides which activaitons may share a region
// TensorArena only turns the resulting physical slot sizes into aligned offsets.
    class TensorArena final {

        public:
            /**
             * @brief Construct an arena backed by the supplied allocator.
             * @param allocator Allocator retained for the arena lifetime.
             */
            explicit TensorArena(Allocator& allocator = DefaultAllocator()) noexcept;

            /**
             * @brief Configure physical slots before any slice is requested.
             * @param slot_elements Maximum elements required by each slot.
             */
            void Configure(std::span<const std::size_t> slot_elements);

            /** @brief Return a writable slice for a configured slot. */
            [[nodiscard]] std::span<float> Slice(
                std::size_t slot,
                std::size_t elements
            );
            /** @brief Return a read-only slice for a configured slot. */
            [[nodiscard]] std::span<const float> Slice(
                std::size_t slot,
                std::size_t elements
            ) const;

            /** @brief Return the number of configured slots. */
            [[nodiscard]] std::size_t slot_count() const noexcept;
            /** @brief Return a slot's element offset within the arena. */
            [[nodiscard]] std::size_t slot_offset_elements(std::size_t slot) const;
            /** @brief Return a slot's configured element capacity. */
            [[nodiscard]] std::size_t slot_capacity_elements(std::size_t slot) const;

            /** @brief Return bytes required by the configured layout. */
            [[nodiscard]] std::size_t layout_bytes() const noexcept;
            /** @brief Return bytes currently available in the backing storage. */
            [[nodiscard]] std::size_t capacity_bytes() const noexcept;

        private:
            
            struct Region final {
                std::size_t offset_elements;
                std::size_t capacity_elements;
            };

            Allocator* allocator_;
            std::vector<Region> regions_;
            std::optional<TensorStorage> storage_;
            std::size_t layout_elements_ = 0;
    };
}
