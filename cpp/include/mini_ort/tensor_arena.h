#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "mini_ort/allocator.h"
#include "mini_ort/tensor_storage.h"


// Owns one aligned allocation and divides it into non-overlapping regions.
// MemoryPlan decides which activaitons may share a region
// TensorArena only turns the resulting physical slot sizes into aligned offsets.
class TensorArena final {

    public:
        explicit TensorArena(Allocator& allocator = DefaultAllocator()) noexcept;

        void Configure(std::span<const std::size_t> slot_elements);

        [[nodiscard]] std::span<float> Slice(
            std::size_t slot,
            std::size_t elements
        );
        [[nodiscard]] std::span<const float> Slice(
            std::size_t slot,
            std::size_t elements
        ) const;

        [[nodiscard]] std::size_t slot_count() const noexcept;
        [[nodiscard]] std::size_t slot_offset_elements(std::size_t slot) const;
        [[nodiscard]] std::size_t slot_capacity_elements(std::size_t slot) const;

        [[nodiscard]] std::size_t layout_bytes() const noexcept;
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

}