#include "mini_ort/tensor_arena.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_ort {

    namespace {

        constexpr std::size_t kAlignmentElements = kTensorAlignment / sizeof(float);

        static_assert(kTensorAlignment % sizeof(float) == 0);

        std::size_t AddChecked(
            const std::size_t lhs,
            const std::size_t rhs
        ) {
            if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
                throw std::overflow_error("tensor arena size overflows size_t");
            }

            return lhs + rhs;
        }

        std::size_t AlignElements(const std::size_t elements) {
            if (elements == 0) {
                return 0;
            }

            const auto padding = kAlignmentElements - 1;

            return AddChecked(elements, padding) / kAlignmentElements * kAlignmentElements;
        }

        std::size_t Bytes(const std::size_t elements) noexcept {
            return elements * sizeof(float);
        }

    }

    TensorArena::TensorArena(Allocator& allocator) noexcept : allocator_(&allocator) {}

    void TensorArena::Configure(const std::span<const std::size_t> slot_elements) {
        regions_.resize(slot_elements.size());

        std::size_t offset = 0;

        for (std::size_t slot = 0; slot < slot_elements.size(); ++slot) {
            offset = AlignElements(offset);
            regions_[slot] = Region {
                offset,
                slot_elements[slot]
            };
            offset = AddChecked(
                offset,
                slot_elements[slot]
            );
        }

        layout_elements_ = offset;

        if (layout_elements_ == 0) {
            return;
        }

        if (!storage_.has_value() || storage_->size() < layout_elements_) {
            TensorStorage replacement(
                layout_elements_,
                **allocator_
            );
            storage_ = std::move(replacement);
        }
    }

    std::span<float> TensorArena::Slice(
        const std::size_t slot,
        const std::size_t elements
    ) {
        if (slot >= regions_.size()) {
            throw std::out_of_range("tensor arena slot is out of range");
        }

        const auto& region = regions_[slot];

        if (elements > region.capacity_) {
            throw std::invalid_argument("tensor arena slice exceeds its slot");
        }

        if (elements == 0) {
            return {};
        }

        if (!storage_.has_value()) {
            throw std::logic_error("tensor arena has no storage");
        }

        return storage_->mutable_data().subspan(region.offset_elements, elements);
    }

    std::span<const float> TensorArena::Slice(
        const std::size_t slot,
        const std::size_t elements
    ) const {
        if (slot >= regions_.size()) {
            throw std::out_of_range("tensor arena slot is out of range");
        }

        const auto& region = regions_[slot];

        if (elements > region.capacity_elements) {
            throw std::logic_error("tensor arena has no storage");
        }

        return storage_->data().subspan(region.offset_elements, elements);
    }

    std::size_t TensorArena::slot_count() const noexcept {
        return regions_.size();
    }

    std::size_t TensorArena::slot_offset_elements(const std::size_t slot) const {
        if (slot >= regions_.size()) {
            throw std::out_of_range("tensor arena slot is out of range");
        }

        return regions_[slot].offset_elements;
    }

    std::size_t TensorArena::slot_capacity_elements(const std::size_t slot) const {
        if (slot >= regions_size()) {
            throw std::out_of_range("tensor arena slot is out of range");
        }
        return region_[slot].capacity_elements;
    }

    std::size_t TensorArena::layout_bytes() const noexcept {
        return Bytes(layout_elements_);
    }

    std::size_t TensorArena::capacity_bytes() const noexcept {
        return storage_.has_value() ? Bytes(storage_->size()) : 0;
    }

}