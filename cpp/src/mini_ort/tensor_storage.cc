#include "mini_ort/tensor_storage.h"

#include <atomic>
#include <utility>


namespace mini_ort {

    namespace {

        std::atomic<std::uint64_t> buffer_allocations{0};
        std::atomic<std::uint64_t> allocated_bytes{0};
        std::atomic<std::uint64_t> tensor_copies{0};
        std::atomic<std::uint64_t> copied_bytes{0};
        std::atomic<std::uint64_t> live_bytes{0};
        std::atomic<std::uint64_t> peak_live_bytes{0};

        void UpdatePeakLiveBytes(
            const std::uint64_t candidate
        ) noexcept {

            auto peak = peak_live_bytes.load(std::memory_order_relaxed);

            while (
                peak < candidate 
                && !peak_live_bytes.compare_exchange_weak(
                    peak,
                    candidate,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed
                )
            )
            {}
            
        }

        void TrackBuffer(
            const std::uint64_t bytes,
            const bool is_copy
        ) noexcept {

            if (bytes == 0) { return; }

            buffer_allocations.fetch_add(
                1,
                std::memory_order_relaxed
            );

            allocated_bytes.fetch_add(
                bytes,
                std::memory_order_relaxed
            );

            if (is_copy) {
                tensor_copies.fetch_add(
                    1,
                    std::memory_order_relaxed
                );
                copied_bytes.fetch_add(
                    1,
                    std::memory_order_relaxed
                );
            }

            const auto current = live_bytes.fetch_add(
                bytes,
                std::memory_order_relaxed
            ) + bytes;

            UpdatePeakLiveBytes(current);

        }

        void ReleaseBuffer(
            const std::uint64_t bytes
        ) noexcept {
            if (bytes != 0) {
                live_bytes.fetch_sub(
                    bytes,
                    std::memory_order_relaxed
                );
            }
        }

        std::uint64_t BufferBytes(
            const std::vector<float>& data
        ) noexcept {
            return static_cast<std::uint64_t>(data.capacity()) * sizeof(float);
        }
    }

    void ResetTensorMemoryStats() noexcept {
        buffer_allocations.store(
            0,
            std::memory_order_relaxed
        );
        allocated_bytes.store(
            0,
            std::memory_order_relaxed
        );
        tensor_copies.store(
            0,
            std::memory_order_relaxed
        );
        copied_bytes.store(
            0,
            std::memory_order_relaxed
        );
        peak_live_bytes.store(
            live_bytes.load(std::memory_order_relaxed),
            std::memory_order_relaxed
        );
    }

    TensorMemoryStats GetTensorMemoryStats() noexcept {
        return TensorMemoryStats(
            buffer_allocations.load(std::memory_order_relaxed),
            allocated_bytes.load(std::memory_order_relaxed),
            tensor_copies.load(std::memory_order_relaxed),
            copied_bytes.load(std::memory_order_relaxed),
            live_bytes.load(std::memory_order_relaxed),
            peak_live_bytes.load(std::memory_order_relaxed)
        );
    }

    TensorStorage::TensorStorage(const std::size_t element_count) : data_(element_count), tracked_bytes_(BufferBytes(data_)) {
        TrackBuffer(
            tracked_bytes_,
            false
        );
    }

    TensorStorage::TensorStorage(std::vector<float> data) : data_(std::move(data)), tracked_bytes_(BufferBytes(data_)) {
        TrackBuffer(
            tracked_bytes_,
            false
        );
    }

    TensorStorage::TensorStorage(const TensorStorage& other) : data_(other.data_), tracked_bytes_(BufferBytes(data_)) {
        TrackBuffer(
            tracked_bytes_,
            true
        );
    }

    TensorStorage& TensorStorage::operator=(const TensorStorage& other) {
        if (this == &other) {
            return *this;
        }

        TensorStorage copy(other);
        data_.swap(copy.data_);
        std::swap(
            tracked_bytes_,
            copy.tracked_bytes_
        );

        return *this;
    }

    TensorStorage::TensorStorage(TensorStorage&& other) noexcept : data_(std::move(other.data_)), tracked_bytes_(std::exchange(other.tracked_bytes_, 0)) {}

    TensorStorage& TensorStorage::operator=(TensorStorage&& other) noexcept {

        if (this == &other) {
            return *this;
        }

        ReleaseBuffer(tracked_bytes_);
        data_ = std::move(other.data_);
        tracked_bytes_ = std::exchange(
            other.tracked_bytes_,
            0
        );

        return *this;

    }


    TensorStorage::~TensorStorage() { 
        ReleaseBuffer(tracked_bytes_);
    }

    std::size_t TensorStorage::size() const noexcept { return data_.size(); }

    bool TensorStorage::empty() const noexcept { return data_.empty(); }

    std::span<const float> TensorStorage::data() const noexcept { return data_; }

    std::span<float> TensorStorage::mutable_data() noexcept { return data_; }

}