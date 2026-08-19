#include "mini_ort/tensor_storage.h"

#include <atomic>
#include <utility>
#include <limits>
#include <stdexcept>
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

        void TrackCopy(
            const std::uint64_t bytes
        ) noexcept {
            if (bytes == 0) {
                return;
            }

            tensor_copies.fetch_add(
                1,
                std::memory_order_relaxed
            );
            copied_bytes.fetch_add(
                bytes,
                std::memory_order_relaxed
            );
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
            // const std::vector<float>& data
            const std::size_t element_count
        ) noexcept {
            if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
                throw std::overflow_error("tensor storage byte size overflows size_t");
            }
            // return static_cast<std::uint64_t>(data.capacity()) * sizeof(float);
            return static_cast<std::uint64_t>(element_count) * sizeof(float);
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

    TensorStorage::TensorStorage(
        const std::size_t element_count,
        Allocator& allocator
    ) : allocator_(&allocator), 
        data_(nullptr),
        element_count_(element_count),
        tracked_bytes_(BufferBytes(element_count))
    {
        data_ = static_cast<float*>(allocator_->Allocate(
            tracked_bytes_,
            kTensorAlignment
        ));
        TrackBuffer(
            tracked_bytes_,
            false
        );
    }

    TensorStorage::TensorStorage(
        std::vector<float> data,
        Allocator& allocator
    ) : TensorStorage(
        data.size(),
        allocator
    ) {
        if (!data.empty()) {
            std::copy(
                data.begin(),
                data.end(),
                data_
            );
            TrackCopy(
                tracked_bytes_
            );
        }
        // TrackBuffer(
        //     tracked_bytes_,
        //     false
        // );
    }

    TensorStorage::TensorStorage(const TensorStorage& other) : TensorStorage(
        other.element_count_,
        *other.allocator_ 
    ) {
        if (!other.empty()) {
            std::copy(
                other.data_,
                other.data_ + other.element_count_,
                data_
            );
            TrackCopy(
                tracked_bytes_
            );
        }
        // TrackBuffer(
        //     tracked_bytes_,
        //     true
        // );
    }

    TensorStorage& TensorStorage::operator=(const TensorStorage& other) {
        if (this == &other) {
            return *this;
        }

        TensorStorage copy(other);
        std::swap(
            allocator_,
            copy.allocator_
        );
        std::swap(
            data_,
            copy.data_
        );
        std::swap(
            element_count_,
            copy.element_count_
        );
        // data_.swap(copy.data_);
        std::swap(
            tracked_bytes_,
            copy.tracked_bytes_
        );

        return *this;
    }

    TensorStorage::TensorStorage(TensorStorage&& other) noexcept : allocator_(std::exchange(other.allocator_, &DefaultAllocator())), data_(std::exchange(other.data_, nullptr)), element_count_(std::exchange(other.element_count_, 0)), tracked_bytes_(std::exchange(other.tracked_bytes_, 0)) {}

    TensorStorage& TensorStorage::operator=(TensorStorage&& other) noexcept {

        if (this == &other) {
            return *this;
        }

        // ReleaseBuffer(tracked_bytes_);
        Release();
        allocator_ = std::exchange(
            other.allocator_,
            &DefaultAllocator()
        );
        // data_ = std::move(other.data_);
        data_ = std::exchange(
            other.data_,
            nullptr
        );

        element_count_ = std::exchange(
            other.element_count_,
            0
        );
        tracked_bytes_ = std::exchange(
            other.tracked_bytes_,
            0
        );

        return *this;

    }


    TensorStorage::~TensorStorage() { 
        // ReleaseBuffer(tracked_bytes_);
        Release();
    }

    void TensorStorage::Release() noexcept {
        if (allocator_ != nullptr) {
            allocator_->Deallocate(data_);
        }

        ReleaseBuffer(tracked_bytes_);
        allocator_ = nullptr;
        data_ = nullptr;
        element_count_ = 0;
        tracked_bytes_ = 0;
    }

    std::size_t TensorStorage::size() const noexcept { return element_count_; }

    bool TensorStorage::empty() const noexcept { return element_count_ == 0; }

    std::span<const float> TensorStorage::data() const noexcept { return {data_, element_count_}; }

    std::span<float> TensorStorage::mutable_data() noexcept { return {data_, element_count_}; }

}