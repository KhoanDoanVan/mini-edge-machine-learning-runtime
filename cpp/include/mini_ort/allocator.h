#pragma once

#include <cstddef>

namespace mini_ort {

    inline constexpr std::size_t kTensorAlignment = 64;

    class Allocator {

        public:

            /**
             * @brief Destroy the allocator through its polymorphic interface.
             */
            virtual ~Allocator() = default;

            /**
             * @brief Allocate an aligned raw memory block.
             *
             * The returned block is owned by the caller and must be released
             * through @ref Deallocate using the same allocator instance.
             *
             * @param byte      Number of bytes to allocate.
             * @param alignment Required alignment in bytes.
             * @return Pointer to the allocated block; never null on success.
             * @throws std::bad_alloc when the request cannot be satisfied.
             */
            [[nodiscard]] virtual void* Allocate(
                std::size_t byte,
                std::size_t alignment
            ) = 0;

            /**
             * @brief Release a block previously returned by @ref Allocate.
             *
             * Passing a null pointer is permitted and has no effect.
             *
             * @param pointer Block to release, or nullptr.
             */
            virtual void Deallocate(void* pointer) noexcept = 0;

    };

    class HeapAllocator final : public Allocator {

        public:

            /**
             * @brief Allocate aligned memory from the host heap.
             *
             * This implementation is intended for desktop execution. An
             * embedded target can provide a different @ref Allocator without
             * changing tensor or execution-plan code.
             *
             * @param bytes     Number of bytes to allocate.
             * @param alignment Required alignment in bytes.
             * @return Pointer to the allocated block.
             * @throws std::bad_alloc when the allocation fails.
             */
            [[nodiscard]] void* Allocate(
                std::size_t bytes,
                std::size_t alignment
            ) override;

            /**
             * @brief Release memory obtained from the host heap.
             *
             * @param pointer Pointer returned by this allocator, or nullptr.
             */
            void Deallocate(void* pointer) noexcept override;

    };

    /**
     * @brief Return the process-wide default allocator.
     *
     * The returned reference remains valid for the lifetime of the process
     * and is used by tensor classes when no allocator is explicitly supplied.
     *
     * @return Reference to the shared default allocator.
     */
    [[nodiscard]] Allocator& DefaultAllocator() noexcept;
    
}
