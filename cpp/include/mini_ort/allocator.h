#pragma once

#include <cstddef>

namespace mini_ort {

    inline constexpr std::size_t kTensorAlignment = 64;

    class Allocator {

        public:

            virtual ~Allocator() = default;

            [[nodiscard]] virtual void* Allocate(
                std::size_t byte,
                std::size_t alignment
            ) = 0;

            virtual void Deallocate(void* pointer) noexcept = 0;

    };

    class HeapAllocator final : public Allocator {

        public:

            [[nodiscard]] void* Allocate(
                std::size_t bytes,
                std::size_t alignment
            ) override;

            void Deallocate(void* pointer) noexcept override;

    };

    [[nodiscard]] Allocator& DefaultAllocator() noexcept;
    
}