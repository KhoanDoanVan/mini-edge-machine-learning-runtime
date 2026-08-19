#include "mini_ort/allocator.h"

#include <bit>
#include <cstdlib>
#include <new>
#include <stdexcept>

#if defined(_WIN32)
#include <malloc.h>
#endif


namespace mini_ort {

    void* HeapAllocator::Allocate(
        const std::size_t bytes,
        const std::size_t alignment
    ) {

        if (bytes == 0) {
            return nullptr;
        }

        if (alignment < sizeof(void*) || !std::has_single_bit(alignment)) {
            throw std::invalid_argument("allocation alignment must be a power of two and pointer-aligned");
        }


#if defined(_WIN32)
        void* pointer = _aligned_malloc(bytes, alignment);

        if (pointer == nullptr) {
            throw std::bad_alloc();
        }

        return pointer;
#else
        void* pointer = nullptr;
        // posix_memalign on macOS/Linux
        if (posix_memalign(&pointer, alignment, bytes) != 0) {
            throw std::bad_alloc();
        }

        return pointer;
#endif
    }


    void HeapAllocator::Deallocate(void* pointer) noexcept {
#if defined(_WIN32)
        _aligned_free(pointer);
#else
        std::free(pointer);
#endif
    }


    Allocator& DefaultAllocator() noexcept {
        static HeapAllocator allocator;
        return allocator;
    }

}