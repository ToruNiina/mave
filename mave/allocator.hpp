#ifndef MAVE_ALLOCATOR_HPP
#define MAVE_ALLOCATOR_HPP
#include <type_traits>
#include <memory>
#include <limits>

#ifndef __GNUC__
#error "this file is for gcc or clang."
#endif

#include <cstdlib>

namespace mave
{

inline void* aligned_alloc(std::size_t alignment, std::size_t size)
{
    void *ptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

inline void aligned_free(void* ptr)
{
    std::free(ptr);
    return;
}

template<typename T, std::size_t Alignment = std::alignment_of<T>::value>
class aligned_allocator
{
  public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using propagate_on_container_move_assignment = std::true_type;

    template<typename U>
    struct rebind
    {
        using other = aligned_allocator<U, std::alignment_of<U>::value>;
    };

    static constexpr std::size_t alignment = Alignment;

  public:

    aligned_allocator() noexcept = default;
    aligned_allocator(const aligned_allocator&) noexcept = default;
    aligned_allocator& operator=(const aligned_allocator&) noexcept = default;

    template<typename U, std::size_t B>
    aligned_allocator(const aligned_allocator<U, B>&) noexcept {}
    template<typename U, std::size_t B>
    aligned_allocator& operator=(const aligned_allocator<U, B>&) noexcept
    {return *this;}

    pointer allocate(std::size_t n)
    {
        void* ptr = aligned_alloc(alignment, sizeof(T) * n);
        if(!ptr) {throw std::bad_alloc{};}
        return reinterpret_cast<pointer>(ptr);
    }
    void deallocate(pointer p, std::size_t)
    {
        aligned_free(p);
    }

    pointer       address(reference       x) const noexcept
    {return std::addressof(x);}
    const_pointer address(const_reference x) const noexcept
    {return std::addressof(x);}

    size_type max_size() const noexcept
    {
        return std::numeric_limits<size_type>::max() /
               std::max(sizeof(value_type), alignment);
    }

    void construct(pointer p, const_reference val)
    {
        new(reinterpret_cast<void*>(p)) value_type(val);
        return;
    }
    template <class U, class... Args>
    void construct(U* p, Args&&... args)
    {
        new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
        return;
    }

    void destroy(pointer p){p->~value_type(); return;}
    template <class U>
    void destroy(U* p){p->~U(); return;}

};
template<typename T, std::size_t A>
constexpr std::size_t aligned_allocator<T, A>::alignment;

template<typename T, std::size_t A>
inline bool
operator==(const aligned_allocator<T, A>&, const aligned_allocator<T, A>&)
{
    return true;
}
template<typename T, std::size_t A>
inline bool
operator!=(const aligned_allocator<T, A>&, const aligned_allocator<T, A>&)
{
    return false;
}

} // mave
#endif// MAVE_ALLOCATOR_HPP
