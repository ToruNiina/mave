#ifndef MAVE_AVX2_MATRIX_3x3_FLOAT_HPP
#define MAVE_AVX2_MATRIX_3x3_FLOAT_HPP

#ifndef __AVX2__
#error "mave/avx2/matrix3x3f.hpp requires avx support but __AVX2__ is not defined."
#endif

#ifndef MAVE_MATRIX_HPP
#error "do not use mave/avx/matrix3x3f.hpp alone. please include mave/mave.hpp."
#endif

#ifdef MAVE_MATRIX_3X3_FLOAT_IMPLEMENTATION
#error "specialization of vector for 3x3 float is already defined"
#endif

#define MAVE_MATRIX_3X3_FLOAT_IMPLEMENTATION "avx2"

#include <x86intrin.h> // for *nix
#include <type_traits>
#include <array>
#include <cmath>

// for readability...
#define _mm128_load_ps  _mm_load_ps
#define _mm128_store_ps _mm_store_ps
#define _mm128_set1_ps  _mm_set1_ps
#define _mm128_add_ps   _mm_add_ps
#define _mm128_sub_ps   _mm_sub_ps
#define _mm128_mul_ps   _mm_mul_ps
#define _mm128_div_ps   _mm_div_ps
#define _mm128_max_ps   _mm_max_ps
#define _mm128_min_ps   _mm_min_ps
#define _mm128_floor_ps _mm_floor_ps
#define _mm128_ceil_ps  _mm_ceil_ps

namespace mave
{

template<>
struct alignas(32) matrix<float, 3, 3>
{
    static constexpr std::size_t row_size    = 3;
    static constexpr std::size_t column_size = 3;
    static constexpr std::size_t total_size  = 9;
    using value_type      = float;
    using storage_type    = std::array<float, 12>;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;

    matrix(float v00, float v01, float v02,
           float v10, float v11, float v12,
           float v20, float v21, float v22)
        : vs_{{v00, v01, v02, 0.0f, v10, v11, v12, 0.0f, v20, v21, v22, 0.0f}}
    {}

    matrix(){vs_.fill(0.0f);}
    ~matrix() = default;
    matrix(const matrix&) = default;
    matrix(matrix&&)      = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&)      = default;

    template<typename T>
    matrix& operator=(const matrix<T, 3, 3>& rhs) noexcept
    {
        vs_[ 0] = static_cast<float>(rhs(0, 0));
        vs_[ 1] = static_cast<float>(rhs(0, 1));
        vs_[ 2] = static_cast<float>(rhs(0, 2));
        vs_[ 3] = 0.0f;
        vs_[ 4] = static_cast<float>(rhs(1, 0));
        vs_[ 5] = static_cast<float>(rhs(1, 1));
        vs_[ 6] = static_cast<float>(rhs(1, 2));
        vs_[ 7] = 0.0f;
        vs_[ 8] = static_cast<float>(rhs(2, 0));
        vs_[ 9] = static_cast<float>(rhs(2, 1));
        vs_[10] = static_cast<float>(rhs(2, 2));
        vs_[11] = 0.0f;
        return *this;
    }

    matrix& operator+=(const matrix<float, 3, 3>& other) noexcept
    {
        pointer       self = this->data();
        const_pointer othr = other.data();
        _mm256_store_ps(self,   _mm256_add_ps(_mm256_load_ps(self  ), _mm256_load_ps(othr  )));
        _mm128_store_ps(self+8, _mm128_add_ps(_mm128_load_ps(self+8), _mm128_load_ps(othr+8)));
        return *this;
    }
    matrix& operator-=(const matrix<float, 3, 3>& other) noexcept
    {
        pointer       self = this->data();
        const_pointer othr = other.data();
        _mm256_store_ps(self,   _mm256_sub_ps(_mm256_load_ps(self  ), _mm256_load_ps(othr  )));
        _mm128_store_ps(self+8, _mm128_sub_ps(_mm128_load_ps(self+8), _mm128_load_ps(othr+8)));
        return *this;
    }
    matrix& operator*=(const float other) noexcept
    {
        pointer self = this->data();
        _mm256_store_ps(self,   _mm256_mul_ps(_mm256_load_ps(self  ), _mm256_set1_ps(other)));
        _mm128_store_ps(self+8, _mm128_mul_ps(_mm128_load_ps(self+8), _mm128_set1_ps(other)));
        return *this;
    }
    matrix& operator/=(const float other) noexcept
    {
        pointer       self = this->data();
        _mm256_store_ps(self,   _mm256_div_ps(_mm256_load_ps(self  ), _mm256_set1_ps(other)));
        _mm128_store_ps(self+8, _mm128_div_ps(_mm128_load_ps(self+8), _mm128_set1_ps(other)));
        return *this;
    }

    size_type size() const noexcept {return total_size;}

    pointer       data()       noexcept {return vs_.data();}
    const_pointer data() const noexcept {return vs_.data();}

    // i = |00|01|02|xx|03|04|05|xx|06|07|08|xx|
    // idx:|00|01|02|03|04|05|06|07|08|09|10|11|
    reference               at(size_type i)       {return vs_.at(i/3+i);}
    const_reference         at(size_type i) const {return vs_.at(i/3+i);}
    reference       operator[](size_type i)       noexcept {return vs_[i/3+i];}
    const_reference operator[](size_type i) const noexcept {return vs_[i/3+i];}

    // i = |          0|          1|          2|
    // j = |00|01|02|xx|01|02|03|xx|01|02|03|xx|
    // idx:|00|01|02|03|04|05|06|07|08|09|10|11|
    reference       at(size_type i, size_type j)       {return vs_.at(i*4+j);}
    const_reference at(size_type i, size_type j) const {return vs_.at(i*4+j);}
    reference       operator()(size_type i, size_type j)       noexcept {return vs_[i*4+j];}
    const_reference operator()(size_type i, size_type j) const noexcept {return vs_[i*4+j];}

  private:
    alignas(32) storage_type vs_;
};

template<>
inline matrix<double, 3, 3> operator-(const matrix<double, 3, 3>& m) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_m = m.data();
    _mm256_store_ps(ptr_r,   _mm256_sub_ps(_mm256_setzero_ps(), _mm256_load_ps(ptr_m  )));
    _mm128_store_ps(ptr_r+8, _mm128_sub_ps(_mm128_setzero_ps(), _mm128_load_ps(ptr_m+8)));
    return retval;
}
template<>
inline matrix<double, 3, 3> operator+(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_add_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_add_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}
template<>
inline matrix<double, 3, 3> operator-(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_sub_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_sub_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}
template<>
inline matrix<double, 3, 3> operator*(
    const double s1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_mul_ps(_mm256_set1_ps(s1), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_mul_ps(_mm128_set1_ps(s1), _mm128_load_ps(ptr_2+8)));
    return retval;
}
template<>
inline matrix<double, 3, 3> operator*(
    const matrix<double, 3, 3>& m1, const double s2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    _mm256_store_ps(ptr_r,   _mm256_mul_ps(_mm256_load_ps(ptr_1  ), _mm256_set1_ps(s2)));
    _mm128_store_ps(ptr_r+8, _mm128_mul_ps(_mm128_load_ps(ptr_1+8), _mm128_set1_ps(s2)));
    return retval;
}
template<>
inline matrix<double, 3, 3> operator/(
    const matrix<double, 3, 3>& m1, const double s2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    _mm256_store_ps(ptr_r,   _mm256_div_ps(_mm256_load_ps(ptr_1  ), _mm256_set1_ps(s2)));
    _mm128_store_ps(ptr_r+8, _mm128_div_ps(_mm128_load_ps(ptr_1+8), _mm128_set1_ps(s2)));
    return retval;
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<>
inline matrix<double, 3, 3> max(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_max_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_max_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}

template<>
inline matrix<double, 3, 3> min(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_min_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_min_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}

// floor ---------------------------------------------------------------------

template<>
inline matrix<double, 3, 3> floor(const matrix<double, 3, 3>& m) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m.data();
    _mm256_store_ps(ptr_r,   _mm256_floor_ps(_mm256_load_ps(ptr_1  )));
    _mm128_store_ps(ptr_r+8, _mm128_floor_ps(_mm128_load_ps(ptr_1+8)));
    return retval;
}

// ceil ----------------------------------------------------------------------

template<>
inline matrix<double, 3, 3> ceil(const matrix<double, 3, 3>& m) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m.data();
    _mm256_store_ps(ptr_r,   _mm256_ceil_ps(_mm256_load_ps(ptr_1  )));
    _mm128_store_ps(ptr_r+8, _mm128_ceil_ps(_mm128_load_ps(ptr_1+8)));
    return retval;
}

// ---------------------------------------------------------------------------

} // mave

#undef _mm128_load_ps
#undef _mm128_store_ps
#undef _mm128_set1_ps
#undef _mm128_add_ps
#undef _mm128_sub_ps
#undef _mm128_mul_ps
#undef _mm128_div_ps
#undef _mm128_max_ps
#undef _mm128_min_ps
#undef _mm128_floor_ps
#undef _mm128_ceil_ps

#endif // MAVE_AVX2_MATRIX_3x3_DOUBLE_HPP
