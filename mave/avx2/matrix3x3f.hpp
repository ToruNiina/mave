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
#define _mm128_load_ps    _mm_load_ps
#define _mm128_store_ps   _mm_store_ps
#define _mm128_set1_ps    _mm_set1_ps
#define _mm128_setzero_ps _mm_setzero_ps
#define _mm128_add_ps     _mm_add_ps
#define _mm128_sub_ps     _mm_sub_ps
#define _mm128_mul_ps     _mm_mul_ps
#define _mm128_div_ps     _mm_div_ps
#define _mm128_max_ps     _mm_max_ps
#define _mm128_min_ps     _mm_min_ps
#define _mm128_floor_ps   _mm_floor_ps
#define _mm128_ceil_ps    _mm_ceil_ps

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
           float v20, float v21, float v22) noexcept
        : vs_{{v00, v01, v02, 0.0f, v10, v11, v12, 0.0f, v20, v21, v22, 0.0f}}
    {}

    matrix(const std::array<float, 9>& arg) noexcept
        : vs_{{arg[0], arg[1], arg[2], 0.0f,
               arg[3], arg[4], arg[5], 0.0f,
               arg[6], arg[7], arg[8], 0.0f}}
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

    bool diagnosis() const noexcept
    {return (vs_[3]==0.0f) && (vs_[7]==0.0f) && (vs_[11]==0.0f);}

    void zero() noexcept
    {
        pointer self = this->data();
        _mm256_store_ps(self,   _mm256_setzero_ps());
        _mm128_store_ps(self+8, _mm128_setzero_ps());
        return;
    }

  private:
    alignas(32) storage_type vs_;
};

template<>
MAVE_INLINE matrix<float, 3, 3> operator-(const matrix<float, 3, 3>& m) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m = m.data();
    _mm256_store_ps(ptr_r,   _mm256_sub_ps(_mm256_setzero_ps(), _mm256_load_ps(ptr_m  )));
    _mm128_store_ps(ptr_r+8, _mm128_sub_ps(_mm128_setzero_ps(), _mm128_load_ps(ptr_m+8)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
operator-(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> ms) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m1 = std::get<0>(ms).data();
    typename matrix<float, 3, 3>::const_pointer ptr_m2 = std::get<1>(ms).data();

    _mm256_store_ps(ptr_r1, _mm256_sub_ps(_mm256_setzero_ps(), _mm256_load_ps(ptr_m1)));
    _mm256_store_ps(ptr_r2, _mm256_sub_ps(_mm256_setzero_ps(), _mm256_load_ps(ptr_m2)));

    const __m256 m12 = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(ptr_m1+8)), _mm_load_ps(ptr_m2+8), 1));

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(m12));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(m12, 1));

    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>, matrix<float, 3, 3>>
operator-(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> ms) noexcept
{
    const auto m12 = -std::tie(std::get<0>(ms), std::get<1>(ms));
    return std::make_tuple(std::get<0>(m12), std::get<1>(m12), -std::get<2>(ms));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                  matrix<float, 3, 3>, matrix<float, 3, 3>>
operator-(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> ms) noexcept
{
    const auto m12 = -std::tie(std::get<0>(ms), std::get<1>(ms));
    const auto m34 = -std::tie(std::get<2>(ms), std::get<3>(ms));
    return std::make_tuple(std::get<0>(m12), std::get<1>(m12),
                            std::get<0>(m34), std::get<1>(m34));
}

// ---------------------------------------------------------------------------
// addition operator+
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> operator+(
    const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<float, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_add_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_add_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
operator+(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> rhs
          ) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_ml1 = std::get<0>(lhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_ml2 = std::get<1>(lhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_mr1 = std::get<0>(rhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_mr2 = std::get<1>(rhs).data();

    _mm256_store_ps(ptr_r1, _mm256_add_ps(_mm256_load_ps(ptr_ml1), _mm256_load_ps(ptr_mr1)));
    _mm256_store_ps(ptr_r2, _mm256_add_ps(_mm256_load_ps(ptr_ml2), _mm256_load_ps(ptr_mr2)));

    const __m256 mid = _mm256_add_ps(
        _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ptr_ml1+8)), _mm_load_ps(ptr_ml2+8), 1),
        _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ptr_mr1+8)), _mm_load_ps(ptr_mr2+8), 1));

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(mid));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(mid, 1));
    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>, matrix<float, 3, 3>>
operator+(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> rhs) noexcept
{
    const auto m12 = std::tie(std::get<0>(lhs), std::get<1>(lhs)) +
                     std::tie(std::get<0>(rhs), std::get<1>(rhs));
    return std::make_tuple(std::get<0>(m12),  std::get<1>(m12),
                           std::get<2>(lhs) + std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                  matrix<float, 3, 3>, matrix<float, 3, 3>>
operator+(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> rhs
          ) noexcept
{
    const auto m12 = std::tie(std::get<0>(lhs), std::get<1>(lhs)) +
                     std::tie(std::get<0>(rhs), std::get<1>(rhs));
    const auto m34 = std::tie(std::get<2>(lhs), std::get<3>(lhs)) +
                     std::tie(std::get<2>(rhs), std::get<3>(rhs));

    return std::make_tuple(std::get<0>(m12), std::get<1>(m12),
                           std::get<0>(m34), std::get<1>(m34));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<float, 3, 3>&,       matrix<float, 3, 3>&> v1,
    std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> v2
    ) noexcept
{
    std::get<0>(v1) += std::get<0>(v2);
    std::get<1>(v1) += std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<float, 3, 3>&, matrix<float, 3, 3>&,
                     matrix<float, 3, 3>&> v1,
    std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
               const matrix<float, 3, 3>&> v2) noexcept
{
    std::get<0>(v1) += std::get<0>(v2);
    std::get<1>(v1) += std::get<1>(v2);
    std::get<2>(v1) += std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<float, 3, 3>&,       matrix<float, 3, 3>&,
                     matrix<float, 3, 3>&,       matrix<float, 3, 3>&> v1,
    std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
               const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> v2
    ) noexcept
{
    std::get<0>(v1) += std::get<0>(v2);
    std::get<1>(v1) += std::get<1>(v2);
    std::get<2>(v1) += std::get<2>(v2);
    std::get<3>(v1) += std::get<3>(v2);
    return ;
}

// ---------------------------------------------------------------------------
// subtraction operator-
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> operator-(
    const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<float, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_sub_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_sub_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
operator-(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> rhs
          ) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_ml1 = std::get<0>(lhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_ml2 = std::get<1>(lhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_mr1 = std::get<0>(rhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_mr2 = std::get<1>(rhs).data();

    _mm256_store_ps(ptr_r1, _mm256_sub_ps(_mm256_load_ps(ptr_ml1), _mm256_load_ps(ptr_mr1)));
    _mm256_store_ps(ptr_r2, _mm256_sub_ps(_mm256_load_ps(ptr_ml2), _mm256_load_ps(ptr_mr2)));

    const __m256 mid = _mm256_sub_ps(
        _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ptr_ml1+8)), _mm_load_ps(ptr_ml2+8), 1),
        _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(ptr_mr1+8)), _mm_load_ps(ptr_mr2+8), 1));

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(mid));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(mid, 1));
    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>, matrix<float, 3, 3>>
operator-(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> rhs) noexcept
{
    const auto m12 = std::tie(std::get<0>(lhs), std::get<1>(lhs)) -
                     std::tie(std::get<0>(rhs), std::get<1>(rhs));
    return std::make_tuple(std::get<0>(m12),  std::get<1>(m12),
                           std::get<2>(lhs) - std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                  matrix<float, 3, 3>, matrix<float, 3, 3>>
operator-(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> rhs
          ) noexcept
{
    const auto m12 = std::tie(std::get<0>(lhs), std::get<1>(lhs)) -
                     std::tie(std::get<0>(rhs), std::get<1>(rhs));
    const auto m34 = std::tie(std::get<2>(lhs), std::get<3>(lhs)) -
                     std::tie(std::get<2>(rhs), std::get<3>(rhs));

    return std::make_tuple(std::get<0>(m12), std::get<1>(m12),
                           std::get<0>(m34), std::get<1>(m34));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<float, 3, 3>&,       matrix<float, 3, 3>&> v1,
    std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> v2
    ) noexcept
{
    std::get<0>(v1) -= std::get<0>(v2);
    std::get<1>(v1) -= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<float, 3, 3>&, matrix<float, 3, 3>&,
                     matrix<float, 3, 3>&> v1,
    std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
               const matrix<float, 3, 3>&> v2) noexcept
{
    std::get<0>(v1) -= std::get<0>(v2);
    std::get<1>(v1) -= std::get<1>(v2);
    std::get<2>(v1) -= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<float, 3, 3>&,       matrix<float, 3, 3>&,
                     matrix<float, 3, 3>&,       matrix<float, 3, 3>&> v1,
    std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
               const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> v2
    ) noexcept
{
    std::get<0>(v1) -= std::get<0>(v2);
    std::get<1>(v1) -= std::get<1>(v2);
    std::get<2>(v1) -= std::get<2>(v2);
    std::get<3>(v1) -= std::get<3>(v2);
    return ;
}

// ---------------------------------------------------------------------------
// multiplication operator*
// ---------------------------------------------------------------------------

// matrix * scalar -----------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> operator*(
    const matrix<float, 3, 3>& m1, const float s2) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m1.data();
    _mm256_store_ps(ptr_r,   _mm256_mul_ps(_mm256_load_ps(ptr_1  ), _mm256_set1_ps(s2)));
    _mm128_store_ps(ptr_r+8, _mm128_mul_ps(_mm128_load_ps(ptr_1+8), _mm128_set1_ps(s2)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
operator*(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<float, float> rhs) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m1 = std::get<0>(lhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_m2 = std::get<1>(lhs).data();

    _mm256_store_ps(ptr_r1, _mm256_mul_ps(_mm256_load_ps(ptr_m1), _mm256_set1_ps(std::get<0>(rhs))));
    _mm256_store_ps(ptr_r2, _mm256_mul_ps(_mm256_load_ps(ptr_m2), _mm256_set1_ps(std::get<1>(rhs))));

    const __m256 mid = _mm256_mul_ps(
        _mm256_insertf128_ps(_mm256_castps128_ps256(
            _mm_load_ps(ptr_m1+8)),         _mm_load_ps(ptr_m2+8),         1),
        _mm256_insertf128_ps(_mm256_castps128_ps256(
            _mm_set1_ps(std::get<0>(rhs))), _mm_set1_ps(std::get<1>(rhs)), 1));

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(mid));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(mid, 1));
    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>, matrix<float, 3, 3>>
operator*(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> lhs,
          std::tuple<float, float, float> rhs) noexcept
{
    const auto m12 =    std::tie(std::get<0>(lhs), std::get<1>(lhs)) *
        std::tuple<float, float>(std::get<0>(rhs), std::get<1>(rhs));
    return std::make_tuple(std::get<0>(m12),  std::get<1>(m12),
                           std::get<2>(lhs) * std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                  matrix<float, 3, 3>, matrix<float, 3, 3>>
operator*(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<float, float, float, float> rhs) noexcept
{
    const auto m12 =    std::tie(std::get<0>(lhs), std::get<1>(lhs)) *
        std::tuple<float, float>(std::get<0>(rhs), std::get<1>(rhs));
    const auto m34 =    std::tie(std::get<2>(lhs), std::get<3>(lhs)) *
        std::tuple<float, float>(std::get<2>(rhs), std::get<3>(rhs));

    return std::make_tuple(std::get<0>(m12), std::get<1>(m12),
                           std::get<0>(m34), std::get<1>(m34));
}

// scalar * matrix -----------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> operator*(
    const float s1, const matrix<float, 3, 3>& m2) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_mul_ps(_mm256_set1_ps(s1), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_mul_ps(_mm128_set1_ps(s1), _mm128_load_ps(ptr_2+8)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
operator*(std::tuple<float, float> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> rhs
          ) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m1 = std::get<0>(rhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_m2 = std::get<1>(rhs).data();

    _mm256_store_ps(ptr_r1, _mm256_mul_ps(_mm256_set1_ps(std::get<0>(lhs)), _mm256_load_ps(ptr_m1)));
    _mm256_store_ps(ptr_r2, _mm256_mul_ps(_mm256_set1_ps(std::get<1>(lhs)), _mm256_load_ps(ptr_m2)));

    const __m256 mid = _mm256_mul_ps(
        _mm256_insertf128_ps(_mm256_castps128_ps256(
            _mm_set1_ps(std::get<0>(lhs))), _mm_set1_ps(std::get<1>(lhs)), 1),
        _mm256_insertf128_ps(_mm256_castps128_ps256(
            _mm_load_ps(ptr_m1+8)),         _mm_load_ps(ptr_m2+8),         1)
        );

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(mid));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(mid, 1));
    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>, matrix<float, 3, 3>>
operator*(std::tuple<float, float, float> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> rhs) noexcept
{
    const auto m12 =
        std::tuple<float, float>(std::get<0>(lhs), std::get<1>(lhs)) *
        std::tie(std::get<0>(rhs), std::get<1>(rhs));
    return std::make_tuple(std::get<0>(m12),  std::get<1>(m12),
                           std::get<2>(lhs) * std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                  matrix<float, 3, 3>, matrix<float, 3, 3>>
operator*(std::tuple<float, float, float, float> lhs,
          std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> rhs
          ) noexcept
{
    const auto m12 =
        std::tuple<float, float>(std::get<0>(lhs), std::get<1>(lhs)) *
        std::tie(std::get<0>(rhs), std::get<1>(rhs));
    const auto m34 =
        std::tuple<float, float>(std::get<2>(lhs), std::get<3>(lhs)) *
        std::tie(std::get<2>(rhs), std::get<3>(rhs));

    return std::make_tuple(std::get<0>(m12), std::get<1>(m12),
                           std::get<0>(m34), std::get<1>(m34));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<float, 3, 3>&, matrix<float, 3, 3>&> v1,
    std::tuple<float, float> v2) noexcept
{
    std::get<0>(v1) *= std::get<0>(v2);
    std::get<1>(v1) *= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<float, 3, 3>&, matrix<float, 3, 3>&, matrix<float, 3, 3>&> v1,
    std::tuple<float, float, float> v2) noexcept
{
    std::get<0>(v1) *= std::get<0>(v2);
    std::get<1>(v1) *= std::get<1>(v2);
    std::get<2>(v1) *= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<float, 3, 3>&, matrix<float, 3, 3>&,
               matrix<float, 3, 3>&, matrix<float, 3, 3>&> v1,
    std::tuple<float, float, float, float> v2
    ) noexcept
{
    std::get<0>(v1) *= std::get<0>(v2);
    std::get<1>(v1) *= std::get<1>(v2);
    std::get<2>(v1) *= std::get<2>(v2);
    std::get<3>(v1) *= std::get<3>(v2);
    return ;
}

// ---------------------------------------------------------------------------
// division operator/
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> operator/(
    const matrix<float, 3, 3>& m1, const float s2) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m1.data();
    _mm256_store_ps(ptr_r,   _mm256_div_ps(_mm256_load_ps(ptr_1  ), _mm256_set1_ps(s2)));
    _mm128_store_ps(ptr_r+8, _mm128_div_ps(_mm128_load_ps(ptr_1+8), _mm128_set1_ps(s2)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
operator/(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<float, float> rhs) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m1 = std::get<0>(lhs).data();
    typename matrix<float, 3, 3>::const_pointer ptr_m2 = std::get<1>(lhs).data();

    _mm256_store_ps(ptr_r1, _mm256_div_ps(_mm256_load_ps(ptr_m1), _mm256_set1_ps(std::get<0>(rhs))));
    _mm256_store_ps(ptr_r2, _mm256_div_ps(_mm256_load_ps(ptr_m2), _mm256_set1_ps(std::get<1>(rhs))));

    const __m256 mid = _mm256_div_ps(
        _mm256_insertf128_ps(_mm256_castps128_ps256(
            _mm_load_ps(ptr_m1+8)),         _mm_load_ps(ptr_m2+8),         1),
        _mm256_insertf128_ps(_mm256_castps128_ps256(
            _mm_set1_ps(std::get<0>(rhs))), _mm_set1_ps(std::get<1>(rhs)), 1));

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(mid));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(mid, 1));
    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>, matrix<float, 3, 3>>
operator/(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&> lhs,
          std::tuple<float, float, float> rhs) noexcept
{
    const auto m12 =    std::tie(std::get<0>(lhs), std::get<1>(lhs)) /
        std::tuple<float, float>(std::get<0>(rhs), std::get<1>(rhs));
    return std::make_tuple(std::get<0>(m12),  std::get<1>(m12),
                           std::get<2>(lhs) / std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                  matrix<float, 3, 3>, matrix<float, 3, 3>>
operator/(std::tuple<const matrix<float, 3, 3>&, const matrix<float, 3, 3>&,
                     const matrix<float, 3, 3>&, const matrix<float, 3, 3>&> lhs,
          std::tuple<float, float, float, float> rhs) noexcept
{
    const auto m12 =    std::tie(std::get<0>(lhs), std::get<1>(lhs)) /
        std::tuple<float, float>(std::get<0>(rhs), std::get<1>(rhs));
    const auto m34 =    std::tie(std::get<2>(lhs), std::get<3>(lhs)) /
        std::tuple<float, float>(std::get<2>(rhs), std::get<3>(rhs));

    return std::make_tuple(std::get<0>(m12), std::get<1>(m12),
                           std::get<0>(m34), std::get<1>(m34));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<float, 3, 3>&, matrix<float, 3, 3>&> v1,
    std::tuple<float, float> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<float, 3, 3>&, matrix<float, 3, 3>&, matrix<float, 3, 3>&> v1,
    std::tuple<float, float, float> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    std::get<2>(v1) /= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<float, 3, 3>&, matrix<float, 3, 3>&,
               matrix<float, 3, 3>&, matrix<float, 3, 3>&> v1,
    std::tuple<float, float, float, float> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    std::get<2>(v1) /= std::get<2>(v2);
    std::get<3>(v1) /= std::get<3>(v2);
    return ;
}



// ---------------------------------------------------------------------------
// matrix3x3-matrix3x3 multiplication
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> operator*(
    const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2) noexcept
{
    // TODO: tune with __m256 and haddps
    const __m128 m1_0 = _mm_load_ps(m1.data()  );
    const __m128 m1_1 = _mm_load_ps(m1.data()+4);
    const __m128 m1_2 = _mm_load_ps(m1.data()+8);

    const __m128 m2_0 = _mm_set_ps(0.0f, m2(2,0), m2(1,0), m2(0,0));
    const __m128 m2_1 = _mm_set_ps(0.0f, m2(2,1), m2(1,1), m2(0,1));
    const __m128 m2_2 = _mm_set_ps(0.0f, m2(2,2), m2(1,2), m2(0,2));

    const auto dot = [](const __m128 l, const __m128 r) noexcept -> float {
        alignas(16) float pack[4];
        _mm_store_ps(pack, _mm_mul_ps(l, r));
        return pack[0] + pack[1] + pack[2];
    };

    return matrix<float, 3, 3>(
        dot(m1_0, m2_0), dot(m1_0, m2_1), dot(m1_0, m2_2),
        dot(m1_1, m2_0), dot(m1_1, m2_1), dot(m1_1, m2_2),
        dot(m1_2, m2_0), dot(m1_2, m2_1), dot(m1_2, m2_2));
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> max(
    const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<float, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_max_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_max_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}

template<>
MAVE_INLINE matrix<float, 3, 3> min(
    const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<float, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_ps(ptr_r,   _mm256_min_ps(_mm256_load_ps(ptr_1  ), _mm256_load_ps(ptr_2  )));
    _mm128_store_ps(ptr_r+8, _mm128_min_ps(_mm128_load_ps(ptr_1+8), _mm128_load_ps(ptr_2+8)));
    return retval;
}

// floor ---------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> floor(const matrix<float, 3, 3>& m) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m.data();
    _mm256_store_ps(ptr_r,   _mm256_floor_ps(_mm256_load_ps(ptr_1  )));
    _mm128_store_ps(ptr_r+8, _mm128_floor_ps(_mm128_load_ps(ptr_1+8)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
floor(const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m1 = m1.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m2 = m2.data();

    _mm256_store_ps(ptr_r1, _mm256_floor_ps(_mm256_load_ps(ptr_m1)));
    _mm256_store_ps(ptr_r2, _mm256_floor_ps(_mm256_load_ps(ptr_m2)));

    __m256 m12 = _mm256_floor_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(ptr_m1+8)), _mm_load_ps(ptr_m2+8), 1));

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(m12));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(m12, 1));

    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                       matrix<float, 3, 3>>
floor(const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2,
      const matrix<float, 3, 3>& m3) noexcept
{
    const auto r12 = floor(m1, m2);
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12), floor(m3));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                       matrix<float, 3, 3>, matrix<float, 3, 3>>
floor(const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2,
      const matrix<float, 3, 3>& m3, const matrix<float, 3, 3>& m4) noexcept
{
    const auto r12 = floor(m1, m2);
    const auto r34 = floor(m3, m4);
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ceil ----------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 3> ceil(const matrix<float, 3, 3>& m) noexcept
{
    matrix<float, 3, 3> retval;
    typename matrix<float, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<float, 3, 3>::const_pointer ptr_1 = m.data();
    _mm256_store_ps(ptr_r,   _mm256_ceil_ps(_mm256_load_ps(ptr_1  )));
    _mm128_store_ps(ptr_r+8, _mm128_ceil_ps(_mm128_load_ps(ptr_1+8)));
    return retval;
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>>
ceil(const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2) noexcept
{
    matrix<float, 3, 3> r1, r2;
    typename matrix<float, 3, 3>::pointer       ptr_r1 = r1.data();
    typename matrix<float, 3, 3>::pointer       ptr_r2 = r2.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m1 = m1.data();
    typename matrix<float, 3, 3>::const_pointer ptr_m2 = m2.data();

    _mm256_store_ps(ptr_r1, _mm256_ceil_ps(_mm256_load_ps(ptr_m1)));
    _mm256_store_ps(ptr_r2, _mm256_ceil_ps(_mm256_load_ps(ptr_m2)));

    __m256 m12 = _mm256_ceil_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(ptr_m1+8)), _mm_load_ps(ptr_m2+8), 1));

    _mm128_store_ps(ptr_r1+8, _mm256_castps256_ps128(m12));
    _mm128_store_ps(ptr_r2+8, _mm256_extractf128_ps(m12, 1));

    return std::make_tuple(r1, r2);
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                       matrix<float, 3, 3>>
ceil(const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2,
      const matrix<float, 3, 3>& m3) noexcept
{
    const auto r12 = ceil(m1, m2);
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12), ceil(m3));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 3>, matrix<float, 3, 3>,
                       matrix<float, 3, 3>, matrix<float, 3, 3>>
ceil(const matrix<float, 3, 3>& m1, const matrix<float, 3, 3>& m2,
      const matrix<float, 3, 3>& m3, const matrix<float, 3, 3>& m4) noexcept
{
    const auto r12 = ceil(m1, m2);
    const auto r34 = ceil(m3, m4);
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
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

#endif // MAVE_AVX2_MATRIX_3x3_FLOAT_HPP
