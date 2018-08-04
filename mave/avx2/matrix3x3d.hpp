#ifndef MAVE_AVX2_MATRIX_3x3_DOUBLE_HPP
#define MAVE_AVX2_MATRIX_3x3_DOUBLE_HPP

#ifndef __AVX2__
#error "mave/avx2/matrix3x3d.hpp requires avx support but __AVX2__ is not defined."
#endif

#ifndef MAVE_MATRIX_HPP
#error "do not use mave/avx/matrix3x3d.hpp alone. please include mave/matrix.hpp."
#endif

#ifdef MAVE_MATRIX_3X3_DOUBLE_IMPLEMENTATION
#error "specialization of vector for 3x double is already defined"
#endif

#define MAVE_MATRIX_3X3_DOUBLE_IMPLEMENTATION "avx2"

#include <x86intrin.h> // for *nix
#include <type_traits>
#include <array>
#include <cmath>

namespace mave
{

template<>
struct alignas(32) matrix<double, 3, 3>
{
    static constexpr std::size_t row_size    = 3;
    static constexpr std::size_t column_size = 3;
    static constexpr std::size_t total_size  = 9;
    using value_type      = double;
    using storage_type    = std::array<double, 12>;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;

    matrix(double v00, double v01, double v02,
           double v10, double v11, double v12,
           double v20, double v21, double v22)
        : vs_{{v00, v01, v02, 0.0, v10, v11, v12, 0.0, v20, v21, v22, 0.0}}
    {}

    matrix(){vs_.fill(0.0);}
    ~matrix() = default;
    matrix(const matrix&) = default;
    matrix(matrix&&)      = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&)      = default;

    template<typename T>
    matrix& operator=(const matrix<T, 3, 3>& rhs) noexcept
    {
        vs_[ 0] = static_cast<double>(rhs(0, 0));
        vs_[ 1] = static_cast<double>(rhs(0, 1));
        vs_[ 2] = static_cast<double>(rhs(0, 2));
        vs_[ 3] = 0.0;
        vs_[ 4] = static_cast<double>(rhs(1, 0));
        vs_[ 5] = static_cast<double>(rhs(1, 1));
        vs_[ 6] = static_cast<double>(rhs(1, 2));
        vs_[ 7] = 0.0;
        vs_[ 8] = static_cast<double>(rhs(2, 0));
        vs_[ 9] = static_cast<double>(rhs(2, 1));
        vs_[10] = static_cast<double>(rhs(2, 2));
        vs_[11] = 0.0;
        return *this;
    }

    matrix& operator+=(const matrix<double, 3, 3>& other) noexcept
    {
        pointer       self = this->data();
        const_pointer othr = other.data();
        _mm256_store_pd(self,   _mm256_add_pd(_mm256_load_pd(self  ), _mm256_load_pd(othr  )));
        _mm256_store_pd(self+4, _mm256_add_pd(_mm256_load_pd(self+4), _mm256_load_pd(othr+4)));
        _mm256_store_pd(self+8, _mm256_add_pd(_mm256_load_pd(self+8), _mm256_load_pd(othr+8)));
        return *this;
    }
    matrix& operator-=(const matrix<double, 3, 3>& other) noexcept
    {
        pointer       self = this->data();
        const_pointer othr = other.data();
        _mm256_store_pd(self,   _mm256_sub_pd(_mm256_load_pd(self  ), _mm256_load_pd(othr  )));
        _mm256_store_pd(self+4, _mm256_sub_pd(_mm256_load_pd(self+4), _mm256_load_pd(othr+4)));
        _mm256_store_pd(self+8, _mm256_sub_pd(_mm256_load_pd(self+8), _mm256_load_pd(othr+8)));
        return *this;
    }
    matrix& operator*=(const double other) noexcept
    {
        const __m256d othr = _mm256_set1_pd(other);
        pointer       self = this->data();
        _mm256_store_pd(self,   _mm256_mul_pd(_mm256_load_pd(self  ), othr));
        _mm256_store_pd(self+4, _mm256_mul_pd(_mm256_load_pd(self+4), othr));
        _mm256_store_pd(self+8, _mm256_mul_pd(_mm256_load_pd(self+8), othr));
        return *this;
    }
    matrix& operator/=(const double other) noexcept
    {
        const __m256d othr = _mm256_set1_pd(other);
        pointer       self = this->data();
        _mm256_store_pd(self,   _mm256_div_pd(_mm256_load_pd(self  ), othr));
        _mm256_store_pd(self+4, _mm256_div_pd(_mm256_load_pd(self+4), othr));
        _mm256_store_pd(self+8, _mm256_div_pd(_mm256_load_pd(self+8), othr));
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
    {return (vs_[3]==0.0) && (vs_[7]==0.0) && (vs_[11]==0.0);}

  private:
    alignas(32) storage_type vs_;
};

// ---------------------------------------------------------------------------
// negation operator
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 3> operator-(const matrix<double, 3, 3>& m) noexcept
{
    const __m256d zero = _mm256_setzero_pd();
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_m = m.data();
    _mm256_store_pd(ptr_r,   _mm256_sub_pd(zero, _mm256_load_pd(ptr_m  )));
    _mm256_store_pd(ptr_r+4, _mm256_sub_pd(zero, _mm256_load_pd(ptr_m+4)));
    _mm256_store_pd(ptr_r+8, _mm256_sub_pd(zero, _mm256_load_pd(ptr_m+8)));
    return retval;
}
template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
operator-(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> ms) noexcept
{
    return std::make_pair(-std::get<0>(ms), -std::get<1>(ms));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>, matrix<double, 3, 3>>
operator-(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> ms) noexcept
{
    return std::make_tuple(-std::get<0>(ms), -std::get<1>(ms), -std::get<2>(ms));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                       matrix<double, 3, 3>, matrix<double, 3, 3>>
operator-(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> ms) noexcept
{
    return std::make_tuple(-std::get<0>(ms), -std::get<1>(ms),
                           -std::get<2>(ms), -std::get<3>(ms));
}

// ---------------------------------------------------------------------------
// addition operator+
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 3> operator+(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_pd(ptr_r,   _mm256_add_pd(_mm256_load_pd(ptr_1  ), _mm256_load_pd(ptr_2  )));
    _mm256_store_pd(ptr_r+4, _mm256_add_pd(_mm256_load_pd(ptr_1+4), _mm256_load_pd(ptr_2+4)));
    _mm256_store_pd(ptr_r+8, _mm256_add_pd(_mm256_load_pd(ptr_1+8), _mm256_load_pd(ptr_2+8)));
    return retval;
}

template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
operator+(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> rhs
          ) noexcept
{
    return std::make_pair(std::get<0>(lhs) + std::get<0>(rhs),
                          std::get<1>(lhs) + std::get<1>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>, matrix<double, 3, 3>>
operator+(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> rhs) noexcept
{
    return std::make_tuple(std::get<0>(lhs) + std::get<0>(rhs),
                           std::get<1>(lhs) + std::get<1>(rhs),
                           std::get<2>(lhs) + std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                  matrix<double, 3, 3>, matrix<double, 3, 3>>
operator+(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> rhs
          ) noexcept
{
    return std::make_tuple(std::get<0>(lhs) + std::get<0>(rhs),
                           std::get<1>(lhs) + std::get<1>(rhs),
                           std::get<2>(lhs) + std::get<2>(rhs),
                           std::get<3>(lhs) + std::get<3>(rhs));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<double, 3, 3>&,       matrix<double, 3, 3>&> v1,
    std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> v2
    ) noexcept
{
    std::get<0>(v1) += std::get<0>(v2);
    std::get<1>(v1) += std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<double, 3, 3>&, matrix<double, 3, 3>&,
                     matrix<double, 3, 3>&> v1,
    std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
               const matrix<double, 3, 3>&> v2) noexcept
{
    std::get<0>(v1) += std::get<0>(v2);
    std::get<1>(v1) += std::get<1>(v2);
    std::get<2>(v1) += std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<double, 3, 3>&,       matrix<double, 3, 3>&,
                     matrix<double, 3, 3>&,       matrix<double, 3, 3>&> v1,
    std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
               const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> v2
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
MAVE_INLINE matrix<double, 3, 3> operator-(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_pd(ptr_r,   _mm256_sub_pd(_mm256_load_pd(ptr_1  ), _mm256_load_pd(ptr_2  )));
    _mm256_store_pd(ptr_r+4, _mm256_sub_pd(_mm256_load_pd(ptr_1+4), _mm256_load_pd(ptr_2+4)));
    _mm256_store_pd(ptr_r+8, _mm256_sub_pd(_mm256_load_pd(ptr_1+8), _mm256_load_pd(ptr_2+8)));
    return retval;
}
template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
operator-(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> rhs
          ) noexcept
{
    return std::make_pair(std::get<0>(lhs) - std::get<0>(rhs),
                          std::get<1>(lhs) - std::get<1>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>, matrix<double, 3, 3>>
operator-(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> rhs) noexcept
{
    return std::make_tuple(std::get<0>(lhs) - std::get<0>(rhs),
                           std::get<1>(lhs) - std::get<1>(rhs),
                           std::get<2>(lhs) - std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                  matrix<double, 3, 3>, matrix<double, 3, 3>>
operator-(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> rhs
          ) noexcept
{
    return std::make_tuple(std::get<0>(lhs) - std::get<0>(rhs),
                           std::get<1>(lhs) - std::get<1>(rhs),
                           std::get<2>(lhs) - std::get<2>(rhs),
                           std::get<3>(lhs) - std::get<3>(rhs));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<double, 3, 3>&,       matrix<double, 3, 3>&> v1,
    std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> v2
    ) noexcept
{
    std::get<0>(v1) -= std::get<0>(v2);
    std::get<1>(v1) -= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<double, 3, 3>&, matrix<double, 3, 3>&,
                     matrix<double, 3, 3>&> v1,
    std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
               const matrix<double, 3, 3>&> v2) noexcept
{
    std::get<0>(v1) -= std::get<0>(v2);
    std::get<1>(v1) -= std::get<1>(v2);
    std::get<2>(v1) -= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<double, 3, 3>&,       matrix<double, 3, 3>&,
                     matrix<double, 3, 3>&,       matrix<double, 3, 3>&> v1,
    std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
               const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> v2
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

template<>
MAVE_INLINE matrix<double, 3, 3> operator*(
    const matrix<double, 3, 3>& m1, const double s2) noexcept
{
    const __m256d scalar = _mm256_set1_pd(s2);
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    _mm256_store_pd(ptr_r,   _mm256_mul_pd(_mm256_load_pd(ptr_1  ), scalar));
    _mm256_store_pd(ptr_r+4, _mm256_mul_pd(_mm256_load_pd(ptr_1+4), scalar));
    _mm256_store_pd(ptr_r+8, _mm256_mul_pd(_mm256_load_pd(ptr_1+8), scalar));
    return retval;
}
template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
operator*(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<double, double> rhs) noexcept
{
    return std::make_pair(std::get<0>(lhs) * std::get<0>(rhs),
                          std::get<1>(lhs) * std::get<1>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>, matrix<double, 3, 3>>
operator*(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> lhs,
          std::tuple<double, double, double> rhs) noexcept
{
    return std::make_tuple(std::get<0>(lhs) * std::get<0>(rhs),
                           std::get<1>(lhs) * std::get<1>(rhs),
                           std::get<2>(lhs) * std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                  matrix<double, 3, 3>, matrix<double, 3, 3>>
operator*(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<double, double, double, double> rhs) noexcept
{
    return std::make_tuple(std::get<0>(lhs) * std::get<0>(rhs),
                           std::get<1>(lhs) * std::get<1>(rhs),
                           std::get<2>(lhs) * std::get<2>(rhs),
                           std::get<3>(lhs) * std::get<3>(rhs));
}

// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 3> operator*(
    const double s1, const matrix<double, 3, 3>& m2) noexcept
{
    const __m256d scalar = _mm256_set1_pd(s1);
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_pd(ptr_r,   _mm256_mul_pd(scalar, _mm256_load_pd(ptr_2  )));
    _mm256_store_pd(ptr_r+4, _mm256_mul_pd(scalar, _mm256_load_pd(ptr_2+4)));
    _mm256_store_pd(ptr_r+8, _mm256_mul_pd(scalar, _mm256_load_pd(ptr_2+8)));
    return retval;
}
template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
operator*(std::tuple<double, double> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> rhs
          ) noexcept
{
    return std::make_pair(std::get<0>(lhs) * std::get<0>(rhs),
                          std::get<1>(lhs) * std::get<1>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>, matrix<double, 3, 3>>
operator*(std::tuple<double, double, double> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> rhs) noexcept
{
    return std::make_tuple(std::get<0>(lhs) * std::get<0>(rhs),
                           std::get<1>(lhs) * std::get<1>(rhs),
                           std::get<2>(lhs) * std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                  matrix<double, 3, 3>, matrix<double, 3, 3>>
operator*(std::tuple<double, double, double, double> lhs,
          std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> rhs
          ) noexcept
{
    return std::make_tuple(std::get<0>(lhs) * std::get<0>(rhs),
                           std::get<1>(lhs) * std::get<1>(rhs),
                           std::get<2>(lhs) * std::get<2>(rhs),
                           std::get<3>(lhs) * std::get<3>(rhs));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<double, 3, 3>&, matrix<double, 3, 3>&> v1,
    std::tuple<double, double> v2) noexcept
{
    std::get<0>(v1) *= std::get<0>(v2);
    std::get<1>(v1) *= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<double, 3, 3>&, matrix<double, 3, 3>&, matrix<double, 3, 3>&> v1,
    std::tuple<double, double, double> v2) noexcept
{
    std::get<0>(v1) *= std::get<0>(v2);
    std::get<1>(v1) *= std::get<1>(v2);
    std::get<2>(v1) *= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<double, 3, 3>&, matrix<double, 3, 3>&,
               matrix<double, 3, 3>&, matrix<double, 3, 3>&> v1,
    std::tuple<double, double, double, double> v2
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
MAVE_INLINE matrix<double, 3, 3> operator/(
    const matrix<double, 3, 3>& m1, const double s2) noexcept
{
    const __m256d scalar = _mm256_set1_pd(s2);
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    _mm256_store_pd(ptr_r,   _mm256_div_pd(_mm256_load_pd(ptr_1  ), scalar));
    _mm256_store_pd(ptr_r+4, _mm256_div_pd(_mm256_load_pd(ptr_1+4), scalar));
    _mm256_store_pd(ptr_r+8, _mm256_div_pd(_mm256_load_pd(ptr_1+8), scalar));
    return retval;
}
template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
operator/(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<double, double> rhs) noexcept
{
    return std::make_pair(std::get<0>(lhs) / std::get<0>(rhs),
                          std::get<1>(lhs) / std::get<1>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>, matrix<double, 3, 3>>
operator/(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&> lhs,
          std::tuple<double, double, double> rhs) noexcept
{
    return std::make_tuple(std::get<0>(lhs) / std::get<0>(rhs),
                           std::get<1>(lhs) / std::get<1>(rhs),
                           std::get<2>(lhs) / std::get<2>(rhs));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                  matrix<double, 3, 3>, matrix<double, 3, 3>>
operator/(std::tuple<const matrix<double, 3, 3>&, const matrix<double, 3, 3>&,
                     const matrix<double, 3, 3>&, const matrix<double, 3, 3>&> lhs,
          std::tuple<double, double, double, double> rhs) noexcept
{
    return std::make_tuple(std::get<0>(lhs) / std::get<0>(rhs),
                           std::get<1>(lhs) / std::get<1>(rhs),
                           std::get<2>(lhs) / std::get<2>(rhs),
                           std::get<3>(lhs) / std::get<3>(rhs));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<double, 3, 3>&, matrix<double, 3, 3>&> v1,
    std::tuple<double, double> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<double, 3, 3>&, matrix<double, 3, 3>&, matrix<double, 3, 3>&> v1,
    std::tuple<double, double, double> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    std::get<2>(v1) /= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<double, 3, 3>&, matrix<double, 3, 3>&,
               matrix<double, 3, 3>&, matrix<double, 3, 3>&> v1,
    std::tuple<double, double, double, double> v2) noexcept
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
MAVE_INLINE matrix<double, 3, 3> operator*(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    const __m256d m1_0 = _mm256_load_pd(m1.data()  );
    const __m256d m1_1 = _mm256_load_pd(m1.data()+4);
    const __m256d m1_2 = _mm256_load_pd(m1.data()+8);

    const __m256d m2_0 = _mm256_set_pd(0.0, m2(2,0), m2(1,0), m2(0,0));
    const __m256d m2_1 = _mm256_set_pd(0.0, m2(2,1), m2(1,1), m2(0,1));
    const __m256d m2_2 = _mm256_set_pd(0.0, m2(2,2), m2(1,2), m2(0,2));

    // TODO: use hadd
    const auto dot = [](const __m256d l, const __m256d r) noexcept -> double {
        alignas(32) double pack[4];
        _mm256_store_pd(pack, _mm256_mul_pd(l, r));
        return pack[0] + pack[1] + pack[2];
    };

    return matrix<double, 3, 3>(
        dot(m1_0, m2_0), dot(m1_0, m2_1), dot(m1_0, m2_2),
        dot(m1_1, m2_0), dot(m1_1, m2_1), dot(m1_1, m2_2),
        dot(m1_2, m2_0), dot(m1_2, m2_1), dot(m1_2, m2_2));
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 3> max(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_pd(ptr_r,   _mm256_max_pd(_mm256_load_pd(ptr_1  ), _mm256_load_pd(ptr_2  )));
    _mm256_store_pd(ptr_r+4, _mm256_max_pd(_mm256_load_pd(ptr_1+4), _mm256_load_pd(ptr_2+4)));
    _mm256_store_pd(ptr_r+8, _mm256_max_pd(_mm256_load_pd(ptr_1+8), _mm256_load_pd(ptr_2+8)));
    return retval;
}

template<>
MAVE_INLINE matrix<double, 3, 3> min(
    const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m1.data();
    typename matrix<double, 3, 3>::const_pointer ptr_2 = m2.data();
    _mm256_store_pd(ptr_r,   _mm256_min_pd(_mm256_load_pd(ptr_1  ), _mm256_load_pd(ptr_2  )));
    _mm256_store_pd(ptr_r+4, _mm256_min_pd(_mm256_load_pd(ptr_1+4), _mm256_load_pd(ptr_2+4)));
    _mm256_store_pd(ptr_r+8, _mm256_min_pd(_mm256_load_pd(ptr_1+8), _mm256_load_pd(ptr_2+8)));
    return retval;
}

// floor ---------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 3> floor(const matrix<double, 3, 3>& m) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m.data();
    _mm256_store_pd(ptr_r,   _mm256_floor_pd(_mm256_load_pd(ptr_1  )));
    _mm256_store_pd(ptr_r+4, _mm256_floor_pd(_mm256_load_pd(ptr_1+4)));
    _mm256_store_pd(ptr_r+8, _mm256_floor_pd(_mm256_load_pd(ptr_1+8)));
    return retval;
}
template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
floor(const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    return std::make_pair(floor(m1), floor(m2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                       matrix<double, 3, 3>>
floor(const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2,
      const matrix<double, 3, 3>& m3) noexcept
{
    return std::make_tuple(floor(m1), floor(m2), floor(m3));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                       matrix<double, 3, 3>, matrix<double, 3, 3>>
floor(const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2,
      const matrix<double, 3, 3>& m3, const matrix<double, 3, 3>& m4) noexcept
{
    return std::make_tuple(floor(m1), floor(m2), floor(m3), floor(m4));
}

// ceil ----------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 3> ceil(const matrix<double, 3, 3>& m) noexcept
{
    matrix<double, 3, 3> retval;
    typename matrix<double, 3, 3>::pointer       ptr_r = retval.data();
    typename matrix<double, 3, 3>::const_pointer ptr_1 = m.data();
    _mm256_store_pd(ptr_r,   _mm256_ceil_pd(_mm256_load_pd(ptr_1  )));
    _mm256_store_pd(ptr_r+4, _mm256_ceil_pd(_mm256_load_pd(ptr_1+4)));
    _mm256_store_pd(ptr_r+8, _mm256_ceil_pd(_mm256_load_pd(ptr_1+8)));
    return retval;
}
template<>
MAVE_INLINE std::pair<matrix<double, 3, 3>, matrix<double, 3, 3>>
ceil(const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2) noexcept
{
    return std::make_pair(ceil(m1), ceil(m2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                       matrix<double, 3, 3>>
ceil(const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2,
     const matrix<double, 3, 3>& m3) noexcept
{
    return std::make_tuple(ceil(m1), ceil(m2), ceil(m3));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 3>, matrix<double, 3, 3>,
                       matrix<double, 3, 3>, matrix<double, 3, 3>>
ceil(const matrix<double, 3, 3>& m1, const matrix<double, 3, 3>& m2,
     const matrix<double, 3, 3>& m3, const matrix<double, 3, 3>& m4) noexcept
{
    return std::make_tuple(ceil(m1), ceil(m2), ceil(m3), ceil(m4));
}

// ---------------------------------------------------------------------------

} // mave
#endif // MAVE_AVX2_MATRIX_3x3_DOUBLE_HPP
