#ifndef MAVE_AVX_VECTOR3_DOUBLE_HPP
#define MAVE_AVX_VECTOR3_DOUBLE_HPP

#ifndef __AVX__
#error "mave/avx/vector3d.hpp requires avx support but __AVX__ is not defined."
#endif

#ifndef MAVE_VECTOR_HPP
#error "do not use mave/avx/vector3d.hpp alone. please include mave/vector.hpp."
#endif

#ifdef MAVE_VECTOR3_DOUBLE_IMPLEMENTATION
#error "specialization of vector for 3x double is already defined"
#endif

#define MAVE_VECTOR3_DOUBLE_IMPLEMENTATION "avx"

#include <x86intrin.h> // for *nix
#include <type_traits>
#include <array>
#include <cmath>

namespace mave
{

template<>
struct alignas(32) matrix<double, 3, 1>
{
    static constexpr std::size_t row_size    = 3;
    static constexpr std::size_t column_size = 1;
    static constexpr std::size_t total_size  = 3;
    using value_type      = double;
    using storage_type    = std::array<double, 4>; // XXX for AVX packing
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;

    template<typename T1, typename T2, typename T3>
    matrix(T1&& v1, T2&& v2, T3&& v3) noexcept
        : vs_{{static_cast<double>(v1), static_cast<double>(v2),
               static_cast<double>(v3), 0.0}}
    {}

    matrix(const std::array<double, 3>& arg) noexcept
        : vs_{{arg[0], arg[1], arg[2], 0.0}}
    {}

    matrix(__m256d pack) noexcept
    {
        _mm256_store_pd(this->data(), pack);
    }
    matrix& operator=(__m256d pack) noexcept
    {
        _mm256_store_pd(this->data(), pack);
        return *this;
    }

    matrix(__m128 pack) noexcept
    {
        _mm256_store_pd(this->data(), _mm256_cvtps_pd(pack));
    }
    matrix& operator=(__m128 pack) noexcept
    {
        _mm256_store_pd(this->data(), _mm256_cvtps_pd(pack));
        return *this;
    }

    matrix(__m128i pack) noexcept
    {
        _mm256_store_pd(this->data(), _mm256_cvtepi32_pd(pack));
    }
    matrix& operator=(__m128i pack) noexcept
    {
        _mm256_store_pd(this->data(), _mm256_cvtepi32_pd(pack));
        return *this;
    }

    matrix(): vs_{{0.0, 0.0, 0.0, 0.0}}{}
    ~matrix() = default;
    matrix(const matrix&) = default;
    matrix(matrix&&)      = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&)      = default;

    template<typename T>
    matrix& operator=(const matrix<T, 3, 1>& rhs) noexcept
    {
        vs_[0] = static_cast<double>(rhs[0]);
        vs_[1] = static_cast<double>(rhs[1]);
        vs_[2] = static_cast<double>(rhs[2]);
        vs_[3] = 0.0;
        return *this;
    }

    matrix& operator+=(const matrix<double, 3, 1>& other) noexcept
    {
        const __m256d v1 = _mm256_load_pd(this->data());
        const __m256d v2 = _mm256_load_pd(other.data());
        _mm256_store_pd(this->data(), _mm256_add_pd(v1, v2));
        return *this;
    }
    matrix& operator-=(const matrix<double, 3, 1>& other) noexcept
    {
        const __m256d v1 = _mm256_load_pd(this->data());
        const __m256d v2 = _mm256_load_pd(other.data());
        _mm256_store_pd(this->data(), _mm256_sub_pd(v1, v2));
        return *this;
    }
    matrix& operator*=(const double other) noexcept
    {
        const __m256d v1 = _mm256_load_pd(this->data());
        const __m256d v2 = _mm256_set1_pd(other);
        _mm256_store_pd(this->data(), _mm256_mul_pd(v1, v2));
        return *this;
    }
    matrix& operator/=(const double other) noexcept
    {
        const __m256d v1 = _mm256_load_pd(this->data());
        const __m256d v2 = _mm256_set1_pd(other);
        _mm256_store_pd(this->data(), _mm256_div_pd(v1, v2));
        return *this;
    }

    size_type size() const noexcept {return total_size;}

    pointer       data()       noexcept {return vs_.data();}
    const_pointer data() const noexcept {return vs_.data();}

    reference               at(size_type i)       {return vs_.at(i);}
    const_reference         at(size_type i) const {return vs_.at(i);}
    reference       operator[](size_type i)       noexcept {return vs_[i];}
    const_reference operator[](size_type i) const noexcept {return vs_[i];}

    reference       at(size_type i, size_type)       {return vs_.at(i);}
    const_reference at(size_type i, size_type) const {return vs_.at(i);}
    reference       operator()(size_type i, size_type)       noexcept {return vs_[i];}
    const_reference operator()(size_type i, size_type) const noexcept {return vs_[i];}

    bool diagnosis() const noexcept {return vs_[3] == 0.0;}

  private:
    alignas(32) storage_type vs_;
};

// ---------------------------------------------------------------------------
// negation operator-
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> operator-(const matrix<double, 3, 1>& v) noexcept
{
    return _mm256_sub_pd(_mm256_setzero_pd(), _mm256_load_pd(v.data()));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
operator-(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> ms
          ) noexcept
{
    return std::make_tuple(-std::get<0>(ms), -std::get<1>(ms));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
operator-(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> ms) noexcept
{
    return std::make_tuple(-std::get<0>(ms), -std::get<1>(ms), -std::get<2>(ms));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
operator-(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> ms
          ) noexcept
{
    return std::make_tuple(-std::get<0>(ms), -std::get<1>(ms),
                           -std::get<2>(ms), -std::get<3>(ms));
}

// ---------------------------------------------------------------------------
// addition operator+
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> operator+(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_add_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
operator+(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v2
          ) noexcept
{
    return std::make_tuple(std::get<0>(v1) + std::get<0>(v2),
                          std::get<1>(v1) + std::get<1>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
operator+(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) + std::get<0>(v2),
                           std::get<1>(v1) + std::get<1>(v2),
                           std::get<2>(v1) + std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
operator+(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> v2
          ) noexcept
{
    return std::make_tuple(std::get<0>(v1) + std::get<0>(v2),
                           std::get<1>(v1) + std::get<1>(v2),
                           std::get<2>(v1) + std::get<2>(v2),
                           std::get<3>(v1) + std::get<3>(v2));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<double,3,1>&,       matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v2
    ) noexcept
{
    std::get<0>(v1) += std::get<0>(v2);
    std::get<1>(v1) += std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<double,3,1>&, matrix<double,3,1>&,
                     matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&> v2) noexcept
{
    std::get<0>(v1) += std::get<0>(v2);
    std::get<1>(v1) += std::get<1>(v2);
    std::get<2>(v1) += std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<double,3,1>&,       matrix<double,3,1>&,
                     matrix<double,3,1>&,       matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&, const matrix<double,3,1>&> v2
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
MAVE_INLINE matrix<double, 3, 1> operator-(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_sub_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
operator-(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v2
          ) noexcept
{
    return std::make_tuple(std::get<0>(v1) - std::get<0>(v2),
                          std::get<1>(v1) - std::get<1>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
operator-(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) - std::get<0>(v2),
                           std::get<1>(v1) - std::get<1>(v2),
                           std::get<2>(v1) - std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
operator-(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> v2
          ) noexcept
{
    return std::make_tuple(std::get<0>(v1) - std::get<0>(v2),
                           std::get<1>(v1) - std::get<1>(v2),
                           std::get<2>(v1) - std::get<2>(v2),
                           std::get<3>(v1) - std::get<3>(v2));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<double,3,1>&,       matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v2
    ) noexcept
{
    std::get<0>(v1) -= std::get<0>(v2);
    std::get<1>(v1) -= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<double,3,1>&, matrix<double,3,1>&,
                     matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&> v2) noexcept
{
    std::get<0>(v1) -= std::get<0>(v2);
    std::get<1>(v1) -= std::get<1>(v2);
    std::get<2>(v1) -= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<double,3,1>&,       matrix<double,3,1>&,
                     matrix<double,3,1>&,       matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&, const matrix<double,3,1>&> v2
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

// scalar * vector -----------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> operator*(
    const double v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(v2.data()));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
operator*(std::tuple<double, double> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v2
          ) noexcept
{
    return std::make_tuple(std::get<0>(v1) * std::get<0>(v2),
                          std::get<1>(v1) * std::get<1>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
operator*(std::tuple<double, double, double> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) * std::get<0>(v2),
                           std::get<1>(v1) * std::get<1>(v2),
                           std::get<2>(v1) * std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
operator*(std::tuple<double, double, double, double> v1,
          std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> v2
          ) noexcept
{
    return std::make_tuple(std::get<0>(v1) * std::get<0>(v2),
                           std::get<1>(v1) * std::get<1>(v2),
                           std::get<2>(v1) * std::get<2>(v2),
                           std::get<3>(v1) * std::get<3>(v2));
}

// vector * scalar -----------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> operator*(
    const matrix<double, 3, 1>& v1, const double v2) noexcept
{
    return _mm256_mul_pd(_mm256_load_pd(v1.data()), _mm256_set1_pd(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
operator*(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<double, double> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) * std::get<0>(v2),
                          std::get<1>(v1) * std::get<1>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
operator*(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> v1,
          std::tuple<double, double, double> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) * std::get<0>(v2),
                           std::get<1>(v1) * std::get<1>(v2),
                           std::get<2>(v1) * std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
operator*(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<double, double, double, double> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) * std::get<0>(v2),
                           std::get<1>(v1) * std::get<1>(v2),
                           std::get<2>(v1) * std::get<2>(v2),
                           std::get<3>(v1) * std::get<3>(v2));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<double,3,1>&, matrix<double,3,1>&> v1,
    std::tuple<double, double> v2) noexcept
{
    std::get<0>(v1) *= std::get<0>(v2);
    std::get<1>(v1) *= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<double,3,1>&, matrix<double,3,1>&, matrix<double,3,1>&> v1,
    std::tuple<double, double, double> v2) noexcept
{
    std::get<0>(v1) *= std::get<0>(v2);
    std::get<1>(v1) *= std::get<1>(v2);
    std::get<2>(v1) *= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<double,3,1>&, matrix<double,3,1>&,
               matrix<double,3,1>&, matrix<double,3,1>&> v1,
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
MAVE_INLINE matrix<double, 3, 1> operator/(
    const matrix<double, 3, 1>& v1, const double v2) noexcept
{
    return _mm256_div_pd(_mm256_load_pd(v1.data()), _mm256_set1_pd(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
operator/(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<double, double> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) / std::get<0>(v2),
                          std::get<1>(v1) / std::get<1>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
operator/(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&> v1,
          std::tuple<double, double, double> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) / std::get<0>(v2),
                           std::get<1>(v1) / std::get<1>(v2),
                           std::get<2>(v1) / std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
operator/(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
                     const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
          std::tuple<double, double, double, double> v2) noexcept
{
    return std::make_tuple(std::get<0>(v1) / std::get<0>(v2),
                           std::get<1>(v1) / std::get<1>(v2),
                           std::get<2>(v1) / std::get<2>(v2),
                           std::get<3>(v1) / std::get<3>(v2));
}

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<double,3,1>&, matrix<double,3,1>&> v1,
    std::tuple<double, double> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<double,3,1>&, matrix<double,3,1>&, matrix<double,3,1>&> v1,
    std::tuple<double, double, double> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    std::get<2>(v1) /= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<double,3,1>&, matrix<double,3,1>&,
               matrix<double,3,1>&, matrix<double,3,1>&> v1,
    std::tuple<double, double, double, double> v2) noexcept
{
    std::get<0>(v1) /= std::get<0>(v2);
    std::get<1>(v1) /= std::get<1>(v2);
    std::get<2>(v1) /= std::get<2>(v2);
    std::get<3>(v1) /= std::get<3>(v2);
    return ;
}

// ---------------------------------------------------------------------------
// length
// ---------------------------------------------------------------------------

// length_sq -----------------------------------------------------------------

template<>
MAVE_INLINE double length_sq(const matrix<double, 3, 1>& v) noexcept
{
    alignas(32) double pack[4];
    _mm256_store_pd(pack, _mm256_mul_pd(
        _mm256_load_pd(v.data()), _mm256_load_pd(v.data())));
    return pack[0] + pack[1] + pack[2];
}

template<>
MAVE_INLINE std::tuple<double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d hadd = _mm256_hadd_pd(mul1, mul2);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //         |  |  extract
    // |a0|b0|<+--'

    alignas(16) double pack[2];
    _mm_store_pd(pack, _mm_add_pd(
        _mm256_castpd256_pd128(hadd), _mm256_extractf128_pd(hadd, 1)));

    return std::make_tuple(pack[0], pack[1]);
}

template<>
MAVE_INLINE std::tuple<double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    const auto l12 = length_sq(v1, v2);
    return std::make_tuple(std::get<0>(l12), std::get<1>(l12), length_sq(v3));
}

template<>
MAVE_INLINE std::tuple<double, double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00| |c1|c2|c3|00| |d1|d2|d3|00|
    //  +--'  |  |    +--'  |  |    +--'  |  |    +--'  |  |
    //  |     +--'    |     |  |    |     +--'    |     |  |
    //  |  .--|--.----+-----+--'    |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| hadd1         |cc|dd|c3|d3| hadd2
    //   '--'-+--+----.--.  .--.----'--'  |  |
    //        |  |   |aa|bb|cc|dd|        |  | extractf128 & insertf128
    //        +--+-->|a3|b3|c3|d3| <------+--+
    //               |a |b |c |d |

    const __m256d hadd1 = _mm256_hadd_pd(mul1, mul2);
    const __m256d hadd2 = _mm256_hadd_pd(mul3, mul4);

    const __m256d v12 = _mm256_insertf128_pd(
            hadd1, _mm256_extractf128_pd(hadd2, 0), 1);
    const __m256d v34 = _mm256_insertf128_pd(_mm256_castpd128_pd256(
            _mm256_extractf128_pd(hadd1, 1)),
            _mm256_extractf128_pd(hadd2, 1), 1);

    alignas(32) double pack[4];
    _mm256_store_pd(pack, _mm256_add_pd(v12, v34));
    return std::make_tuple(pack[0], pack[1], pack[2], pack[3]);
}

// length -------------------------------------------------------------------

template<>
MAVE_INLINE double length(const matrix<double, 3, 1>& v) noexcept
{
    return std::sqrt(length_sq(v));
}

template<>
MAVE_INLINE std::tuple<double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d hadd = _mm256_hadd_pd(mul1, mul2);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //         |  |  extract
    // |a0|b0|<+--'

    alignas(16) double pack[2];
    _mm_store_pd(pack, _mm_sqrt_pd(_mm_add_pd(
        _mm256_castpd256_pd128(hadd), _mm256_extractf128_pd(hadd, 1))));

    return std::make_tuple(pack[0], pack[1]);
}

template<>
MAVE_INLINE std::tuple<double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);

    const __m256d hadd = _mm256_hadd_pd(mul1, mul2);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //         |  |  extract
    // |a0|b0|<+--'

    alignas(32) double pack[4];
    _mm256_store_pd(pack, mul3);
    _mm256_store_pd(pack, _mm256_sqrt_pd(_mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_add_pd(
            _mm256_castpd256_pd128(hadd), _mm256_extractf128_pd(hadd, 1))),
        _mm_set_sd(pack[0] + pack[1] + pack[2]), 1)));

    return std::make_tuple(pack[0], pack[1], pack[2]);
}

template<>
MAVE_INLINE std::tuple<double, double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00| |c1|c2|c3|00| |d1|d2|d3|00|
    //  +--'  |  |    +--'  |  |    +--'  |  |    +--'  |  |
    //  |     +--'    |     |  |    |     +--'    |     |  |
    //  |  .--|--.----+-----+--'    |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| hadd1         |cc|dd|c3|d3| hadd2
    //   '--'-+--+----.--.  .--.----'--'  |  |
    //        |  |   |aa|bb|cc|dd|        |  | extractf128 & insertf128
    //        +--+-->|a3|b3|c3|d3| <------+--+
    //               |a |b |c |d |

    const __m256d hadd1 = _mm256_hadd_pd(mul1, mul2);
    const __m256d hadd2 = _mm256_hadd_pd(mul3, mul4);

    const __m256d v12 = _mm256_insertf128_pd(
            hadd1, _mm256_extractf128_pd(hadd2, 0), 1);
    const __m256d v34 = _mm256_insertf128_pd(_mm256_castpd128_pd256(
            _mm256_extractf128_pd(hadd1, 1)),
            _mm256_extractf128_pd(hadd2, 1), 1);

    _mm256_store_pd(pack, _mm256_sqrt_pd(_mm256_add_pd(v12, v34)));
    return std::make_tuple(pack[0], pack[1], pack[2], pack[3]);
}

// rlength -------------------------------------------------------------------

template<>
MAVE_INLINE double rlength(const matrix<double, 3, 1>& v) noexcept
{
    return 1.0 / std::sqrt(length_sq(v));
}
template<>
MAVE_INLINE std::tuple<double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d hadd = _mm256_hadd_pd(mul1, mul2);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //         |  |  extract
    // |a0|b0|<+--'

    alignas(16) double pack[2];
    _mm_store_pd(pack, _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(_mm_add_pd(
        _mm256_castpd256_pd128(hadd), _mm256_extractf128_pd(hadd, 1)))));

    return std::make_tuple(pack[0], pack[1]);
}
template<>
MAVE_INLINE std::tuple<double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);

    const __m256d hadd = _mm256_hadd_pd(mul1, mul2);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //         |  |  extract
    // |a0|b0|<+--'

    alignas(32) double pack[4];
    _mm256_store_pd(pack, mul3);
    _mm256_store_pd(pack, _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(
        _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_add_pd(
            _mm256_castpd256_pd128(hadd), _mm256_extractf128_pd(hadd, 1))),
            _mm_set_sd(pack[0] + pack[1] + pack[2]), 1))));

    return std::make_tuple(pack[0], pack[1], pack[2]);
}
template<>
MAVE_INLINE std::tuple<double, double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00| |c1|c2|c3|00| |d1|d2|d3|00|
    //  +--'  |  |    +--'  |  |    +--'  |  |    +--'  |  |
    //  |     +--'    |     |  |    |     +--'    |     |  |
    //  |  .--|--.----+-----+--'    |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| hadd1         |cc|dd|c3|d3| hadd2
    //   '--'-+--+----.--.  .--.----'--'  |  |
    //        |  |   |aa|bb|cc|dd|        |  | extractf128 & insertf128
    //        +--+-->|a3|b3|c3|d3| <------+--+
    //               |a |b |c |d |

    const __m256d hadd1 = _mm256_hadd_pd(mul1, mul2);
    const __m256d hadd2 = _mm256_hadd_pd(mul3, mul4);

    const __m256d v12 = _mm256_insertf128_pd(
            hadd1, _mm256_extractf128_pd(hadd2, 0), 1);
    const __m256d v34 = _mm256_insertf128_pd(_mm256_castpd128_pd256(
            _mm256_extractf128_pd(hadd1, 1)),
            _mm256_extractf128_pd(hadd2, 1), 1);

    _mm256_store_pd(pack, _mm256_div_pd(_mm256_set1_pd(1.0),
                    _mm256_sqrt_pd(_mm256_add_pd(v12, v34))));
    return std::make_tuple(pack[0], pack[1], pack[2], pack[3]);
}

// regularize ----------------------------------------------------------------

template<>
MAVE_INLINE std::pair<matrix<double, 3, 1>, double>
regularize(const matrix<double, 3, 1>& v) noexcept
{
    const auto len = length(v);
    return std::make_pair(v * (1.0 / len), len);
}
template<>
MAVE_INLINE std::tuple<std::pair<matrix<double, 3, 1>, double>,
                       std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2
           ) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d hadd = _mm256_hadd_pd(mul1, mul2);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //         |  |  extract
    // |a0|b0|<+--'

    const __m128d len = _mm_sqrt_pd(_mm_add_pd(
        _mm256_castpd256_pd128(hadd), _mm256_extractf128_pd(hadd, 1)));

    alignas(16) double pack[4];
    _mm_store_pd(pack,   _mm_div_pd(_mm_set1_pd(1.0), len));
    _mm_store_pd(pack+2, len);

    return std::make_tuple(
        std::make_pair(_mm256_mul_pd(arg1, _mm256_set1_pd(pack[0])), pack[2]),
        std::make_pair(_mm256_mul_pd(arg2, _mm256_set1_pd(pack[1])), pack[3]));
}
template<>
MAVE_INLINE std::tuple<std::pair<matrix<double, 3, 1>, double>,
                       std::pair<matrix<double, 3, 1>, double>,
                       std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
           const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);

    // |a1|a2|a3|00| |b1|b2|b3|00| |c1|c2|c3|00| |c1|c2|c3|00|
    //  +--'  |  |    +--'  |  |    +--'  |  |    +--'  |  |
    //  |     +--'    |     |  |    |     +--'    |     |  |
    //  |  .--|--.----+-----+--'    |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| hadd1         |cc|cc|c3|c3| hadd2
    //   '--'-+--+----.--.  .--.----'--'  |  |
    //        |  |   |aa|bb|cc|cc|        |  | extractf128 & insertf128
    //        +--+-->|a3|b3|c3|c3| <------+--+
    //               |a |b |c |c |

    const __m256d hadd1 = _mm256_hadd_pd(mul1, mul2);
    const __m256d hadd2 = _mm256_hadd_pd(mul3, mul3);

    const __m256d v12 = _mm256_insertf128_pd(
            hadd1, _mm256_extractf128_pd(hadd2, 0), 1);
    const __m256d v34 = _mm256_insertf128_pd(_mm256_castpd128_pd256(
            _mm256_extractf128_pd(hadd1, 1)),
            _mm256_extractf128_pd(hadd2, 1), 1);

    const __m256d len = _mm256_sqrt_pd(_mm256_add_pd(v12, v34));
    _mm256_store_pd(pack, _mm256_div_pd(_mm256_set1_pd(1.0), len));

    const __m256d rv1 = _mm256_mul_pd(arg1, _mm256_set1_pd(pack[0]));
    const __m256d rv2 = _mm256_mul_pd(arg2, _mm256_set1_pd(pack[1]));
    const __m256d rv3 = _mm256_mul_pd(arg3, _mm256_set1_pd(pack[2]));

    _mm256_store_pd(pack, len);
    return std::make_tuple(std::make_pair(rv1, pack[0]),
                           std::make_pair(rv2, pack[1]),
                           std::make_pair(rv3, pack[2]));
}
template<>
MAVE_INLINE std::tuple<std::pair<matrix<double, 3, 1>, double>,
                       std::pair<matrix<double, 3, 1>, double>,
                       std::pair<matrix<double, 3, 1>, double>,
                       std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
           const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4
           ) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00| |c1|c2|c3|00| |d1|d2|d3|00|
    //  +--'  |  |    +--'  |  |    +--'  |  |    +--'  |  |
    //  |     +--'    |     |  |    |     +--'    |     |  |
    //  |  .--|--.----+-----+--'    |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| hadd1         |cc|dd|c3|d3| hadd2
    //   '--'-+--+----.--.  .--.----'--'  |  |
    //        |  |   |aa|bb|cc|dd|        |  | extractf128 & insertf128
    //        +--+-->|a3|b3|c3|d3| <------+--+
    //               |a |b |c |d |

    const __m256d hadd1 = _mm256_hadd_pd(mul1, mul2);
    const __m256d hadd2 = _mm256_hadd_pd(mul3, mul4);

    const __m256d v12 = _mm256_insertf128_pd(
            hadd1, _mm256_extractf128_pd(hadd2, 0), 1);
    const __m256d v34 = _mm256_insertf128_pd(_mm256_castpd128_pd256(
            _mm256_extractf128_pd(hadd1, 1)),
            _mm256_extractf128_pd(hadd2, 1), 1);

    const __m256d len = _mm256_sqrt_pd(_mm256_add_pd(v12, v34));

    _mm256_store_pd(pack, _mm256_div_pd(_mm256_set1_pd(1.0), len));

    const __m256d rv1 = _mm256_mul_pd(arg1, _mm256_set1_pd(pack[0]));
    const __m256d rv2 = _mm256_mul_pd(arg2, _mm256_set1_pd(pack[1]));
    const __m256d rv3 = _mm256_mul_pd(arg3, _mm256_set1_pd(pack[2]));
    const __m256d rv4 = _mm256_mul_pd(arg4, _mm256_set1_pd(pack[3]));

    _mm256_store_pd(pack, len);
    return std::make_tuple(std::make_pair(rv1, pack[0]),
                           std::make_pair(rv2, pack[1]),
                           std::make_pair(rv3, pack[2]),
                           std::make_pair(rv4, pack[3]));
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> max(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_max_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
max(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v2
    ) noexcept
{
    return std::make_tuple(max(std::get<0>(v1), std::get<0>(v2)),
                          max(std::get<1>(v1), std::get<1>(v2)));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
max(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&> v2) noexcept
{
    return std::make_tuple(max(std::get<0>(v1), std::get<0>(v2)),
                           max(std::get<1>(v1), std::get<1>(v2)),
                           max(std::get<2>(v1), std::get<2>(v2)));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
max(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&, const matrix<double,3,1>&> v2
    ) noexcept
{
    return std::make_tuple(max(std::get<0>(v1), std::get<0>(v2)),
                           max(std::get<1>(v1), std::get<1>(v2)),
                           max(std::get<2>(v1), std::get<2>(v2)),
                           max(std::get<3>(v1), std::get<3>(v2)));
}


template<>
MAVE_INLINE matrix<double, 3, 1> min(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_min_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
min(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&> v2
    ) noexcept
{
    return std::make_tuple(min(std::get<0>(v1), std::get<0>(v2)),
                          min(std::get<1>(v1), std::get<1>(v2)));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
min(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&> v2) noexcept
{
    return std::make_tuple(min(std::get<0>(v1), std::get<0>(v2)),
                           min(std::get<1>(v1), std::get<1>(v2)),
                           min(std::get<2>(v1), std::get<2>(v2)));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
min(std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&, const matrix<double,3,1>&> v1,
    std::tuple<const matrix<double,3,1>&, const matrix<double,3,1>&,
               const matrix<double,3,1>&, const matrix<double,3,1>&> v2
    ) noexcept
{
    return std::make_tuple(min(std::get<0>(v1), std::get<0>(v2)),
                           min(std::get<1>(v1), std::get<1>(v2)),
                           min(std::get<2>(v1), std::get<2>(v2)),
                           min(std::get<3>(v1), std::get<3>(v2)));
}

// floor ---------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> floor(const matrix<double, 3, 1>& v) noexcept
{
    return _mm256_floor_pd(_mm256_load_pd(v.data()));
}

template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
floor(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return std::make_tuple(floor(v1), floor(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
floor(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
      const matrix<double, 3, 1>& v3) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
floor(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
      const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3), floor(v4));
}

// ceil ----------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> ceil(const matrix<double, 3, 1>& v) noexcept
{
    return _mm256_ceil_pd(_mm256_load_pd(v.data()));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
ceil(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
ceil(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
     const matrix<double, 3, 1>& v3) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3));
}
template<>
MAVE_INLINE std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
ceil(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
     const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3), ceil(v4));
}

// ---------------------------------------------------------------------------
// dot_product
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE double dot_product(
    const matrix<double, 3, 1>& lhs, const matrix<double, 3, 1>& rhs) noexcept
{
    const matrix<double, 3, 1> sq(_mm256_mul_pd(
        _mm256_load_pd(lhs.data()), _mm256_load_pd(rhs.data())));
    return sq[0] + sq[1] + sq[2];
}
template<>
MAVE_INLINE std::tuple<double, double>
dot_product(
    std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> lhs,
    std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> rhs
    ) noexcept
{
    const __m256d mul1 = _mm256_mul_pd(_mm256_load_pd(std::get<0>(lhs).data()),
                                       _mm256_load_pd(std::get<0>(rhs).data()));
    const __m256d mul2 = _mm256_mul_pd(_mm256_load_pd(std::get<1>(lhs).data()),
                                       _mm256_load_pd(std::get<1>(rhs).data()));
    const __m256d hadd = _mm256_hadd_pd(mul1, mul2);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //         |  |  extract
    // |a0|b0|<+--'

    alignas(16) double pack[2];
    _mm_store_pd(pack, _mm_add_pd(
        _mm256_castpd256_pd128(hadd), _mm256_extractf128_pd(hadd, 1)));

    return std::make_tuple(pack[0], pack[1]);
}

template<>
MAVE_INLINE std::tuple<double, double, double>
dot_product(
    std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
               const matrix<double, 3, 1>&> lhs,
    std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
               const matrix<double, 3, 1>&> rhs) noexcept
{
    const auto d12 = dot_product(std::tie(std::get<0>(lhs), std::get<1>(lhs)),
                                 std::tie(std::get<0>(rhs), std::get<1>(rhs)));
    return std::make_tuple(std::get<0>(d12), std::get<1>(d12),
                           dot_product(std::get<2>(lhs), std::get<2>(rhs)));
}

template<>
MAVE_INLINE std::tuple<double, double, double, double>
dot_product(
    std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
               const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> lhs,
    std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
               const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> rhs
    ) noexcept
{
    const __m256d mul1 = _mm256_mul_pd(_mm256_load_pd(std::get<0>(lhs).data()),
                                       _mm256_load_pd(std::get<0>(rhs).data()));
    const __m256d mul2 = _mm256_mul_pd(_mm256_load_pd(std::get<1>(lhs).data()),
                                       _mm256_load_pd(std::get<1>(rhs).data()));
    const __m256d mul3 = _mm256_mul_pd(_mm256_load_pd(std::get<2>(lhs).data()),
                                       _mm256_load_pd(std::get<2>(rhs).data()));
    const __m256d mul4 = _mm256_mul_pd(_mm256_load_pd(std::get<3>(lhs).data()),
                                       _mm256_load_pd(std::get<3>(rhs).data()));
    // |a1|a2|a3|00| |b1|b2|b3|00| |c1|c2|c3|00| |d1|d2|d3|00|
    //  +--'  |  |    +--'  |  |    +--'  |  |    +--'  |  |
    //  |     +--'    |     |  |    |     +--'    |     |  |
    //  |  .--|--.----+-----+--'    |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| hadd1         |cc|dd|c3|d3| hadd2
    //   '--'-+--+----.--.  .--.----'--'  |  |
    //        |  |   |aa|bb|cc|dd|        |  | extractf128 & insertf128
    //        +--+-->|a3|b3|c3|d3| <------+--+
    //               |a |b |c |d |

    const __m256d hadd1 = _mm256_hadd_pd(mul1, mul2);
    const __m256d hadd2 = _mm256_hadd_pd(mul3, mul4);

    const __m256d v12 = _mm256_insertf128_pd(
            hadd1, _mm256_extractf128_pd(hadd2, 0), 1);
    const __m256d v34 = _mm256_insertf128_pd(_mm256_castpd128_pd256(
            _mm256_extractf128_pd(hadd1, 1)),
            _mm256_extractf128_pd(hadd2, 1), 1);

    alignas(32) double pack[4];
    _mm256_store_pd(pack, _mm256_add_pd(v12, v34));
    return std::make_tuple(pack[0], pack[1], pack[2], pack[3]);
}

// -------------------------------------------------------------------------
// cross_product
// -------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> cross_product(
    const matrix<double, 3, 1>& x, const matrix<double, 3, 1>& y) noexcept
{
    const __m256d y_ = _mm256_set_pd(0.0, y[0], y[2], y[1]);
    const __m256d x_ = _mm256_set_pd(0.0, x[0], x[2], x[1]);

    const matrix<double, 3, 1> tmp(_mm256_sub_pd(
            _mm256_mul_pd(_mm256_load_pd(x.data()), y_),
            _mm256_mul_pd(_mm256_load_pd(y.data()), x_)));

    return matrix<double, 3, 1>(tmp[1], tmp[2], tmp[0]);
}

} // mave
#endif // MAVE_MATH_MATRIX_HPP
