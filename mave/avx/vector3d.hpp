#ifndef MAVE_AVX_VECTOR3_DOUBLE_HPP
#define MAVE_AVX_VECTOR3_DOUBLE_HPP

#ifndef __AVX__
#error "mave/avx/vector3d.hpp requires avx support but __AVX__ is not defined."
#endif

#ifndef MAVE_VECTOR_HPP
#error "do not use mave/avx/vector3d.hpp alone. please include mave/vector.hpp."
#endif

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

    matrix(): vs_{{0.0, 0.0, 0.0, 0.0}} {}
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

    reference       at(size_type, size_type j)       {return vs_.at(j);}
    const_reference at(size_type, size_type j) const {return vs_.at(j);}
    reference       operator()(size_type, size_type j)       noexcept {return vs_[j];}
    const_reference operator()(size_type, size_type j) const noexcept {return vs_[j];}

  private:
    alignas(32) storage_type vs_;
};

template<>
inline matrix<double, 3, 1> operator+(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_add_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}
template<>
inline matrix<double, 3, 1> operator-(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_sub_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}
template<>
inline matrix<double, 3, 1> operator*(
    const double v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_mul_pd(_mm256_set1_pd(v1), _mm256_load_pd(v2.data()));
}
template<>
inline matrix<double, 3, 1> operator*(
    const matrix<double, 3, 1>& v1, const double v2) noexcept
{
    return _mm256_mul_pd(_mm256_load_pd(v1.data()), _mm256_set1_pd(v2));
}
template<>
inline matrix<double, 3, 1> operator/(
    const matrix<double, 3, 1>& v1, const double v2) noexcept
{
    return _mm256_div_pd(_mm256_load_pd(v1.data()), _mm256_set1_pd(v2));
}

// ---------------------------------------------------------------------------
// length
// ---------------------------------------------------------------------------

// length_sq -----------------------------------------------------------------

template<>
inline double length_sq(const matrix<double, 3, 1>& v) noexcept
{
    const matrix<double, 3, 1> sq(_mm256_mul_pd(
        _mm256_load_pd(v.data()), _mm256_load_pd(v.data())));
    return sq[0] + sq[1] + sq[2];
}

template<>
inline std::pair<double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    // to assure the 4th value is 0, mask it
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    const matrix<double, 3, 1> hadd(_mm256_hadd_pd(mul1, mul2));
    return std::make_pair(hadd[0] + hadd[2], hadd[1] + hadd[3]);
}

template<>
inline std::tuple<double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    // to assure the 4th value is 0, mask it
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);

    // |ax|ay|az|00| |bx|by|bz|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |aa|bb|a0|b0|
    //  |   \ /  |   at AVX, there are no equivalent operation.
    //  |    X   |   after AVX2, there is `_mm_permute4x64_pd`.
    //  |   / \  |
    // |aa|a0|bb|b0| |c1|c2|c3|00|
    //  |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'
    //  |  .--|-------'     |
    //  |  |  |  .----------'
    // |a2|cc|b2|c0|

    const matrix<double, 3, 1> hadd1(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd2(_mm256_hadd_pd(
        _mm256_set_pd(hadd1[3], hadd1[1], hadd1[2], hadd1[0]), mul3));

    return std::make_tuple(hadd2[0], hadd2[2], hadd2[1] + hadd2[3]);
}

template<>
inline std::tuple<double, double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    // to assure the 4th value is 0, mask it
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);
    const __m256d arg4 = _mm256_maskload_pd(v4.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |ax|ay|az|00| |bx|by|bz|00| |cx|cy|cz|00| |dx|dy|dz|00|
    //  |  |  |  |    |  |  |  |    |  |  |  |    |  |  |  |
    //  +--'  +--'    +--'  +--'    +--'  +--'    +--'  +--'
    //  |  .--|-------'     |       |  .--|-------'     |
    //  |  |  |  .----------'       |  |  |  .----------'
    // |aa|bb|a0|b0|               |cc|dd|c0|d0|

    const matrix<double, 3, 1> hadd12(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd34(_mm256_hadd_pd(mul3, mul4));

    const matrix<double, 3, 1> hadd(_mm256_hadd_pd(
        _mm256_set_pd(hadd12[3], hadd12[1], hadd12[2], hadd12[0]),
        _mm256_set_pd(hadd34[3], hadd34[1], hadd34[2], hadd34[0])));

    return std::make_tuple(hadd[0], hadd[1], hadd[2], hadd[3]);
}

template<>
inline double length(const matrix<double, 3, 1>& v) noexcept
{
    return std::sqrt(length_sq(v));
}

template<>
inline std::pair<double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    const matrix<double, 3, 1> hadd(_mm256_hadd_pd(mul1, mul2));
    const __m128d lensq = _mm_set_pd(hadd[1] + hadd[3], hadd[0] + hadd[2]);
    alignas(16) double len[2];
    _mm_store_pd(len, _mm_sqrt_pd(lensq));

    return std::make_pair(len[0], len[1]);
}

template<>
inline std::tuple<double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);

    const matrix<double, 3, 1> hadd1(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd2(_mm256_hadd_pd(
        _mm256_set_pd(hadd1[3], hadd1[1], hadd1[2], hadd1[0]), mul3));

    const matrix<double, 3, 1> retval(_mm256_sqrt_pd(
        _mm256_set_pd(0.0, hadd2[1] + hadd2[3], hadd2[2], hadd2[0])));

    return std::make_tuple(retval[0], retval[1], retval[2]);
}

template<>
inline std::tuple<double, double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);
    const __m256d arg4 = _mm256_maskload_pd(v4.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    const matrix<double, 3, 1> hadd12(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd34(_mm256_hadd_pd(mul3, mul4));

    const matrix<double, 3, 1> retval = _mm256_sqrt_pd(_mm256_hadd_pd(
        _mm256_set_pd(hadd12[3], hadd12[1], hadd12[2], hadd12[0]),
        _mm256_set_pd(hadd34[3], hadd34[1], hadd34[2], hadd34[0])));

    return std::make_tuple(retval[0], retval[1], retval[2], retval[3]);
}

// rlength -------------------------------------------------------------------

template<>
inline double rlength(const matrix<double, 3, 1>& v) noexcept
{
    return 1.0 / std::sqrt(length_sq(v));
}
template<>
inline std::pair<double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    const matrix<double, 3, 1> hadd(_mm256_hadd_pd(mul1, mul2));
    const __m128d lensq = _mm_set_pd(hadd[1] + hadd[3], hadd[0] + hadd[2]);
    alignas(16) double len[2];
    _mm_store_pd(len, _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(lensq)));
    return std::make_pair(len[0], len[1]);
}
template<>
inline std::tuple<double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);

    const matrix<double, 3, 1> hadd1(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd2(_mm256_hadd_pd(
        _mm256_set_pd(hadd1[3], hadd1[1], hadd1[2], hadd1[0]), mul3));

    const matrix<double, 3, 1> retval(_mm256_div_pd(
        _mm256_set1_pd(1.0), _mm256_sqrt_pd(
            _mm256_set_pd(0.0, hadd2[1] + hadd2[3], hadd2[2], hadd2[0]))));
    return std::make_tuple(retval[0], retval[1], retval[2]);
}
template<>
inline std::tuple<double, double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);
    const __m256d arg4 = _mm256_maskload_pd(v4.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    const matrix<double, 3, 1> hadd12(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd34(_mm256_hadd_pd(mul3, mul4));

    const matrix<double, 3, 1> retval = _mm256_div_pd(_mm256_set1_pd(1.0),
        _mm256_sqrt_pd(_mm256_hadd_pd(
            _mm256_set_pd(hadd12[3], hadd12[1], hadd12[2], hadd12[0]),
            _mm256_set_pd(hadd34[3], hadd34[1], hadd34[2], hadd34[0]))));
    return std::make_tuple(retval[0], retval[1], retval[2], retval[3]);
}

// regularize ----------------------------------------------------------------

template<>
inline std::pair<matrix<double, 3, 1>, double>
regularize(const matrix<double, 3, 1>& v) noexcept
{
    const auto len = length(v);
    return std::make_pair(v * (1.0 / len), len);
}
template<>
inline std::pair<std::pair<matrix<double, 3, 1>, double>,
                 std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2
           ) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    const matrix<double, 3, 1> hadd(_mm256_hadd_pd(mul1, mul2));
    const double l1 = std::sqrt(hadd[0] + hadd[2]);
    const double l2 = std::sqrt(hadd[1] + hadd[3]);

    const matrix<double, 3, 1> rv1 = _mm256_div_pd(arg1, _mm256_set1_pd(l1));
    const matrix<double, 3, 1> rv2 = _mm256_div_pd(arg2, _mm256_set1_pd(l2));

    return std::make_pair(std::make_pair(rv1, l1), std::make_pair(rv2, l2));
}
template<>
inline std::tuple<std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
           const matrix<double, 3, 1>& v3) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);

    const matrix<double, 3, 1> hadd1(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd2(_mm256_hadd_pd(
        _mm256_set_pd(hadd1[3], hadd1[1], hadd1[2], hadd1[0]), mul3));

    const double l1 = std::sqrt(hadd2[0]);
    const double l2 = std::sqrt(hadd2[2]);
    const double l3 = std::sqrt(hadd2[1] + hadd2[3]);

    const matrix<double, 3, 1> rv1 = _mm256_div_pd(arg1, _mm256_set1_pd(l1));
    const matrix<double, 3, 1> rv2 = _mm256_div_pd(arg2, _mm256_set1_pd(l2));
    const matrix<double, 3, 1> rv3 = _mm256_div_pd(arg3, _mm256_set1_pd(l3));

    return std::make_tuple(std::make_pair(rv1, l1), std::make_pair(rv2, l2),
                           std::make_pair(rv3, l3));
}
template<>
inline std::tuple<std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
           const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4
           ) noexcept
{
    const __m256i mask = _mm256_set_epi64x(0, 1, 1, 1);

    const __m256d arg1 = _mm256_maskload_pd(v1.data(), mask);
    const __m256d arg2 = _mm256_maskload_pd(v2.data(), mask);
    const __m256d arg3 = _mm256_maskload_pd(v3.data(), mask);
    const __m256d arg4 = _mm256_maskload_pd(v4.data(), mask);

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    const matrix<double, 3, 1> hadd12(_mm256_hadd_pd(mul1, mul2));
    const matrix<double, 3, 1> hadd34(_mm256_hadd_pd(mul3, mul4));

    const matrix<double, 3, 1> hadd(_mm256_hadd_pd(
        _mm256_set_pd(hadd12[3], hadd12[1], hadd12[2], hadd12[0]),
        _mm256_set_pd(hadd34[3], hadd34[1], hadd34[2], hadd34[0])));

    const double l1 = std::sqrt(hadd[0]);
    const double l2 = std::sqrt(hadd[1]);
    const double l3 = std::sqrt(hadd[2]);
    const double l4 = std::sqrt(hadd[3]);

    const matrix<double, 3, 1> rv1 = _mm256_div_pd(arg1, _mm256_set1_pd(l1));
    const matrix<double, 3, 1> rv2 = _mm256_div_pd(arg2, _mm256_set1_pd(l2));
    const matrix<double, 3, 1> rv3 = _mm256_div_pd(arg3, _mm256_set1_pd(l3));
    const matrix<double, 3, 1> rv4 = _mm256_div_pd(arg4, _mm256_set1_pd(l4));

    return std::make_pair(std::make_pair(rv1, l1), std::make_pair(rv2, l2),
                          std::make_pair(rv3, l3), std::make_pair(rv4, l4));
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<>
inline matrix<double, 3, 1> max(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_max_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}

template<>
inline matrix<double, 3, 1> min(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return _mm256_min_pd(_mm256_load_pd(v1.data()), _mm256_load_pd(v2.data()));
}

// floor ---------------------------------------------------------------------

template<>
inline matrix<double, 3, 1> floor(const matrix<double, 3, 1>& v) noexcept
{
    return _mm256_floor_pd(_mm256_load_pd(v.data()));
}

template<>
inline std::pair<matrix<double, 3, 1>, matrix<double, 3, 1>>
floor(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return std::make_pair(floor(v1), floor(v2));
}
template<>
inline std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
floor(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
      const matrix<double, 3, 1>& v3) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3));
}
template<>
inline std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
floor(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
      const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3), floor(v4));
}

// ceil ----------------------------------------------------------------------

template<>
inline matrix<double, 3, 1> ceil(const matrix<double, 3, 1>& v) noexcept
{
    return _mm256_ceil_pd(_mm256_load_pd(v.data()));
}
template<>
inline std::pair<matrix<double, 3, 1>, matrix<double, 3, 1>>
ceil(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    return std::make_pair(ceil(v1), ceil(v2));
}
template<>
inline std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>>
ceil(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
     const matrix<double, 3, 1>& v3) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3));
}
template<>
inline std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>,
                  matrix<double, 3, 1>, matrix<double, 3, 1>>
ceil(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
     const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3), ceil(v4));
}

// ---------------------------------------------------------------------------

template<>
inline double dot_product(
    const matrix<double, 3, 1>& lhs, const matrix<double, 3, 1>& rhs) noexcept
{
    const matrix<double, 3, 1> sq(_mm256_mul_pd(
        _mm256_load_pd(lhs.data()), _mm256_load_pd(rhs.data())));
    return sq[0] + sq[1] + sq[2];
}

template<>
inline matrix<double, 3, 1> cross_product(
    const matrix<double, 3, 1>& x, const matrix<double, 3, 1>& y) noexcept
{
    const __m256d y_ = _mm256_set_pd(0.0, y[0], y[2], y[1]);
    const __m256d x_ = _mm256_set_pd(0.0, x[0], x[2], x[1]);

    const matrix<double, 3, 1> tmp(_mm256_sub_pd(
            _mm256_mul_pd(_mm256_load_pd(x.data()), y_),
            _mm256_mul_pd(_mm256_load_pd(y.data()), x_)));

    return matrix<double, 3, 1>(tmp[1], tmp[2], tmp[0]);
}

template<>
inline double scalar_triple_product(
    const matrix<double, 3, 1>& lhs, const matrix<double, 3, 1>& mid,
    const matrix<double, 3, 1>& rhs) noexcept
{
    return (lhs[1] * mid[2] - lhs[2] * mid[1]) * rhs[0] +
           (lhs[2] * mid[0] - lhs[0] * mid[2]) * rhs[1] +
           (lhs[0] * mid[1] - lhs[1] * mid[0]) * rhs[2];
}

} // mave
#endif // MAVE_MATH_MATRIX_HPP
