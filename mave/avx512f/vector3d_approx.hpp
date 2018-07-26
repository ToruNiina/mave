#ifndef MAVE_AVX512F_VECTOR3_DOUBLE_APPROX_HPP
#define MAVE_AVX512F_VECTOR3_DOUBLE_APPROX_HPP

#ifndef __AVX512F__
#error "mave/avx512f/vector3d.hpp requires avx512f support but __AVX512F__ is not defined."
#endif

#ifndef MAVE_VECTOR_HPP
#error "do not use mave/avx/vector3d_approx.hpp alone. please include mave/vector.hpp."
#endif

#ifdef MAVE_VECTOR3_DOUBLE_IMPLEMENTATION
#error "specialization of vector for 3x double is already defined"
#endif

#define MAVE_VECTOR3_DOUBLE_IMPLEMENTATION "avx512f-approx"

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
    using storage_type    = std::array<double, 4>;
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

    matrix(): vs_{{0.0, 0.0, 0.0, 0.0}}{};
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

template<>
inline matrix<double, 3, 1> operator-(const matrix<double, 3, 1>& v) noexcept
{
    return _mm256_sub_pd(_mm256_setzero_pd(), _mm256_load_pd(v.data()));
}
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
    const __m256d arg = _mm256_load_pd(v.data());
    alignas(32) double pack[4];
    _mm256_store_pd(pack, _mm256_mul_pd(arg, arg));
    return pack[0] + pack[1] + pack[2];
}

template<>
inline std::pair<double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    alignas(16) double pack[2];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  +--'    +--'  |  |
    //  |  .--|-------'     |  | hadd
    //  |  |  |  .----------+--'
    // |aa|bb|a3|b3| hadd
    //  |  |   |  |
    //  v  v   |  |
    // |aa|bb| |  | extractf128_pd
    // |a3|b3|<+--+

    const __m256d hadd = _mm256_hadd_pd(
        _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2));

    _mm_store_pd(pack, _mm_add_pd(_mm256_extractf128_pd(hadd, 0),
                                  _mm256_extractf128_pd(hadd, 1)));
    return std::make_pair(pack[0], pack[1]);
}

template<>
inline std::tuple<double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul3x = _mm512_castpd256_pd512(_mm256_mul_pd(arg3,  arg3));

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|xx|xx|xx|xx|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  3  1  5  9  7  _mm512_permutex2var_pd
    // |a1|b1|c1|00|a2|b2|c2|00|
    //   2  6  A  3  2  6  A  3  _mm512_permutex2var_pd
    // |a3|b3|c3|00|a3|b3|c3|00|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00), mul3x);
    const __m512d abc3x = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x03, 0x0A, 0x06, 0x02, 0x03, 0x0A, 0x06, 0x02), mul3x);

    _mm256_store_pd(pack, _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc3x)));
    return std::make_tuple(pack[0], pack[1], pack[2]);
}

template<>
inline std::tuple<double, double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);
    const __m512d arg34 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg3), arg4, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul34 = _mm512_mul_pd(arg34, arg34);

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|d1|d2|d3|00|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  C  1  5  9  D  _mm512_permutex2var_pd
    // |a1|b1|c1|d1|a2|b2|c2|d2|
    //   2  6  A  E  2  6  A  E  _mm512_permutex2var_pd
    // |a3|b3|c3|d3|a3|b3|c3|d3|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00), mul34);
    const __m512d abc3x = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0E, 0x0A, 0x06, 0x02, 0x0E, 0x0A, 0x06, 0x02), mul34);

    _mm256_store_pd(pack, _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc3x)));

    return std::make_tuple(pack[0], pack[1], pack[2], pack[3]);
}

// length --------------------------------------------------------------------

template<>
inline double length(const matrix<double, 3, 1>& v) noexcept
{
    const double lsq = length_sq(v);
    return lsq * _mm_cvtsd_f64(_mm_rsqrt14_sd(_mm_undefined_pd(), _mm_set_sd(lsq)));
}

template<>
inline std::pair<double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    alignas(16) double pack[2];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack
    //  |  |   |  |
    //  v  v   |  |
    // |aa|bb| |  | extractf128_pd
    // |a3|b3|<+--+

    const __m256d hadd = _mm256_hadd_pd(
        _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2));

    const __m128d lsq = _mm_add_pd(
        _mm256_extractf128_pd(hadd, 0), _mm256_extractf128_pd(hadd, 1));

    _mm_store_pd(pack, _mm_mul_pd(lsq, _mm512_castpd512_pd128(
                 _mm512_rsqrt14_pd(_mm512_castpd128_pd512(lsq)))));

    return std::make_pair(pack[0], pack[1]);
}

template<>
inline std::tuple<double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul3x = _mm512_castpd256_pd512(_mm256_mul_pd(arg3,  arg3));

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|xx|xx|xx|xx|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  3  1  5  9  7  _mm512_permutex2var_pd
    // |a1|b1|c1|00|a2|b2|c2|00|
    //   2  6  A  3  2  6  A  3  _mm512_permutex2var_pd
    // |a3|b3|c3|00|a3|b3|c3|00|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00), mul3x);
    const __m512d abc3x = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x03, 0x0A, 0x06, 0x02, 0x03, 0x0A, 0x06, 0x02), mul3x);

    const __m256d lsq = _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc3x));

    _mm256_store_pd(pack, _mm256_mul_pd(lsq, _mm512_castpd512_pd256(
                    _mm512_rsqrt14_pd(_mm512_castpd256_pd512(lsq)))));

    return std::make_tuple(pack[0], pack[1], pack[2]);
}

template<>
inline std::tuple<double, double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);
    const __m512d arg34 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg3), arg4, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul34 = _mm512_mul_pd(arg34, arg34);

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|d1|d2|d3|00|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  C  1  5  9  D  _mm512_permutex2var_pd
    // |a1|b1|c1|d1|a2|b2|c2|d2|
    //   2  6  A  E  2  6  A  E  _mm512_permutex2var_pd
    // |a3|b3|c3|d3|a3|b3|c3|d3|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00), mul34);
    const __m512d abc34 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0E, 0x0A, 0x06, 0x02, 0x0E, 0x0A, 0x06, 0x02), mul34);

    const __m256d lsq = _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc34));

    _mm256_store_pd(pack, _mm256_mul_pd(lsq, _mm512_castpd512_pd256(
                    _mm512_rsqrt14_pd(_mm512_castpd256_pd512(lsq)))));

    return std::make_tuple(pack[0], pack[1], pack[2], pack[3]);
}

// rlength -------------------------------------------------------------------

template<>
inline double rlength(const matrix<double, 3, 1>& v) noexcept
{
    const double lsq = length_sq(v);
    return _mm_cvtsd_f64(_mm_rsqrt14_sd(_mm_undefined_pd(), _mm_set_sd(lsq)));
}
template<>
inline std::pair<double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    alignas(16) double pack[2];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  +--'    +--'  |  |
    //  |  .--|-------'     |  | hadd
    //  |  |  |  .----------+--'
    // |aa|bb|a3|b3| hadd
    //  |  |   |  |
    //  v  v   |  |
    // |aa|bb| |  | extractf128_pd
    // |a3|b3|<+--+

    const __m256d hadd = _mm256_hadd_pd(
        _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2));

    const __m128d lsq = _mm_add_pd(
        _mm256_extractf128_pd(hadd, 0), _mm256_extractf128_pd(hadd, 1));

    _mm_store_pd(pack, _mm512_castpd512_pd128(
                 _mm512_rsqrt14_pd(_mm512_castpd128_pd512(lsq))));
    return std::make_pair(pack[0], pack[1]);
}
template<>
inline std::tuple<double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul3x = _mm512_castpd256_pd512(_mm256_mul_pd(arg3,  arg3));

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|xx|xx|xx|xx|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  3  1  5  9  7  _mm512_permutex2var_pd
    // |a1|b1|c1|00|a2|b2|c2|00|
    //   2  6  A  3  2  6  A  3  _mm512_permutex2var_pd
    // |a3|b3|c3|00|a3|b3|c3|00|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00), mul3x);
    const __m512d abc3x = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x03, 0x0A, 0x06, 0x02, 0x03, 0x0A, 0x06, 0x02), mul3x);

    const __m256d lsq = _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc3x));

    _mm256_store_pd(pack, _mm512_castpd512_pd256(
                    _mm512_rsqrt14_pd(_mm512_castpd256_pd512(lsq))));

    return std::make_tuple(pack[0], pack[1], pack[2]);
}
template<>
inline std::tuple<double, double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);
    const __m512d arg34 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg3), arg4, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul34 = _mm512_mul_pd(arg34, arg34);

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|d1|d2|d3|00|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  C  1  5  9  D  _mm512_permutex2var_pd
    // |a1|b1|c1|d1|a2|b2|c2|d2|
    //   2  6  A  E  2  6  A  E  _mm512_permutex2var_pd
    // |a3|b3|c3|d3|a3|b3|c3|d3|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00), mul34);
    const __m512d abc34 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0E, 0x0A, 0x06, 0x02, 0x0E, 0x0A, 0x06, 0x02), mul34);

    const __m256d lsq = _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc34));

    _mm256_store_pd(pack, _mm512_castpd512_pd256(
                    _mm512_rsqrt14_pd(_mm512_castpd256_pd512(lsq))));

    return std::make_tuple(pack[0], pack[1], pack[2], pack[3]);
}

// regularize ----------------------------------------------------------------

template<>
inline std::pair<matrix<double, 3, 1>, double>
regularize(const matrix<double, 3, 1>& v) noexcept
{
    const double lsq = length_sq(v);
    const double rln = _mm_cvtsd_f64(
            _mm_rsqrt14_sd(_mm_undefined_pd(), _mm_set_sd(lsq)));
    return std::make_pair(v * rln, lsq * rln);
}
template<>
inline std::pair<std::pair<matrix<double, 3, 1>, double>,
                 std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2
           ) noexcept
{
    alignas(16) double pack[2];

    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack
    //  |  |   |  |
    //  v  v   |  |
    // |aa|bb| |  | extractf128_pd
    // |a3|b3|<+--+

    const __m256d hadd = _mm256_hadd_pd(
            _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2));
    const __m128d lnsq = _mm_add_pd(
            _mm256_extractf128_pd(hadd, 0), _mm256_extractf128_pd(hadd, 1));
    const __m128d rlen = _mm512_castpd512_pd128(
            _mm512_rsqrt14_pd(_mm512_castpd128_pd512(lnsq)));

    _mm_store_pd(pack, rlen);

    const __m256d rv1 = _mm256_mul_pd(arg1, _mm256_set1_pd(pack[0]));
    const __m256d rv2 = _mm256_mul_pd(arg2, _mm256_set1_pd(pack[1]));

    _mm_store_pd(pack, _mm_mul_pd(rlen, lnsq));

    return std::make_pair(std::make_pair(matrix<double, 3, 1>(rv1), pack[0]),
                          std::make_pair(matrix<double, 3, 1>(rv2), pack[1]));
}
template<>
inline std::tuple<std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
           const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul3x = _mm512_castpd256_pd512(_mm256_mul_pd(arg3,  arg3));

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|xx|xx|xx|xx|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  3  1  5  9  7  _mm512_permutex2var_pd
    // |a1|b1|c1|00|a2|b2|c2|00|
    //   2  6  A  3  2  6  A  3  _mm512_permutex2var_pd
    // |a3|b3|c3|00|a3|b3|c3|00|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00), mul3x);
    const __m512d abc3x = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x03, 0x0A, 0x06, 0x02, 0x03, 0x0A, 0x06, 0x02), mul3x);

    const __m256d lnsq = _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc3x));

    const __m256d rlen = _mm512_castpd512_pd256(
            _mm512_rsqrt14_pd(_mm512_castpd256_pd512(lnsq)));

    _mm256_store_pd(pack, rlen);
    const __m256d rv1 = _mm256_mul_pd(arg1, _mm256_set1_pd(pack[0]));
    const __m256d rv2 = _mm256_mul_pd(arg2, _mm256_set1_pd(pack[1]));
    const __m256d rv3 = _mm256_mul_pd(arg3, _mm256_set1_pd(pack[2]));

    _mm256_store_pd(pack, _mm256_mul_pd(lnsq, rlen));

    return std::make_tuple(std::make_pair(matrix<double, 3, 1>(rv1), pack[0]),
                           std::make_pair(matrix<double, 3, 1>(rv2), pack[1]),
                           std::make_pair(matrix<double, 3, 1>(rv3), pack[2]));
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
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m512d arg12 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg1), arg2, 1);
    const __m512d arg34 = _mm512_insertf64x4(_mm512_castpd256_pd512(arg3), arg4, 1);

    const __m512d mul12 = _mm512_mul_pd(arg12, arg12);
    const __m512d mul34 = _mm512_mul_pd(arg34, arg34);

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|d1|d2|d3|00|
    //   0  1  2  3  4  5  6  7    8  9  A  B  C  D  E  F
    //
    //   0  4  8  C  1  5  9  D  _mm512_permutex2var_pd
    // |a1|b1|c1|d1|a2|b2|c2|d2|
    //   2  6  A  E  2  6  A  E  _mm512_permutex2var_pd
    // |a3|b3|c3|d3|a3|b3|c3|d3|

    const __m512d abc12 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00), mul34);
    const __m512d abc34 = _mm512_permutex2var_pd(mul12,_mm512_set_epi64(
            0x0E, 0x0A, 0x06, 0x02, 0x0E, 0x0A, 0x06, 0x02), mul34);

    const __m256d lnsq = _mm256_add_pd(_mm256_add_pd(
            _mm512_castpd512_pd256(abc12), _mm512_extractf64x4_pd(abc12, 1)
            ), _mm512_castpd512_pd256(abc34));
    const __m256d rlen = _mm512_castpd512_pd256(
            _mm512_rsqrt14_pd(_mm512_castpd256_pd512(lnsq)));

    _mm256_store_pd(pack, rlen);
    const __m256d rv1 = _mm256_mul_pd(arg1, _mm256_set1_pd(pack[0]));
    const __m256d rv2 = _mm256_mul_pd(arg2, _mm256_set1_pd(pack[1]));
    const __m256d rv3 = _mm256_mul_pd(arg3, _mm256_set1_pd(pack[2]));
    const __m256d rv4 = _mm256_mul_pd(arg4, _mm256_set1_pd(pack[3]));

    _mm256_store_pd(pack, _mm256_mul_pd(lnsq, rlen));
    return std::make_tuple(std::make_pair(matrix<double, 3, 1>(rv1), pack[0]),
                           std::make_pair(matrix<double, 3, 1>(rv2), pack[1]),
                           std::make_pair(matrix<double, 3, 1>(rv3), pack[2]),
                           std::make_pair(matrix<double, 3, 1>(rv4), pack[3]));
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
    alignas(32) double pack[4];
    _mm256_store_pd(pack, _mm256_mul_pd(
        _mm256_load_pd(lhs.data()), _mm256_load_pd(rhs.data())));
    return pack[0] + pack[1] + pack[2];
}

template<>
inline matrix<double, 3, 1> cross_product(
    const matrix<double, 3, 1>& x, const matrix<double, 3, 1>& y) noexcept
{
    const __m256d arg1 = _mm256_load_pd(x.data());
    const __m256d arg2 = _mm256_load_pd(y.data());

    // 3 0 2 1 --> 0b 11 00 10 01 == 201
    const __m256d y_ = _mm256_permute4x64_pd(arg1, 201u);
    const __m256d x_ = _mm256_permute4x64_pd(arg2, 201u);

    const __m256d z =
#ifdef __FMA__
        _mm256_fmsub_pd(arg1, y_, _mm256_mul_pd(arg2, x_));
#else
        _mm256_sub_pd(_mm256_mul_pd(arg1, y_), _mm256_mul_pd(arg2, x_));
#endif

    return _mm256_permute4x64_pd(z, 201u);
}

template<>
inline double scalar_triple_product(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    // 3 0 2 1 --> 0b 11 00 10 01 == 201
    const __m256d y_ = _mm256_permute4x64_pd(arg1, 201u);
    const __m256d x_ = _mm256_permute4x64_pd(arg2, 201u);

    _mm256_store_pd(pack, _mm256_mul_pd(_mm256_permute4x64_pd(arg3, 201u),
#ifdef __FMA__
        _mm256_fmsub_pd(arg1, y_, _mm256_mul_pd(arg2, x_))
#else
        _mm256_sub_pd(_mm256_mul_pd(arg1, y_), _mm256_mul_pd(arg2, x_))
#endif
        ));

    return pack[0] + pack[1] + pack[2];
}

} // mave
#endif // MAVE_MATH_MATRIX_HPP
