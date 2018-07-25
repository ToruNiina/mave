#ifndef MAVE_AVX2_VECTOR3_FLOAT_APPROX_HPP
#define MAVE_AVX2_VECTOR3_FLOAT_APPROX_HPP

#ifndef __AVX2__
#error "mave/avx2/vector3f.hpp requires avx support but __AVX2__ is not defined."
#endif

#ifdef MAVE_VECTOR3_FLOAT_IMPLEMENTATION
#error "specialization of vector for 3x float is already defined"
#endif
#define MAVE_VECTOR3_FLOAT_IMPLEMENTATION "avx2"

#include <x86intrin.h>
#include <type_traits>
#include <array>
#include <cmath>

namespace mave
{

template<>
struct alignas(16) matrix<float, 3, 1>
{
    static constexpr std::size_t alignment   = 16;
    static constexpr std::size_t row_size    = 3;
    static constexpr std::size_t column_size = 1;
    static constexpr std::size_t total_size  = 3;
    using value_type      = float;
    using storage_type    = std::array<value_type, 4>;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;

    template<typename T1, typename T2, typename T3>
    matrix(T1&& v1, T2&& v2, T3&& v3) noexcept
        : vs_{{static_cast<value_type>(v1), static_cast<value_type>(v2),
               static_cast<value_type>(v3), 0.0f}}
    {}

    matrix(__m128 pack) noexcept
    {
        _mm_store_ps(this->data(), pack);
    }
    matrix& operator=(__m128 pack) noexcept
    {
        _mm_store_ps(this->data(), pack);
        return *this;
    }

    matrix(__m256d pack) noexcept
    {
        _mm_store_ps(this->data(), _mm256_cvtpd_ps(pack));
    }
    matrix& operator=(__m256d pack) noexcept
    {
        _mm_store_ps(this->data(), _mm256_cvtpd_ps(pack));
        return *this;
    }

    matrix(__m128i pack) noexcept
    {
        _mm_store_ps(this->data(), _mm_cvtepi32_ps(pack));
    }
    matrix& operator=(__m128i pack) noexcept
    {
        _mm_store_ps(this->data(), _mm_cvtepi32_ps(pack));
        return *this;
    }

    matrix(): vs_{{0.0f, 0.0f, 0.0f, 0.0f}}{}
    ~matrix() = default;
    matrix(const matrix&) = default;
    matrix(matrix&&)      = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&)      = default;

    template<typename T>
    matrix& operator=(const matrix<T, 3, 1>& rhs) noexcept
    {
        vs_[0] = static_cast<float>(rhs[0]);
        vs_[1] = static_cast<float>(rhs[1]);
        vs_[2] = static_cast<float>(rhs[2]);
        vs_[3] = 0.0;
        return *this;
    }

    matrix& operator+=(const matrix<float, 3, 1>& other) noexcept
    {
        const __m128 v1 = _mm_load_ps(this->data());
        const __m128 v2 = _mm_load_ps(other.data());
        _mm_store_ps(this->data(), _mm_add_ps(v1, v2));
        return *this;
    }
    matrix& operator-=(const matrix<float, 3, 1>& other) noexcept
    {
        const __m128 v1 = _mm_load_ps(this->data());
        const __m128 v2 = _mm_load_ps(other.data());
        _mm_store_ps(this->data(), _mm_sub_ps(v1, v2));
        return *this;
    }
    matrix& operator*=(const float other) noexcept
    {
        const __m128 v1 = _mm_load_ps(this->data());
        const __m128 v2 = _mm_set1_ps(other);
        _mm_store_ps(this->data(), _mm_mul_ps(v1, v2));
        return *this;
    }
    matrix& operator/=(const float other) noexcept
    {
        const __m128 v1 = _mm_load_ps(this->data());
        const __m128 v2 = _mm_set1_ps(_mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(other))));
        _mm_store_ps(this->data(), _mm_mul_ps(v1, v2));
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

    bool diagnosis() const noexcept {return vs_[3] == 0.0f;}

  private:
    alignas(16) storage_type vs_;
};

template<>
inline matrix<float, 3, 1> operator-(const matrix<float, 3, 1>& lhs) noexcept
{
    return _mm_sub_ps(_mm_setzero_ps(), _mm_load_ps(lhs.data()));
}
template<>
inline matrix<float, 3, 1> operator+(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    return _mm_add_ps(_mm_load_ps(lhs.data()), _mm_load_ps(rhs.data()));
}
template<>
inline matrix<float, 3, 1> operator-(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    return _mm_sub_ps(_mm_load_ps(lhs.data()), _mm_load_ps(rhs.data()));
}
template<>
inline matrix<float, 3, 1> operator*(
    const float lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    return _mm_mul_ps(_mm_set1_ps(lhs), _mm_load_ps(rhs.data()));
}
template<>
inline matrix<float, 3, 1> operator*(
    const matrix<float, 3, 1>& lhs, const float rhs) noexcept
{
    return _mm_mul_ps(_mm_load_ps(lhs.data()), _mm_set1_ps(rhs));
}
template<>
inline matrix<float, 3, 1> operator/(
    const matrix<float, 3, 1>& lhs, const float rhs) noexcept
{
    return _mm_mul_ps(_mm_load_ps(lhs.data()), _mm_set1_ps(
                _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(other)))));
}

// ---------------------------------------------------------------------------
// length
// ---------------------------------------------------------------------------

// length_sq -----------------------------------------------------------------

template<>
inline float length_sq(const matrix<float, 3, 1>& v) noexcept
{
    const __m128 arg = _mm_load_ps(v.data());
    const matrix<float, 3, 1> sq(_mm_mul_ps(arg, arg));
    return sq[0] + sq[1] + sq[2];
}

template<>
inline std::pair<float, float> length_sq(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    // gcc does not support _mm256_set_m128(arg2, arg1)
    const __m256 v12 =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);

    const __m256 mul = _mm256_mul_ps(v12, v12);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |aa|a0|xx|xx|bb|b0|xx|xx| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |as|xx|xx|xx|bs|xx|xx|xx| hadd2

    const __m256 hadd1 = _mm256_hadd_ps(mul,   mul);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    return std::make_pair(
            _mm_cvtss_f32(_mm256_extractf128_ps(hadd2, 0)),
            _mm_cvtss_f32(_mm256_extractf128_ps(hadd2, 1)));
}

template<>
inline std::tuple<float, float, float> length_sq(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m256 v12   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 v3x   = _mm256_castps128_ps256(arg3);
    const __m256 mul12 = _mm256_mul_ps(v12, v12);
    const __m256 mul3x = _mm256_mul_ps(v3x, v3x);

    // |a1|a2|a3|00|b1|b2|b3|00| |c1|c2|c3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |
    //  |  |  .--.--|--|----------+-----'
    // |aa|a0|cc|c0|bb|b0|xx|xx|
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|cs|xx|xx|bs|xx|xx|xx| result

    const __m256 hadd1 = _mm256_hadd_ps(mul12, mul3x);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    const __m128 acxx = _mm256_extractf128_ps(hadd2, 0);
    const __m128 bxxx = _mm256_extractf128_ps(hadd2, 1);

    // 01010101 == 85
    return std::make_tuple(_mm_cvtss_f32(acxx), _mm_cvtss_f32(bxxx),
                           _mm_cvtss_f32(_mm_permute_ps(acxx, 85u)));
}

template<>
inline std::tuple<float, float, float, float> length_sq(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());
    const __m128 arg4 = _mm_load_ps(v4.data());

    const __m256 v13   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v24   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg2), arg4, 1);

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul24 = _mm256_mul_ps(v24, v24);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|d1|d2|d3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'  |  |  |  |
    //  |  .--+--'  |  .--+--'    |     |     |  |  |  |
    //  |  |  .--.--|--|--.--.----+-----'-----+--'--+--'
    // |aa|a0|bb|b0|cc|c0|dd|d0| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|ds|cs|ds| hadd2

    const __m256 hadd1 = _mm256_hadd_ps(mul13, mul24);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    const __m128 abab = _mm256_extractf128_ps(hadd2, 0);
    const __m128 cdcd = _mm256_extractf128_ps(hadd2, 1);
    // 1100 == 12
    const __m128 abcd = _mm_blend_ps(abab, cdcd, 12u);

    alignas(16) float result[4];
    _mm_store_ps(result, abcd);
    return std::make_tuple(result[0], result[1], result[2], result[3]);
}

// length --------------------------------------------------------------------

template<>
inline float length(const matrix<float, 3, 1>& v) noexcept
{
    const float lsq = length_sq(v);
    return lsq * _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lsq)));
}

template<>
inline std::pair<float, float> length(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 v12 =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 mul = _mm256_mul_ps(v12, v12);

    const __m256 hadd1 = _mm256_hadd_ps(mul,   mul);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |aa|a0|aa|a0|bb|b0|bb|b0| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |as|as|as|as|bs|bs|bs|bs| hadd2

    const __m128 aaaa = _mm256_extractf128_ps(hadd2, 0);
    const __m128 bbbb = _mm256_extractf128_ps(hadd2, 1);
    // 1010 = 10
    const __m128 abab = _mm_blend_ps(aaaa, bbbb, 10u);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_mul_ps(abab, _mm_rsqrt_ps(abab)));
    return std::make_pair(result[0], result[1]);
}

template<>
inline std::tuple<float, float, float> length(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m256 v13   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v2x   = _mm256_castps128_ps256(arg2); // upper bits are undef

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul2x = _mm256_mul_ps(v2x, v2x);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |
    //  |  |  .--.--|--|----------+-----'
    // |aa|a0|bb|b0|cc|c0|xx|xx| |aa|a0|bb|b0|cc|c0|xx|xx|
    //  +--'  |  |  +--'  |  |    +--'  +--'  +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |     |     |
    //  |  |  .--.--|--|--.-------+-----'-----+-----'
    // |as|bs|as|bs|cs|xx|cs|xx| result

    const __m256 hadd1 = _mm256_hadd_ps(mul13, mul2x);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    const __m128 abab = _mm256_extractf128_ps(hadd2, 0);
    const __m128 cxcx = _mm256_extractf128_ps(hadd2, 1);
    // 0100 == 4
    const __m128 abcb = _mm_blend_ps(abab, cxcx, 4u);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_mul_ps(abcb, _mm_rsqrt_ps(abcb)));

    return std::make_tuple(result[0], result[1], result[2]);
}

template<>
inline std::tuple<float, float, float, float> length(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());
    const __m128 arg4 = _mm_load_ps(v4.data());

    const __m256 v13   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v24   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg2), arg4, 1);

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul24 = _mm256_mul_ps(v24, v24);

    const __m256 hadd1 = _mm256_hadd_ps(mul13, mul24);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|d1|d2|d3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'  |  |  |  |
    //  |  .--+--'  |  .--+--'    |     |     |  |  |  |
    //  |  |  .--.--|--|--.--.----+-----'-----+--'--+--'
    // |aa|a0|bb|b0|cc|c0|dd|d0| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|ds|cs|ds| hadd2

    const __m128 abab = _mm256_extractf128_ps(hadd2, 0);
    const __m128 cdcd = _mm256_extractf128_ps(hadd2, 1);
    // 1100 == 12
    const __m128 abcd = _mm_blend_ps(abab, cdcd, 12u);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_mul_ps(abcd, _mm_rsqrt_ps(abcd)));

    return std::make_tuple(result[0], result[1], result[2], result[3]);
}

// rlength -------------------------------------------------------------------

template<>
inline float rlength(const matrix<float, 3, 1>& v) noexcept
{
    const float lsq = length_sq(v);
    return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lsq)));
}

template<>
inline std::pair<float, float> rlength(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 v12 =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 mul = _mm256_mul_ps(v12, v12);

    const __m256 hadd1 = _mm256_hadd_ps(mul,   mul);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |aa|a0|aa|a0|bb|b0|bb|b0| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |as|as|as|as|bs|bs|bs|bs| hadd2

    const __m128 aaaa = _mm256_extractf128_ps(hadd2, 0);
    const __m128 bbbb = _mm256_extractf128_ps(hadd2, 1);
    // 1010 = 10
    const __m128 abab = _mm_blend_ps(aaaa, bbbb, 10u);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_rsqrt_ps(abab));

    return std::make_pair(result[0], result[1]);

}

template<>
inline std::tuple<float, float, float> rlength(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m256 v13   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v2x   = _mm256_castps128_ps256(arg2); // upper bits are undef

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul2x = _mm256_mul_ps(v2x, v2x);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |
    //  |  |  .--.--|--|----------+-----'
    // |aa|a0|bb|b0|cc|c0|xx|xx| |aa|a0|bb|b0|cc|c0|xx|xx|
    //  +--'  |  |  +--'  |  |    +--'  +--'  +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |     |     |
    //  |  |  .--.--|--|--.-------+-----'-----+-----'
    // |as|bs|as|bs|cs|xx|cs|xx| result

    const __m256 hadd1 = _mm256_hadd_ps(mul13, mul2x);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    const __m128 abab = _mm256_extractf128_ps(hadd2, 0);
    const __m128 cxcx = _mm256_extractf128_ps(hadd2, 1);
    // 0100 == 4
    const __m128 abcb = _mm_blend_ps(abab, cxcx, 4u);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_rsqrt_ps(abcb));

    return std::make_tuple(result[0], result[1], result[2]);
}

template<>
inline std::tuple<float, float, float, float> rlength(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());
    const __m128 arg4 = _mm_load_ps(v4.data());

    const __m256 v13   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v24   =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg2), arg4, 1);

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul24 = _mm256_mul_ps(v24, v24);

    const __m256 hadd1 = _mm256_hadd_ps(mul13, mul24);
    const __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|d1|d2|d3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'  |  |  |  |
    //  |  .--+--'  |  .--+--'    |     |     |  |  |  |
    //  |  |  .--.--|--|--.--.----+-----'-----+--'--+--'
    // |aa|a0|bb|b0|cc|c0|dd|d0| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|ds|cs|ds| hadd2

    const __m128 abab = _mm256_extractf128_ps(hadd2, 0);
    const __m128 cdcd = _mm256_extractf128_ps(hadd2, 1);
    // 1100 == 12
    const __m128 abcd = _mm_blend_ps(abab, cdcd, 12u);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_rsqrt_ps(abcd));

    return std::make_tuple(result[0], result[1], result[2], result[3]);
}

// regularize ----------------------------------------------------------------

template<>
inline std::pair<matrix<float, 3, 1>, float>
regularize(const matrix<float, 3, 1>& v) noexcept
{
    const __m128 arg  = _mm_load_ps(v.data());
    const __m128 mul  = _mm_mul_ps(arg, arg);
    const __m128 hadd = _mm_hadd_ps(mul, mul);
    const __m128 lsq  = _mm_hadd_ps(hadd, hadd);
    const __m128 rlen = _mm_rsqrt_ps(lsq);
    // |a1|a2|a3|00|
    //  +--'  |  |
    //  |  .--+--'
    //  |  |
    // |aa|a0|aa|a0| hadd
    //  +--'  |  |
    //  |  .--+--'
    //  |  |
    // |as|as|as|as|
    return std::make_pair(matrix<float, 3, 1>(_mm_mul_ps(arg, rlen)),
                          lsq * _mm_cvtss_f32(rlen));
}
template<>
inline std::pair<std::pair<matrix<float, 3, 1>, float>,
                 std::pair<matrix<float, 3, 1>, float>>
regularize(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2
           ) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 v12 =
        _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 mul = _mm256_mul_ps(v12, v12);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'   mm256_hadd_ps(mul, mul)
    //  |  |        |  |
    // |aa|a0|aa|a0|bb|b0|bb|b0| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'   mm256_hadd_ps(hadd1, hadd1)
    //  |  |        |  |
    // |as|as|as|as|bs|bs|bs|bs| len

    const __m256 hadd1 = _mm256_hadd_ps(mul, mul);
    const __m256 lensq = _mm256_hadd_ps(hadd1, hadd1);
    const __m256 rlens = _mm256_rsqrt_ps(lensq);
    const __m256 rv12  = _mm256_mul_ps(v12,   rlens);
    const __m256 lens  = _mm256_mul_ps(lensq, rlens);

    return std::make_pair(
        std::make_pair(
            matrix<float, 3, 1>(_mm256_extractf128_ps(rv12, 0)),
            _mm_cvtss_f32(_mm256_extractf128_ps(lens, 0))
            ),
        std::make_pair(
            matrix<float, 3, 1>(_mm256_extractf128_ps(rv12, 1)),
            _mm_cvtss_f32(_mm256_extractf128_ps(lens, 1))
            )
        );
}
template<>
inline std::tuple<std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>>
regularize(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
           const matrix<float, 3, 1>& v3) noexcept
{
    const auto v12 = regularize(v1, v2);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12), regularize(v3));
}
template<>
inline std::tuple<std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>>
regularize(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
           const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4
           ) noexcept
{
    const auto v12 = regularize(v1, v2);
    const auto v34 = regularize(v3, v4);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12),
                           std::get<0>(v34), std::get<1>(v34));
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<>
inline matrix<float, 3, 1> max(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    return _mm_max_ps(_mm_load_ps(v1.data()), _mm_load_ps(v2.data()));
}

template<>
inline matrix<float, 3, 1> min(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    return _mm_min_ps(_mm_load_ps(v1.data()), _mm_load_ps(v2.data()));
}

// floor ---------------------------------------------------------------------

template<>
inline matrix<float, 3, 1> floor(const matrix<float, 3, 1>& v) noexcept
{
    return _mm_floor_ps(_mm_load_ps(v.data()));
}

template<>
inline std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
floor(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m256 v12  = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 flr  = _mm256_floor_ps(v12);

    return std::make_pair(matrix<float, 3, 1>(_mm256_extractf128_ps(flr, 0)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(flr, 1)));
}
template<>
inline std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
floor(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
      const matrix<float, 3, 1>& v3) noexcept
{
    const auto v12 = floor(v1, v2);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12), floor(v3));
}
template<>
inline std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>, matrix<float, 3, 1>>
floor(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
      const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const auto v12 = floor(v1, v2);
    const auto v34 = floor(v3, v4);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12),
                           std::get<0>(v34), std::get<1>(v34));
}

// ceil ----------------------------------------------------------------------

template<>
inline matrix<float, 3, 1> ceil(const matrix<float, 3, 1>& v) noexcept
{
    return _mm_ceil_ps(_mm_load_ps(v.data()));
}

template<>
inline std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
ceil(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m256 v12  = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 cil  = _mm256_ceil_ps(v12);

    return std::make_pair(matrix<float, 3, 1>(_mm256_extractf128_ps(cil, 0)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(cil, 1)));
}
template<>
inline std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
ceil(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
      const matrix<float, 3, 1>& v3) noexcept
{
    const auto v12 = ceil(v1, v2);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12), ceil(v3));
}
template<>
inline std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>, matrix<float, 3, 1>>
ceil(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
      const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const auto v12 = ceil(v1, v2);
    const auto v34 = ceil(v3, v4);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12),
                           std::get<0>(v34), std::get<1>(v34));
}
// ---------------------------------------------------------------------------

template<>
inline float dot_product(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    const matrix<float, 3, 1> sq(
        _mm_mul_ps(_mm_load_ps(lhs.data()), _mm_load_ps(rhs.data())));
    return sq[0] + sq[1] + sq[2];
}

template<>
inline matrix<float, 3, 1> cross_product(
    const matrix<float, 3, 1>& x, const matrix<float, 3, 1>& y) noexcept
{
    const __m128 y_ = _mm_set_ps(0.0, y[0], y[2], y[1]);
    const __m128 x_ = _mm_set_ps(0.0, x[0], x[2], x[1]);

    const matrix<float, 3, 1> tmp(_mm_sub_ps(
            _mm_mul_ps(_mm_load_ps(x.data()), y_),
            _mm_mul_ps(_mm_load_ps(y.data()), x_)));

    return matrix<float, 3, 1>(tmp[1], tmp[2], tmp[0]);
}

template<>
inline float scalar_triple_product(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& mid,
    const matrix<float, 3, 1>& rhs) noexcept
{
    return (lhs[1] * mid[2] - lhs[2] * mid[1]) * rhs[0] +
           (lhs[2] * mid[0] - lhs[0] * mid[2]) * rhs[1] +
           (lhs[0] * mid[1] - lhs[1] * mid[0]) * rhs[2];
}

} // mave
#endif // MAVE_MATH_MATRIX_HPP
