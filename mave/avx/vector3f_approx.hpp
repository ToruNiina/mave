#ifndef MAVE_AVX_VECTOR3_FLOAT_APPROX_HPP
#define MAVE_AVX_VECTOR3_FLOAT_APPROX_HPP

#ifndef __AVX__
#error "mave/avx/vector3f_approx.hpp requires avx support but __AVX__ is not defined."
#endif

#ifdef MAVE_VECTOR3_FLOAT_IMPLEMENTATION
#error "specialization of vector for 3x float is already defined"
#endif
#define MAVE_VECTOR3_FLOAT_IMPLEMENTATION "avx-approx"

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

    matrix(): vs_{{0.0f, 0.0f, 0.0f, 0.0f}} {}
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
MAVE_INLINE matrix<float, 3, 1> operator-(const matrix<float, 3, 1>& v) noexcept
{
    return _mm_sub_ps(_mm_setzero_ps(), _mm_load_ps(v.data()));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> vs
          ) noexcept
{
    const __m256 v12 =_mm256_sub_ps(_mm256_setzero_ps(), _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(vs).data())),
                               _mm_load_ps(std::get<1>(vs).data()), 1));

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> vs) noexcept
{
    const auto r12 = -std::tie(std::get<0>(vs), std::get<1>(vs));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12), -std::get<2>(vs));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> vs
          ) noexcept
{
    const auto r12 = -std::tie(std::get<0>(vs), std::get<1>(vs));
    const auto r34 = -std::tie(std::get<2>(vs), std::get<3>(vs));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// addition operator+
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> operator+(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    return _mm_add_ps(_mm_load_ps(lhs.data()), _mm_load_ps(rhs.data()));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator+(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
                               _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v12 = _mm256_add_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
operator+(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v2) noexcept
{
    const auto r12 = std::tie(std::get<0>(v1), std::get<1>(v1)) +
                     std::tie(std::get<0>(v2), std::get<1>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<2>(v1) + std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>, matrix<float, 3, 1>>
operator+(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const auto r12 = std::tie(std::get<0>(v1), std::get<1>(v1)) +
                     std::tie(std::get<0>(v2), std::get<1>(v2));
    const auto r34 = std::tie(std::get<2>(v1), std::get<3>(v1)) +
                     std::tie(std::get<2>(v2), std::get<3>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// subtraction operator-
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> operator-(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    return _mm_sub_ps(_mm_load_ps(lhs.data()), _mm_load_ps(rhs.data()));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
                               _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v12 = _mm256_sub_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v2) noexcept
{
    const auto r12 = std::tie(std::get<0>(v1), std::get<1>(v1)) -
                     std::tie(std::get<0>(v2), std::get<1>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<2>(v1) - std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const auto r12 = std::tie(std::get<0>(v1), std::get<1>(v1)) -
                     std::tie(std::get<0>(v2), std::get<1>(v2));
    const auto r34 = std::tie(std::get<2>(v1), std::get<3>(v1)) -
                     std::tie(std::get<2>(v2), std::get<3>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// multiplication operator*
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> operator*(
    const float lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    return _mm_mul_ps(_mm_set1_ps(lhs), _mm_load_ps(rhs.data()));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<float, float> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v1))),
                               _mm_set1_ps(std::get<1>(v1)), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
                               _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v12 = _mm256_mul_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
operator*(std::tuple<float, float, float> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v2) noexcept
{
    const auto r12 = std::tuple<float, float>(std::get<0>(v1), std::get<1>(v1)) *
                                     std::tie(std::get<0>(v2), std::get<1>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<2>(v1) * std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<float, float, float, float> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const auto r12 = std::tuple<float, float>(std::get<0>(v1), std::get<1>(v1)) *
                                     std::tie(std::get<0>(v2), std::get<1>(v2));
    const auto r34 = std::tuple<float, float>(std::get<2>(v1), std::get<3>(v1)) *
                                     std::tie(std::get<2>(v2), std::get<3>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

template<>
MAVE_INLINE matrix<float, 3, 1> operator*(
    const matrix<float, 3, 1>& lhs, const float rhs) noexcept
{
    return _mm_mul_ps(_mm_load_ps(lhs.data()), _mm_set1_ps(rhs));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<float, float> v2) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v2))),
                               _mm_set1_ps(std::get<1>(v2)), 1);
    const __m256 v12 = _mm256_mul_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
operator*(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<float, float, float> v2) noexcept
{
    const auto r12 =    std::tie(std::get<0>(v1), std::get<1>(v1)) *
        std::tuple<float, float>(std::get<0>(v2), std::get<1>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<2>(v1) * std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<float, float, float, float> v2) noexcept
{
    const auto r12 =    std::tie(std::get<0>(v1), std::get<1>(v1)) *
        std::tuple<float, float>(std::get<0>(v2), std::get<1>(v2));
    const auto r34 =    std::tie(std::get<2>(v1), std::get<3>(v1)) *
        std::tuple<float, float>(std::get<2>(v2), std::get<3>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// division operator/
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> operator/(
    const matrix<float, 3, 1>& lhs, const float rhs) noexcept
{
    return _mm_mul_ps(_mm_load_ps(lhs.data()), _mm_set1_ps(_mm_cvtss_f32(
                    _mm_rcp_ss(_mm_set_ss(rhs)))));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator/(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<float, float> v2) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_rcp_ps(_mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v2))),
                               _mm_set1_ps(std::get<1>(v2)), 1));
    const __m256 v12 = _mm256_mul_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
operator/(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<float, float, float> v2) noexcept
{
    const auto r12 =    std::tie(std::get<0>(v1), std::get<1>(v1)) /
        std::tuple<float, float>(std::get<0>(v2), std::get<1>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<2>(v1) / std::get<2>(v2));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>, matrix<float, 3, 1>>
operator/(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<float, float, float, float> v2) noexcept
{
    const auto r12 =    std::tie(std::get<0>(v1), std::get<1>(v1)) /
        std::tuple<float, float>(std::get<0>(v2), std::get<1>(v2));
    const auto r34 =    std::tie(std::get<2>(v1), std::get<3>(v1)) /
        std::tuple<float, float>(std::get<2>(v2), std::get<3>(v2));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// length
// ---------------------------------------------------------------------------

// length_sq -----------------------------------------------------------------

template<>
MAVE_INLINE float length_sq(const matrix<float, 3, 1>& v) noexcept
{
    const __m128 arg = _mm_load_ps(v.data());
    const matrix<float, 3, 1> sq(_mm_mul_ps(arg, arg));
    return sq[0] + sq[1] + sq[2];
}

template<>
MAVE_INLINE std::pair<float, float> length_sq(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 v12 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 mul = _mm256_mul_ps(v12, v12);
    const __m256 hadd1  = _mm256_hadd_ps(mul,   mul);
    const __m256 result = _mm256_hadd_ps(hadd1, hadd1);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |aa|a0|xx|xx|bb|b0|xx|xx| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |as|xx|xx|xx|bs|xx|xx|xx| result

    return std::make_pair(_mm_cvtss_f32(_mm256_castps256_ps128(result)),
                          _mm_cvtss_f32(_mm256_extractf128_ps(result, 1)));
}

template<>
MAVE_INLINE std::tuple<float, float, float> length_sq(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m256 v13   = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul2x = _mm256_castps128_ps256(_mm_mul_ps(arg2, arg2));
    const __m256 hadd  = _mm256_hadd_ps(mul13, mul2x);
    const __m256 rslt  = _mm256_hadd_ps(hadd,  hadd);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |
    //  |  |  .--.--|--|----------+-----'
    // |aa|a0|bb|b0|cc|c0|xx|xx|
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|xx|cs|xx| result
    //   0  0  1  0  blend
    // |cs|xx|cs|xx|

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_blend_ps(_mm256_castps256_ps128(rslt),
                                      _mm256_extractf128_ps(rslt, 1), 4));

    return std::make_tuple(result[0], result[1], result[2]);
}

template<>
MAVE_INLINE std::tuple<float, float, float, float> length_sq(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());
    const __m128 arg4 = _mm_load_ps(v4.data());

    const __m256 v13 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v24 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg2), arg4, 1);

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul24 = _mm256_mul_ps(v24, v24);
    const __m256 hadd  = _mm256_hadd_ps(mul13, mul24);
    const __m256 rslt  = _mm256_hadd_ps(hadd,  hadd);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|d1|d2|d3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'  |  |  |  |
    //  |  .--+--'  |  .--+--'    |     |     |  |  |  |
    //  |  |  .--.--|--|--.--.----+-----'-----+--'--+--'
    // |aa|a0|bb|b0|cc|c0|dd|d0|
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|ds|cs|ds| result

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_blend_ps(_mm256_castps256_ps128(rslt),
                                      _mm256_extractf128_ps(rslt, 1), 12));
     return std::make_tuple(result[0], result[1], result[2], result[3]);
}

// length --------------------------------------------------------------------

template<>
MAVE_INLINE float length(const matrix<float, 3, 1>& v) noexcept
{
    const float lsq = length_sq(v);
    return lsq * _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lsq)));
}

template<>
MAVE_INLINE std::pair<float, float> length(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 v12 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 mul = _mm256_mul_ps(v12, v12);
    const __m256 hadd1 = _mm256_hadd_ps(mul,   mul);
    const __m256 lsq   = _mm256_hadd_ps(hadd1, hadd1);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |aa|a0|xx|xx|bb|b0|xx|xx| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |as|xx|xx|xx|bs|xx|xx|xx| result

    const __m128 lsq12 = _mm_set_ps(0.0f, 0.0f,
        _mm_cvtss_f32(_mm256_extractf128_ps(lsq, 1)),
        _mm_cvtss_f32(_mm256_castps256_ps128(lsq)));

    alignas(16) float len[4];
    _mm_store_ps(len, _mm_mul_ps(lsq12, _mm_rsqrt_ps(lsq12)));

    return std::make_pair(len[0], len[1]);
}

template<>
MAVE_INLINE std::tuple<float, float, float> length(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m256 v13   = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul2x = _mm256_castps128_ps256(_mm_mul_ps(arg2, arg2));
    const __m256 hadd  = _mm256_hadd_ps(mul13, mul2x);
    const __m256 rslt  = _mm256_hadd_ps(hadd,  hadd);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |
    //  |  |  .--.--|--|----------+-----'
    // |aa|a0|bb|b0|cc|c0|xx|xx|
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|xx|cs|xx| result
    //   0  0  1  0  blend
    // |cs|xx|cs|xx|

    const __m128 lsq = _mm_blend_ps(
        _mm256_castps256_ps128(rslt), _mm256_extractf128_ps(rslt, 1), 4);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_mul_ps(lsq, _mm_rsqrt_ps(lsq)));

    return std::make_tuple(result[0], result[1], result[2]);
}

template<>
MAVE_INLINE std::tuple<float, float, float, float> length(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());
    const __m128 arg4 = _mm_load_ps(v4.data());

    const __m256 v13 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v24 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg2), arg4, 1);

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul24 = _mm256_mul_ps(v24, v24);
    const __m256 hadd  = _mm256_hadd_ps(mul13, mul24);
    const __m256 rslt  = _mm256_hadd_ps(hadd,  hadd);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|d1|d2|d3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'  |  |  |  |
    //  |  .--+--'  |  .--+--'    |     |     |  |  |  |
    //  |  |  .--.--|--|--.--.----+-----'-----+--'--+--'
    // |aa|a0|bb|b0|cc|c0|dd|d0|
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|ds|cs|ds| result
    // |cs|ds|cs|ds| blend (0b1100 == 12)
    // |as|bs|cs|ds|

    const __m128 lsq = _mm_blend_ps(
        _mm256_castps256_ps128(rslt), _mm256_extractf128_ps(rslt, 1), 12);

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_mul_ps(lsq, _mm_rsqrt_ps(lsq)));
     return std::make_tuple(result[0], result[1], result[2], result[3]);
}

// rlength -------------------------------------------------------------------

template<>
MAVE_INLINE float rlength(const matrix<float, 3, 1>& v) noexcept
{
    const float lsq = length_sq(v);
    return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lsq)));
}

template<>
MAVE_INLINE std::pair<float, float> rlength(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 v12 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 mul = _mm256_mul_ps(v12, v12);
    const __m256 hadd1 = _mm256_hadd_ps(mul,   mul);
    const __m256 lsq   = _mm256_hadd_ps(hadd1, hadd1);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |aa|a0|xx|xx|bb|b0|xx|xx| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    //  |  |        |  |
    // |as|xx|xx|xx|bs|xx|xx|xx| result

    alignas(16) float len[4];
    _mm_store_ps(len, _mm_rsqrt_ps(
        _mm_set_ps(0.0f, 0.0f,
            _mm_cvtss_f32(_mm256_extractf128_ps(lsq, 1)),
            _mm_cvtss_f32(_mm256_castps256_ps128(lsq)))));

    return std::make_pair(len[0], len[1]);
}

template<>
MAVE_INLINE std::tuple<float, float, float> rlength(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m256 v13   = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul2x = _mm256_castps128_ps256(_mm_mul_ps(arg2, arg2));
    const __m256 hadd  = _mm256_hadd_ps(mul13, mul2x);
    const __m256 rslt  = _mm256_hadd_ps(hadd,  hadd);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'
    //  |  .--+--'  |  .--+--'    |     |
    //  |  |  .--.--|--|----------+-----'
    // |aa|a0|bb|b0|cc|c0|xx|xx|
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|xx|cs|xx| result
    //   0  0  1  0  blend
    // |cs|xx|cs|xx|

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_rsqrt_ps(_mm_blend_ps(
        _mm256_castps256_ps128(rslt), _mm256_extractf128_ps(rslt, 1), 4)));

    return std::make_tuple(result[0], result[1], result[2]);
}

template<>
MAVE_INLINE std::tuple<float, float, float, float> rlength(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3, const matrix<float, 3, 1>& v4) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());
    const __m128 arg4 = _mm_load_ps(v4.data());

    const __m256 v13 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg3, 1);
    const __m256 v24 = _mm256_insertf128_ps(_mm256_castps128_ps256(arg2), arg4, 1);

    const __m256 mul13 = _mm256_mul_ps(v13, v13);
    const __m256 mul24 = _mm256_mul_ps(v24, v24);
    const __m256 hadd  = _mm256_hadd_ps(mul13, mul24);
    const __m256 rslt  = _mm256_hadd_ps(hadd,  hadd);

    // |a1|a2|a3|00|c1|c2|c3|00| |b1|b2|b3|00|d1|d2|d3|00|
    //  +--'  |  |  +--'  |  |    +--'  +--'  |  |  |  |
    //  |  .--+--'  |  .--+--'    |     |     |  |  |  |
    //  |  |  .--.--|--|--.--.----+-----'-----+--'--+--'
    // |aa|a0|bb|b0|cc|c0|dd|d0|
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'
    // |as|bs|as|bs|cs|ds|cs|ds| result

    alignas(16) float result[4];
    _mm_store_ps(result, _mm_rsqrt_ps(_mm_blend_ps(
        _mm256_castps256_ps128(rslt), _mm256_extractf128_ps(rslt, 1), 12)));
     return std::make_tuple(result[0], result[1], result[2], result[3]);
}

// regularize ----------------------------------------------------------------

template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, float>
regularize(const matrix<float, 3, 1>& v) noexcept
{
    const __m128 arg = _mm_load_ps(v.data());
    const __m128 mul = _mm_mul_ps(arg, arg);
    const __m128 had = _mm_hadd_ps(mul, mul);
    const __m128 lsq = _mm_hadd_ps(had, had);
    const __m128 rln = _mm_rsqrt_ps(lsq);

    // |a1|a2|a3|00| mul
    //  +--'  |  |
    //  |  .--+--'
    //  |  |
    // |aa|a0|aa|a0|
    //  +--'  |  |
    //  |  .--+--'
    //  |  |
    // |as|as|as|as|
    return std::make_pair(matrix<float, 3, 1>(_mm_mul_ps(arg, rln)),
                          _mm_cvtss_f32(lsq) * _mm_cvtss_f32(rln));
}
template<>
MAVE_INLINE
std::pair<std::pair<matrix<float, 3, 1>, float>,
          std::pair<matrix<float, 3, 1>, float>>
regularize(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2
           ) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 v12   = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 mul   = _mm256_mul_ps(v12, v12);
    const __m256 hadd1 = _mm256_hadd_ps(mul, mul);
    const __m256 lensq = _mm256_hadd_ps(hadd1, hadd1);
    const __m256 rlen  = _mm256_rsqrt_ps(lensq);

    // |a1|a2|a3|00|b1|b2|b3|00| mul
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'   mm256_hadd_ps(mul, mul)
    //  |  |        |  |
    // |aa|a0|aa|a0|bb|b0|bb|b0| hadd1
    //  +--'  |  |  +--'  |  |
    //  |  .--+--'  |  .--+--'   mm256_hadd_ps(hadd1, hadd1)
    //  |  |        |  |
    // |as|as|as|as|bs|bs|bs|bs|
    const __m256 rv12 = _mm256_mul_ps(v12, rlen);
    const __m256 len  = _mm256_mul_ps(lensq, rlen);

    return std::make_pair(
        std::make_pair(_mm256_castps256_ps128(rv12),
                       _mm_cvtss_f32(_mm256_castps256_ps128(len))),
        std::make_pair(_mm256_extractf128_ps(rv12, 1),
                       _mm_cvtss_f32(_mm256_extractf128_ps(len, 1))));
}
template<>
MAVE_INLINE std::tuple<std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>>
regularize(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
           const matrix<float, 3, 1>& v3) noexcept
{
    const auto v12 = regularize(v1, v2);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12), regularize(v3));
}
template<>
MAVE_INLINE std::tuple<std::pair<matrix<float, 3, 1>, float>,
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
MAVE_INLINE matrix<float, 3, 1> max(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    return _mm_max_ps(_mm_load_ps(v1.data()), _mm_load_ps(v2.data()));
}

template<>
MAVE_INLINE matrix<float, 3, 1> min(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    return _mm_min_ps(_mm_load_ps(v1.data()), _mm_load_ps(v2.data()));
}

// floor ---------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> floor(const matrix<float, 3, 1>& v) noexcept
{
    return _mm_floor_ps(_mm_load_ps(v.data()));
}

template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
floor(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 rslt = _mm256_floor_ps(
                _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1));

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(rslt)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(rslt, 1)));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
floor(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
      const matrix<float, 3, 1>& v3) noexcept
{
    const auto v12 = floor(v1, v2);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12), floor(v3));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
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
MAVE_INLINE matrix<float, 3, 1> ceil(const matrix<float, 3, 1>& v) noexcept
{
    return _mm_ceil_ps(_mm_load_ps(v.data()));
}

template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
ceil(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());

    const __m256 rslt = _mm256_ceil_ps(
                _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1));

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(rslt)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(rslt, 1)));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
                  matrix<float, 3, 1>>
ceil(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
      const matrix<float, 3, 1>& v3) noexcept
{
    const auto v12 = ceil(v1, v2);
    return std::make_tuple(std::get<0>(v12), std::get<1>(v12), ceil(v3));
}
template<>
MAVE_INLINE std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
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
MAVE_INLINE float dot_product(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    const matrix<float, 3, 1> sq(
        _mm_mul_ps(_mm_load_ps(lhs.data()), _mm_load_ps(rhs.data())));
    return sq[0] + sq[1] + sq[2];
}

template<>
MAVE_INLINE matrix<float, 3, 1> cross_product(
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
MAVE_INLINE float scalar_triple_product(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& mid,
    const matrix<float, 3, 1>& rhs) noexcept
{
    return (lhs[1] * mid[2] - lhs[2] * mid[1]) * rhs[0] +
           (lhs[2] * mid[0] - lhs[0] * mid[2]) * rhs[1] +
           (lhs[0] * mid[1] - lhs[1] * mid[0]) * rhs[2];
}


} // mave
#endif // MAVE_MATH_MATRIX_HPP
