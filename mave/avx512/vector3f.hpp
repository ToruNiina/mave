#ifndef MAVE_AVX512F_VECTOR3_FLOAT_HPP
#define MAVE_AVX512F_VECTOR3_FLOAT_HPP

#ifndef __AVX512F__
#error "mave/avx512f/vector3f.hpp requires avx512F support but __AVX512F__ is not defined."
#endif

#ifdef MAVE_VECTOR3_FLOAT_IMPLEMENTATION
#error "specialization of vector for 3x float is already defined"
#endif
#define MAVE_VECTOR3_FLOAT_IMPLEMENTATION "avx512f"

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
        const __m128 v2 = _mm_set1_ps(other);
        _mm_store_ps(this->data(), _mm_div_ps(v1, v2));
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

// ---------------------------------------------------------------------------
// negation
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> operator-(const matrix<float, 3, 1>& lhs) noexcept
{
    return _mm_sub_ps(_mm_setzero_ps(), _mm_load_ps(lhs.data()));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> vs
          ) noexcept
{
    const __m256 rslt = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(vs).data())),
        _mm_load_ps(std::get<1>(vs).data()), 1));
    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(rslt)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(rslt, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                     const matrix<float, 3, 1>&> vs) noexcept
{
    const __m512 rslt = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_insertf32x4(
        _mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(vs).data())),
            _mm_load_ps(std::get<1>(vs).data()), 1),
        _mm_load_ps(std::get<2>(vs).data()), 2));

    return std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                          matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                          matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                     const matrix<float, 3, 1>&> vs) noexcept
{
    const __m256 v12 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(vs).data())),
            _mm_load_ps(std::get<1>(vs).data()), 1);

    const __m256 v34 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(vs).data())),
            _mm_load_ps(std::get<3>(vs).data()), 1);

    const __m512 rslt = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_insertf64x4(
            _mm512_castps256_ps512(v12), v34, 1));

    return std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                          matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                          matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)),
                          matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 3)));
}

// ---------------------------------------------------------------------------
// addition
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
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
operator+(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v2) noexcept
{
    const __m512 v111 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(v1).data())),
            _mm_load_ps(std::get<1>(v1).data()), 1),
        _mm_load_ps(std::get<2>(v1).data()), 2);

    const __m512 v222 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(v2).data())),
            _mm_load_ps(std::get<1>(v2).data()), 1),
        _mm_load_ps(std::get<2>(v2).data()), 2);

    const __m512 rslt = _mm512_add_ps(v111, v222);
    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)));
}

template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
operator+(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const __m256 v11_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
            _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v11_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(v1).data())),
            _mm_load_ps(std::get<3>(v1).data()), 1);

    const __m256 v22_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
            _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v22_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(v2).data())),
            _mm_load_ps(std::get<3>(v2).data()), 1);

    const __m256 v1111 = _mm512_insertf64x4(_mm512_castps256_ps512(v11_1), v11_2, 1);
    const __m256 v2222 = _mm512_insertf64x4(_mm512_castps256_ps512(v22_1), v22_2, 1);

    const __m512 rslt = _mm512_add_ps(v1111, v2222);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 3)));
}

// --------------------------------------------------------------------------
// subtraction
// -------------------------------------------------------------------------

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
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v2) noexcept
{
    const __m512 v111 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(v1).data())),
            _mm_load_ps(std::get<1>(v1).data()), 1),
        _mm_load_ps(std::get<2>(v1).data()), 2);

    const __m512 v222 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(v2).data())),
            _mm_load_ps(std::get<1>(v2).data()), 1),
        _mm_load_ps(std::get<2>(v2).data()), 2);

    const __m512 rslt = _mm512_sub_ps(v111, v222);
    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)));
}

template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
operator-(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const __m256 v11_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
            _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v11_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(v1).data())),
            _mm_load_ps(std::get<3>(v1).data()), 1);

    const __m256 v22_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
            _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v22_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(v2).data())),
            _mm_load_ps(std::get<3>(v2).data()), 1);

    const __m256 v1111 = _mm512_insertf64x4(_mm512_castps256_ps512(v11_1), v11_2, 1);
    const __m256 v2222 = _mm512_insertf64x4(_mm512_castps256_ps512(v22_1), v22_2, 1);

    const __m512 rslt = _mm512_sub_ps(v1111, v2222);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 3)));
}

// ---------------------------------------------------------------------------
// multiplication
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
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<float, float, float> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v2) noexcept
{
    const __m512 v111 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(v1))),
                                   _mm_set1_ps(std::get<1>(v1)), 1),
                                   _mm_set1_ps(std::get<2>(v1)), 2);

    const __m512 v222 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(v2).data())),
                                   _mm_load_ps(std::get<1>(v2).data()), 1),
                                   _mm_load_ps(std::get<2>(v2).data()), 2);

    const __m512 rslt = _mm512_mul_ps(v111, v222);
    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)));
}

template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<float, float, float, float> v1,
          std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v2
          ) noexcept
{
    const __m256 v11_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v1))),
                                   _mm_set1_ps(std::get<1>(v1)), 1);
    const __m256 v11_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(std::get<2>(v1))),
                                   _mm_set1_ps(std::get<3>(v1)), 1);

    const __m256 v22_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
                                   _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v22_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(v2).data())),
                                   _mm_load_ps(std::get<3>(v2).data()), 1);

    const __m256 v1111 = _mm512_insertf64x4(_mm512_castps256_ps512(v11_1), v11_2, 1);
    const __m256 v2222 = _mm512_insertf64x4(_mm512_castps256_ps512(v22_1), v22_2, 1);

    const __m512 rslt = _mm512_mul_ps(v1111, v2222);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 3)));
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
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<float, float, float> v2) noexcept
{
    const __m512 v111 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(v1).data())),
                                   _mm_load_ps(std::get<1>(v1).data()), 1),
                                   _mm_load_ps(std::get<2>(v1).data()), 2);

    const __m512 v222 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(v2))),
                                   _mm_set1_ps(std::get<1>(v2)), 1),
                                   _mm_set1_ps(std::get<2>(v2)), 2);

    const __m512 rslt = _mm512_mul_ps(v111, v222);
    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)));
}

template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
operator*(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<float, float, float, float> v2) noexcept
{
    const __m256 v11_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                                   _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v11_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(v1).data())),
                                   _mm_load_ps(std::get<3>(v1).data()), 1);

    const __m256 v22_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v2))),
                                   _mm_set1_ps(std::get<1>(v2)), 1);
    const __m256 v22_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(std::get<2>(v2))),
                                   _mm_set1_ps(std::get<3>(v2)), 1);

    const __m256 v1111 = _mm512_insertf64x4(_mm512_castps256_ps512(v11_1), v11_2, 1);
    const __m256 v2222 = _mm512_insertf64x4(_mm512_castps256_ps512(v22_1), v22_2, 1);

    const __m512 rslt = _mm512_mul_ps(v1111, v2222);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 3)));
}

// ---------------------------------------------------------------------------
// division
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> operator/(
    const matrix<float, 3, 1>& lhs, const float rhs) noexcept
{
    return _mm_div_ps(_mm_load_ps(lhs.data()), _mm_set1_ps(rhs));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
operator/(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<float, float> v2) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v2))),
                               _mm_set1_ps(std::get<1>(v2)), 1);
    const __m256 v12 = _mm256_div_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
operator/(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&> v1,
          std::tuple<float, float, float> v2) noexcept
{
    const __m512 v111 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_load_ps(std::get<0>(v1).data())),
                                   _mm_load_ps(std::get<1>(v1).data()), 1),
                                   _mm_load_ps(std::get<2>(v1).data()), 2);

    const __m512 v222 = _mm512_insertf32x4(_mm512_insertf32x4(
            _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(v2))),
                                   _mm_set1_ps(std::get<1>(v2)), 1),
                                   _mm_set1_ps(std::get<2>(v2)), 2);

    const __m512 rslt = _mm512_div_ps(v111, v222);
    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)));
}

template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
operator/(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
                     const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
          std::tuple<float, float, float, float> v2) noexcept
{
    const __m256 v11_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                                   _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v11_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_load_ps(std::get<2>(v1).data())),
                                   _mm_load_ps(std::get<3>(v1).data()), 1);

    const __m256 v22_1 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v2))),
                                   _mm_set1_ps(std::get<1>(v2)), 1);
    const __m256 v22_2 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(std::get<2>(v2))),
                                   _mm_set1_ps(std::get<3>(v2)), 1);

    const __m256 v1111 = _mm512_insertf64x4(_mm512_castps256_ps512(v11_1), v11_2, 1);
    const __m256 v2222 = _mm512_insertf64x4(_mm512_castps256_ps512(v22_1), v22_2, 1);

    const __m512 rslt = _mm512_div_ps(v1111, v2222);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4(rslt, 3)));
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
MAVE_INLINE std::tuple<float, float, float> length_sq(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m512 v123x = _mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2);
    const __m512 m123x = _mm512_mul_ps(v123x, v123x);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|00|00|00|00| m123x
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  3  1  5  9  7  2  6  A  B  C  D  E  F  v
    // |a1|b1|c1|00|a2|b2|c2|00|a3|b3|c3|00|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x06, 0x02,
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00
        ), m123x);

    alignas(16) float abcsq[4];
    _mm_store_ps(abcsq, _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(abc0, 0),
                                              _mm512_extractf32x4_ps(abc0, 1)),
                                   _mm512_extractf32x4_ps(abc0, 2)));

    return std::make_tuple(abcsq[0], abcsq[1], abcsq[2]);
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

    const __m512 v1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2), arg4, 3);
    const __m512 m1234 = _mm512_mul_ps(v1234, v1234);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|d1|d2|d3|00| m1234
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  C  1  5  9  D  2  6  A  E  3  7  B  F  v
    // |a1|b1|c1|d1|a2|b2|c2|d2|a3|b3|c3|d3|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0B, 0x07, 0x03, 0x0E, 0x0A, 0x06, 0x02,
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00
        ), m1234);

    alignas(16) float abcsq[4];
    _mm_store_ps(abcsq, _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(abc0, 0),
                                              _mm512_extractf32x4_ps(abc0, 1)),
                                   _mm512_extractf32x4_ps(abc0, 2)));

    return std::make_tuple(abcsq[0], abcsq[1], abcsq[2], abcsq[3]);
}

// length --------------------------------------------------------------------

template<>
MAVE_INLINE float length(const matrix<float, 3, 1>& v) noexcept
{
    return std::sqrt(length_sq(v));
}

template<>
MAVE_INLINE std::pair<float, float> length(
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
    _mm_store_ps(result, _mm_sqrt_ps(abab));
    return std::make_pair(result[0], result[1]);
}

template<>
MAVE_INLINE std::tuple<float, float, float> length(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m512 v123x = _mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2);
    const __m512 m123x = _mm512_mul_ps(v123x, v123x);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|00|00|00|00| m123x
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  3  1  5  9  7  2  6  A  B  C  D  E  F  v
    // |a1|b1|c1|00|a2|b2|c2|00|a3|b3|c3|00|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x06, 0x02,
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00
        ), m123x);

    alignas(16) float abcsq[4];
    _mm_store_ps(abcsq, _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(
        _mm512_extractf32x4_ps(abc0, 0), _mm512_extractf32x4_ps(abc0, 1)),
        _mm512_extractf32x4_ps(abc0, 2))));

    return std::make_tuple(abcsq[0], abcsq[1], abcsq[2]);
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

    const __m512 v1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2), arg4, 3);
    const __m512 m1234 = _mm512_mul_ps(v1234, v1234);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|d1|d2|d3|00| m1234
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  C  1  5  9  D  2  6  A  E  3  7  B  F  v
    // |a1|b1|c1|d1|a2|b2|c2|d2|a3|b3|c3|d3|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0B, 0x07, 0x03, 0x0E, 0x0A, 0x06, 0x02,
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00
        ), m1234);

    alignas(16) float abcsq[4];
    _mm_store_ps(abcsq, _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(
        _mm512_extractf32x4_ps(abc0, 0), _mm512_extractf32x4_ps(abc0, 1)),
        _mm512_extractf32x4_ps(abc0, 2))));

    return std::make_tuple(abcsq[0], abcsq[1], abcsq[2], abcsq[3]);
}

// rlength -------------------------------------------------------------------

template<>
MAVE_INLINE float rlength(const matrix<float, 3, 1>& v) noexcept
{
    return 1.0f / std::sqrt(length_sq(v));
}

template<>
MAVE_INLINE std::pair<float, float> rlength(
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
    _mm_store_ps(result, _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(abab)));

    return std::make_pair(result[0], result[1]);
}

template<>
MAVE_INLINE std::tuple<float, float, float> rlength(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
    const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m512 v123x = _mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2);
    const __m512 m123x = _mm512_mul_ps(v123x, v123x);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|00|00|00|00| m123x
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  3  1  5  9  7  2  6  A  B  C  D  E  F  v
    // |a1|b1|c1|00|a2|b2|c2|00|a3|b3|c3|00|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x06, 0x02,
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00
        ), m123x);

    alignas(16) float abcsq[4];
    _mm_store_ps(abcsq, _mm_div_ps(_mm_set1_ps(1.0f),
        _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(
            _mm512_extractf32x4_ps(abc0, 0), _mm512_extractf32x4_ps(abc0, 1)),
            _mm512_extractf32x4_ps(abc0, 2)))));

    return std::make_tuple(abcsq[0], abcsq[1], abcsq[2]);

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

    const __m512 v1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2), arg4, 3);
    const __m512 m1234 = _mm512_mul_ps(v1234, v1234);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|d1|d2|d3|00| m1234
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  C  1  5  9  D  2  6  A  E  3  7  B  F  v
    // |a1|b1|c1|d1|a2|b2|c2|d2|a3|b3|c3|d3|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0B, 0x07, 0x03, 0x0E, 0x0A, 0x06, 0x02,
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00
        ), m1234);

    alignas(16) float abcsq[4];
    _mm_store_ps(abcsq, _mm_div_ps(_mm_set1_ps(1.0f),
        _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(
            _mm512_extractf32x4_ps(abc0, 0), _mm512_extractf32x4_ps(abc0, 1)),
            _mm512_extractf32x4_ps(abc0, 2)))));

    return std::make_tuple(abcsq[0], abcsq[1], abcsq[2], abcsq[3]);
}

// regularize ----------------------------------------------------------------

template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, float>
regularize(const matrix<float, 3, 1>& v) noexcept
{
    const __m128 arg = _mm_load_ps(v.data());
    const __m128 mul = _mm_mul_ps(arg, arg);
    const __m128 had = _mm_hadd_ps(mul, mul);
    const __m128 len = _mm_sqrt_ps(_mm_hadd_ps(had, had));
    // |a1|a2|a3|00| mul
    //  +--'  |  |
    //  |  .--+--'
    //  |  |
    // |aa|a0|aa|a0|
    //  +--'  |  |
    //  |  .--+--'
    //  |  |
    // |as|as|as|as|
    return std::make_pair(matrix<float, 3, 1>(_mm_div_ps(arg, len)),
                          _mm_cvtss_f32(len));
}
template<>
MAVE_INLINE std::pair<std::pair<matrix<float, 3, 1>, float>,
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
    const __m256 len   = _mm256_sqrt_ps(_mm256_hadd_ps(hadd1, hadd1));
    const __m256 rv12  = _mm256_div_ps(v12, len);

    return std::make_pair(
        std::make_pair(
            matrix<float, 3, 1>(_mm256_extractf128_ps(rv12, 0)),
            _mm_cvtss_f32(_mm256_extractf128_ps(len, 0))
            ),
        std::make_pair(
            matrix<float, 3, 1>(_mm256_extractf128_ps(rv12, 1)),
            _mm_cvtss_f32(_mm256_extractf128_ps(len, 1))
            )
        );
}
template<>
MAVE_INLINE std::tuple<std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>,
                  std::pair<matrix<float, 3, 1>, float>>
regularize(const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2,
           const matrix<float, 3, 1>& v3) noexcept
{
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());

    const __m512 v123x = _mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2);
    const __m512 m123x = _mm512_mul_ps(v123x, v123x);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|00|00|00|00| m123x
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  3  1  5  9  7  2  6  A  B  C  D  E  F  v
    // |a1|b1|c1|00|a2|b2|c2|00|a3|b3|c3|00|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x06, 0x02,
            0x07, 0x09, 0x05, 0x01, 0x03, 0x08, 0x04, 0x00
        ), m123x);

    const __m128 len = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(
        _mm512_extractf32x4_ps(abc0, 0), _mm512_extractf32x4_ps(abc0, 1)),
        _mm512_extractf32x4_ps(abc0, 2)));

    const __m128 rl  = _mm_div_ps(_mm_set1_ps(1.0f), len);
    alignas(16) float pack[4];
    _mm_store_ps(pack, rl);

    // -------------------------------------------------------------------
    const __m512 r1xxx = _mm512_castps128_ps512(_mm_set1_ps(pack[0]));
    const __m512 r12xx = _mm512_insertf32x4(r1xxx, _mm_set1_ps(pack[1]), 1);
    const __m512 r123x = _mm512_insertf32x4(r12xx, _mm_set1_ps(pack[2]), 2);
    const __m512 rvs   = _mm512_mul_ps(r123x, v123x);

    _mm_store_ps(pack, len);
    return std::make_tuple(
            std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rvs, 0)), pack[0]),
            std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rvs, 1)), pack[1]),
            std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rvs, 2)), pack[2])
            );
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
    const __m128 arg1 = _mm_load_ps(v1.data());
    const __m128 arg2 = _mm_load_ps(v2.data());
    const __m128 arg3 = _mm_load_ps(v3.data());
    const __m128 arg4 = _mm_load_ps(v4.data());

    const __m512 v1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
                _mm512_castps128_ps512(arg1), arg2, 1), arg3, 2), arg4, 3);
    const __m512 m1234 = _mm512_mul_ps(v1234, v1234);

    // |a1|a2|a3|00|b1|b2|b3|00|c1|c2|c3|00|d1|d2|d3|00| m1234
    //   0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F  |
    //                                                   | mm512_permute_ps
    //   0  4  8  C  1  5  9  D  2  6  A  E  3  7  B  F  v
    // |a1|b1|c1|d1|a2|b2|c2|d2|a3|b3|c3|d3|00|00|00|00| abc0

    const __m512 abc0  = _mm512_permutexvar_ps(_mm512_set_epi32(
            0x0F, 0x0B, 0x07, 0x03, 0x0E, 0x0A, 0x06, 0x02,
            0x0D, 0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00
        ), m1234);

    const __m128 len = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(
        _mm512_extractf32x4_ps(abc0, 0), _mm512_extractf32x4_ps(abc0, 1)),
        _mm512_extractf32x4_ps(abc0, 2)));

    const __m128 rl  = _mm_div_ps(_mm_set1_ps(1.0f), len);
    alignas(16) float pack[4];
    _mm_store_ps(pack, rl);

    const __m512 r1xxx = _mm512_castps128_ps512(_mm_set1_ps(pack[0]));
    const __m512 r12xx = _mm512_insertf32x4(r1xxx, _mm_set1_ps(pack[1]), 1);
    const __m512 r123x = _mm512_insertf32x4(r12xx, _mm_set1_ps(pack[2]), 2);
    const __m512 r1234 = _mm512_insertf32x4(r123x, _mm_set1_ps(pack[3]), 3);

    const __m512 rvs = _mm512_mul_ps(r1234, v1234);

    _mm_store_ps(pack, len);
    return std::make_tuple(
            std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rvs, 0)), pack[0]),
            std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rvs, 1)), pack[1]),
            std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rvs, 2)), pack[2]),
            std::make_pair(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rvs, 3)), pack[3])
            );
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
    const __m256 v12  = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 flr  = _mm256_floor_ps(v12);

    return std::make_pair(matrix<float, 3, 1>(_mm256_extractf128_ps(flr, 0)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(flr, 1)));
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
    const __m256 v12  = _mm256_insertf128_ps(_mm256_castps128_ps256(arg1), arg2, 1);
    const __m256 cil  = _mm256_ceil_ps(v12);

    return std::make_pair(matrix<float, 3, 1>(_mm256_extractf128_ps(cil, 0)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(cil, 1)));
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
