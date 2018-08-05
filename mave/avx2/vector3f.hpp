#ifndef MAVE_AVX2_VECTOR3_FLOAT_HPP
#define MAVE_AVX2_VECTOR3_FLOAT_HPP

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

    matrix(const std::array<float, 3>& arg) noexcept
        : vs_{{arg[0], arg[1], arg[2], 0.0f}}
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

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<float,3,1>&,       matrix<float,3,1>&> v1,
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

    _mm_store_ps(std::get<0>(v1).data(), _mm256_castps256_ps128(v12));
    _mm_store_ps(std::get<1>(v1).data(), _mm256_extractf128_ps(v12, 1));
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<float,3,1>&, matrix<float,3,1>&,
                     matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&> v2) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) +=
        std::tie(std::get<0>(v2), std::get<1>(v2));
    std::get<2>(v1) += std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<float,3,1>&,       matrix<float,3,1>&,
                     matrix<float,3,1>&,       matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&, const matrix<float,3,1>&> v2
    ) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) +=
        std::tie(std::get<0>(v2), std::get<1>(v2));
    std::tie(std::get<2>(v1), std::get<3>(v1)) +=
        std::tie(std::get<2>(v2), std::get<3>(v2));
    return ;
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

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<float,3,1>&,       matrix<float,3,1>&> v1,
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

    _mm_store_ps(std::get<0>(v1).data(), _mm256_castps256_ps128(v12));
    _mm_store_ps(std::get<1>(v1).data(), _mm256_extractf128_ps(v12, 1));
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<float,3,1>&, matrix<float,3,1>&,
                     matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&> v2) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) -=
        std::tie(std::get<0>(v2), std::get<1>(v2));
    std::get<2>(v1) -= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<float,3,1>&,       matrix<float,3,1>&,
                     matrix<float,3,1>&,       matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&, const matrix<float,3,1>&> v2
    ) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) -=
        std::tie(std::get<0>(v2), std::get<1>(v2));
    std::tie(std::get<2>(v1), std::get<3>(v1)) -=
        std::tie(std::get<2>(v2), std::get<3>(v2));
    return ;
}

// ---------------------------------------------------------------------------
// multiplication operator*
// ---------------------------------------------------------------------------

// scalar * matrix -----------------------------------------------------------

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

// matrix * scalar -----------------------------------------------------------

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

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<float,3,1>&, matrix<float,3,1>&> v1,
    std::tuple<float, float> v2) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v2))),
                               _mm_set1_ps(std::get<1>(v2)), 1);
    const __m256 v12 = _mm256_mul_ps(v11, v22);

    _mm_store_ps(std::get<0>(v1).data(), _mm256_castps256_ps128(v12));
    _mm_store_ps(std::get<1>(v1).data(), _mm256_extractf128_ps(v12, 1));
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<float,3,1>&, matrix<float,3,1>&, matrix<float,3,1>&> v1,
    std::tuple<float, float, float> v2) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) *=
        std::make_tuple(std::get<0>(v2), std::get<1>(v2));
    std::get<2>(v1) *= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator*=(
    std::tuple<matrix<float, 3, 1>&, matrix<float, 3, 1>&,
               matrix<float, 3, 1>&, matrix<float, 3, 1>&> v1,
    std::tuple<float, float, float, float> v2) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) *=
        std::make_tuple(std::get<0>(v2), std::get<1>(v2));
    std::tie(std::get<2>(v1), std::get<3>(v1)) *=
        std::make_tuple(std::get<2>(v2), std::get<3>(v2));
    return ;
}

// ---------------------------------------------------------------------------
// division operator/
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

// assignment ----------------------------------------------------------------

template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<float,3,1>&, matrix<float,3,1>&> v1,
    std::tuple<float, float> v2) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(v2))),
                               _mm_set1_ps(std::get<1>(v2)), 1);
    const __m256 v12 = _mm256_div_ps(v11, v22);

    _mm_store_ps(std::get<0>(v1).data(), _mm256_castps256_ps128(v12));
    _mm_store_ps(std::get<1>(v1).data(), _mm256_extractf128_ps(v12, 1));
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<float,3,1>&, matrix<float,3,1>&, matrix<float,3,1>&> v1,
    std::tuple<float, float, float> v2) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) /=
        std::make_tuple(std::get<0>(v2), std::get<1>(v2));
    std::get<2>(v1) /= std::get<2>(v2);
    return ;
}
template<>
MAVE_INLINE void operator/=(
    std::tuple<matrix<float, 3, 1>&, matrix<float, 3, 1>&,
               matrix<float, 3, 1>&, matrix<float, 3, 1>&> v1,
    std::tuple<float, float, float, float> v2) noexcept
{
    std::tie(std::get<0>(v1), std::get<1>(v1)) /=
        std::make_tuple(std::get<0>(v2), std::get<1>(v2));
    std::tie(std::get<2>(v1), std::get<3>(v1)) /=
        std::make_tuple(std::get<2>(v2), std::get<3>(v2));
    return ;
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
MAVE_INLINE std::tuple<float, float, float, float> length_sq(
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
    _mm_store_ps(result, _mm_sqrt_ps(abcb));

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
    _mm_store_ps(result, _mm_sqrt_ps(abcd));

    return std::make_tuple(result[0], result[1], result[2], result[3]);
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
    _mm_store_ps(result, _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(abcb)));

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
    _mm_store_ps(result, _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(abcd)));

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
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
max(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v2
    ) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
                               _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v12 = _mm256_max_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
max(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&> v2) noexcept
{
    const auto r12 = max(std::tie(std::get<0>(v1), std::get<1>(v1)),
                         std::tie(std::get<0>(v2), std::get<1>(v2)));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           max(std::get<2>(v1), std::get<2>(v2)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
max(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&, const matrix<float,3,1>&> v2
    ) noexcept
{
    const auto r12 = max(std::tie(std::get<0>(v1), std::get<1>(v1)),
                         std::tie(std::get<0>(v2), std::get<1>(v2)));
    const auto r34 = max(std::tie(std::get<2>(v1), std::get<3>(v1)),
                         std::tie(std::get<2>(v2), std::get<3>(v2)));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

template<>
MAVE_INLINE matrix<float, 3, 1> min(
    const matrix<float, 3, 1>& v1, const matrix<float, 3, 1>& v2) noexcept
{
    return _mm_min_ps(_mm_load_ps(v1.data()), _mm_load_ps(v2.data()));
}
template<>
MAVE_INLINE std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
min(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&> v2
    ) noexcept
{
    const __m256 v11 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v1).data())),
                               _mm_load_ps(std::get<1>(v1).data()), 1);
    const __m256 v22 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(v2).data())),
                               _mm_load_ps(std::get<1>(v2).data()), 1);
    const __m256 v12 = _mm256_min_ps(v11, v22);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(v12)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(v12, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
min(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&> v2) noexcept
{
    const auto r12 = min(std::tie(std::get<0>(v1), std::get<1>(v1)),
                         std::tie(std::get<0>(v2), std::get<1>(v2)));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           min(std::get<2>(v1), std::get<2>(v2)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>,
           matrix<float, 3, 1>, matrix<float, 3, 1>>
min(std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&, const matrix<float,3,1>&> v1,
    std::tuple<const matrix<float,3,1>&, const matrix<float,3,1>&,
               const matrix<float,3,1>&, const matrix<float,3,1>&> v2
    ) noexcept
{
    const auto r12 = min(std::tie(std::get<0>(v1), std::get<1>(v1)),
                         std::tie(std::get<0>(v2), std::get<1>(v2)));
    const auto r34 = min(std::tie(std::get<2>(v1), std::get<3>(v1)),
                         std::tie(std::get<2>(v2), std::get<3>(v2)));
    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
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
// dot_product
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
MAVE_INLINE std::pair<float, float> dot_product(
    std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> lhs,
    std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> rhs
    ) noexcept
{
    // gcc does not support _mm256_set_m128(arg2, arg1)
    const __m256 vl12 = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<0>(lhs).data())), _mm_load_ps(std::get<1>(lhs).data()), 1);
    const __m256 vr12 = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<0>(rhs).data())), _mm_load_ps(std::get<1>(rhs).data()), 1);

    const __m256 mul = _mm256_mul_ps(vl12, vr12);

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
MAVE_INLINE std::tuple<float, float, float> dot_product(
    std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
               const matrix<float, 3, 1>&> lhs,
    std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
               const matrix<float, 3, 1>&> rhs) noexcept
{
    const __m256 vl12   = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<0>(lhs).data())), _mm_load_ps(std::get<1>(lhs).data()), 1);
    const __m256 vr12   = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<0>(rhs).data())), _mm_load_ps(std::get<1>(rhs).data()), 1);

    const __m256 vl3x   = _mm256_castps128_ps256(_mm_load_ps(std::get<2>(lhs).data()));
    const __m256 vr3x   = _mm256_castps128_ps256(_mm_load_ps(std::get<2>(rhs).data()));
    const __m256 mul12 = _mm256_mul_ps(vl12, vr12);
    const __m256 mul3x = _mm256_mul_ps(vl3x, vr3x);

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
MAVE_INLINE std::tuple<float, float, float, float> dot_product(
    std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
               const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> lhs,
    std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
               const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> rhs
               ) noexcept
{
    const __m256 vl13   = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<0>(lhs).data())), _mm_load_ps(std::get<2>(lhs).data()), 1);
    const __m256 vl24   = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<1>(lhs).data())), _mm_load_ps(std::get<3>(lhs).data()), 1);

    const __m256 vr13   = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<0>(rhs).data())), _mm_load_ps(std::get<2>(rhs).data()), 1);
    const __m256 vr24   = _mm256_insertf128_ps(_mm256_castps128_ps256(
        _mm_load_ps(std::get<1>(rhs).data())), _mm_load_ps(std::get<3>(rhs).data()), 1);

    const __m256 mul13 = _mm256_mul_ps(vl13, vr13);
    const __m256 mul24 = _mm256_mul_ps(vl24, vr24);

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

// ---------------------------------------------------------------------------
// cross_product
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> cross_product(
    const matrix<float, 3, 1>& x, const matrix<float, 3, 1>& y) noexcept
{
    const __m128 y_ = _mm_set_ps(0.0, y[0], y[2], y[1]);
    const __m128 x_ = _mm_set_ps(0.0, x[0], x[2], x[1]);

    const matrix<float, 3, 1> tmp(_mm_fmsub_ps(
            _mm_load_ps(x.data()), y_,
            _mm_mul_ps(_mm_load_ps(y.data()), x_)));

    return matrix<float, 3, 1>(tmp[1], tmp[2], tmp[0]);
}

} // mave
#endif // MAVE_MATH_MATRIX_HPP
