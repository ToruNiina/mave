#ifndef MAVE_AVX_VECTOR3_FLOAT_HPP
#define MAVE_AVX_VECTOR3_FLOAT_HPP

#ifndef __AVX__
#error "mave/avx/vector3f.hpp requires avx support but __AVX__ is not defined."
// it also uses SSE operations
#endif

#include <immintrin.h>
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

    matrix() = default;
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

    reference       at(size_type i, size_type j)       {return vs_.at(j);}
    const_reference at(size_type i, size_type j) const {return vs_.at(j);}
    reference       operator()(size_type i, size_type j)       noexcept {return vs_[j];}
    const_reference operator()(size_type i, size_type j) const noexcept {return vs_[j];}

  private:
    alignas(16) storage_type vs_;
};

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
    return _mm_div_ps(_mm_load_ps(lhs.data()), _mm_set1_ps(rhs));
}

template<>
inline float dot_product(
    const matrix<float, 3, 1>& lhs, const matrix<float, 3, 1>& rhs) noexcept
{
    const matrix<float, 3, 1> sq(_mm_mul_ps(
        _mm_load_ps(lhs.data()), _mm_load_ps(rhs.data())));
    return sq[0] + sq[1] + sq[2];
}

template<>
inline matrix<float, 3, 1> cross_product(
    const matrix<float, 3, 1>& x, const matrix<float, 3, 1>& y) noexcept
{
    const __m128 y_ = _mm_set_ps(0.0, y[0], y[2], y[1]);
    const __m128 x_ = _mm_set_ps(0.0, x[0], x[2], x[1]);

    matrix<float, 3, 1> tmp(_mm_sub_ps(
        _mm_mul_ps(_mm_load_ps(x.data()), y_),
        _mm_mul_ps(_mm_load_ps(y.data()), x_)));

    return matrix<float, 3, 1>(tmp[1], tmp[2], tmp[0]);
}

template<>
inline float length_sq(const matrix<float, 3, 1>& v) noexcept
{
    return dot_product(v, v);
}
inline float length(const matrix<float, 3, 1>& v) noexcept
{
    return std::sqrt(length_sq(v));
}

template<>
inline float rlength(const matrix<float, 3, 1>& v) noexcept
{
    float retval;// _mm_store_ss does not require to be aligned
    _mm_store_ss(&retval, _mm_rsqrt_ss(_mm_set_ss(dot_product(v, v))));
    return retval;
}

template<>
inline matrix<float, 3, 1> regularize(const matrix<float, 3, 1>& v) noexcept
{
    return _mm_mul_pd(
        _mm_load_ps(v.data()), _mm_rsqrt_ps(_mm_set1_ps(dot_product(v, v))));
}

} // mave
#endif // MAVE_MATH_MATRIX_HPP
