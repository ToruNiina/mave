#ifndef MAVE_AVX_VECTOR3_DOUBLE_HPP
#define MAVE_AVX_VECTOR3_DOUBLE_HPP

#ifndef __AVX__
#error "mave/avx/vector3d.hpp requires avx support but __AVX__ is not defined."
#endif

#ifndef __SSE2__
#error "mave/avx/vector3d.hpp requires sse2 support but __SSE2__ is not defined."
#endif

#include <immintrin.h>
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

    matrix() = default;
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
        const __m256d v3 = _mm256_add_pd(v1, v2);
        _mm256_store_pd(this->data(), v3);
        return *this;
    }
    matrix& operator-=(const matrix<double, 3, 1>& other) noexcept
    {
        const __m256d v1 = _mm256_load_pd(this->data());
        const __m256d v2 = _mm256_load_pd(other.data());
        const __m256d v3 = _mm256_sub_pd(v1, v2);
        _mm256_store_pd(this->data(), v3);
        return *this;
    }
    matrix& operator*=(const double other) noexcept
    {
        const __m256d v1 = _mm256_load_pd(this->data());
        const __m256d v2 = _mm256_set1_pd(other);
        const __m256d v3 = _mm256_mul_pd(v1, v2);
        _mm256_store_pd(this->data(), v3);
        return *this;
    }
    matrix& operator/=(const double other) noexcept
    {
        const __m256d v1 = _mm256_load_pd(this->data());
        const __m256d v2 = _mm256_set1_pd(other);
        const __m256d v3 = _mm256_div_pd(v1, v2);
        _mm256_store_pd(this->data(), v3);
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
    const matrix<double, 3, 1>& lhs, const matrix<double, 3, 1>& rhs) noexcept
{
    return _mm256_add_pd(_mm256_load_pd(lhs.data()), _mm256_load_pd(rhs.data()));
}
template<>
inline matrix<double, 3, 1> operator-(
    const matrix<double, 3, 1>& lhs, const matrix<double, 3, 1>& rhs) noexcept
{
    return _mm256_sub_pd(_mm256_load_pd(lhs.data()), _mm256_load_pd(rhs.data()));
}
template<>
inline matrix<double, 3, 1> operator*(
    const double lhs, const matrix<double, 3, 1>& rhs) noexcept
{
    return _mm256_mul_pd(_mm256_set1_pd(lhs), _mm256_load_pd(rhs.data()));
}
template<>
inline matrix<double, 3, 1> operator*(
    const matrix<double, 3, 1>& lhs, const double rhs) noexcept
{
    return _mm256_mul_pd(_mm256_load_pd(lhs.data()), _mm256_set1_pd(rhs));
}
template<>
inline matrix<double, 3, 1> operator/(
    const matrix<double, 3, 1>& lhs, const double rhs) noexcept
{
    return _mm256_div_pd(_mm256_load_pd(lhs.data()), _mm256_set1_pd(rhs));
}

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
inline double length_sq(const matrix<double, 3, 1>& v) noexcept
{
    return dot_product(v, v);
}
inline double length(const matrix<double, 3, 1>& v) noexcept
{
    return std::sqrt(length_sq(v));
}

template<>
inline double rlength(const matrix<double, 3, 1>& v) noexcept
{
    // Q. WTF
    // A. see https://en.wikipedia.org/wiki/Fast_inverse_square_root.
    //    or do some google with "fast inverse square root".

    double lsq = dot_product(v, v);
    const double lsq_half = lsq * 0.5;
    std::int64_t i = *reinterpret_cast<const std::int64_t*>(&lsq);
    i = 0x5FE6EB50C7B537A9 - (i >> 1);
    lsq = *reinterpret_cast<const double*>(&i);
    return lsq * (1.5 - lsq_half * lsq * lsq);
}

template<>
inline matrix<double, 3, 1> regularize(const matrix<double, 3, 1>& v) noexcept
{
    return matrix<double, 3, 1>(_mm256_mul_pd(
            _mm256_load_pd(v.data()), _mm256_set1_pd(rlength(v))));
}

template<>
inline std::pair<double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    const matrix<double, 3, 1> hadd(_mm256_hadd_pd(mul1, mul2));
    return std::make_pair(hadd[0] + hadd[2], hadd[1] + hadd[3]);
}

template<>
inline std::pair<double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    const std::pair<double, double> lensq(length_sq(v1, v2));
    alignas(16) double len[2];
    _mm_store_pd(len, _mm_sqrt_pd(_mm_set_pd(lensq.second, lensq.first)));
    return std::make_pair(len[0], len[1]);
}

// inline matrix<int, 3, 1> ceil(const matrix<double, 3, 1>& v) noexcept
// {
//     return matrix<int, 3, 1>(_mm256_cvtpd_epi32(
//                 _mm256_ceil_pd(_mm256_load_pd(v.data()))));
// }
// inline matrix<int, 3, 1> floor(const matrix<double, 3, 1>& v) noexcept
// {
//     return matrix<int, 3, 1>(_mm256_cvtpd_epi32(
//                 _mm256_floor_pd(_mm256_load_pd(v.data()))));
// }

} // mave
#endif // MAVE_MATH_MATRIX_HPP
