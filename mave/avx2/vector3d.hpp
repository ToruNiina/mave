#ifndef MAVE_AVX2_VECTOR3_DOUBLE_HPP
#define MAVE_AVX2_VECTOR3_DOUBLE_HPP

#ifndef __AVX2__
#error "mave/avx2/vector3d.hpp requires avx support but __AVX2__ is not defined."
#endif

#ifndef MAVE_VECTOR_HPP
#error "do not use mave/avx/vector3d.hpp alone. please include mave/vector.hpp."
#endif

#ifdef MAVE_VECTOR3_DOUBLE_IMPLEMENTATION
#error "specialization of vector for 3x double is already defined"
#endif

#define MAVE_VECTOR3_DOUBLE_IMPLEMENTATION "avx2"

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
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack, _mm256_hadd_pd(
        _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2)));

    return std::make_pair(pack[0] + pack[2], pack[1] + pack[3]);
}

template<>
inline std::tuple<double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack[4];
    std::tuple<double, double, double> retval;

    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack, _mm256_hadd_pd(mul1, mul2));
    std::get<0>(retval) = pack[0] + pack[2];
    std::get<1>(retval) = pack[1] + pack[3];

    _mm256_store_pd(pack, _mm256_mul_pd(arg3, arg3));
    std::get<2>(retval) = pack[0] + pack[1] + pack[2];

    return retval;
}

template<>
inline std::tuple<double, double, double, double> length_sq(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack[4];
    std::tuple<double, double, double, double> retval;
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack, _mm256_hadd_pd(mul1, mul2));
    std::get<0>(retval) = pack[0] + pack[2];
    std::get<1>(retval) = pack[1] + pack[3];

    _mm256_store_pd(pack, _mm256_hadd_pd(mul3, mul4));
    std::get<2>(retval) = pack[0] + pack[2];
    std::get<3>(retval) = pack[1] + pack[3];

    return retval;
}

// length --------------------------------------------------------------------

template<>
inline double length(const matrix<double, 3, 1>& v) noexcept
{
    return std::sqrt(length_sq(v));
}

template<>
inline std::pair<double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2) noexcept
{
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack, _mm256_hadd_pd(
        _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2)));
    _mm_store_pd(pack, _mm_sqrt_pd(
                _mm_set_pd(pack[1] + pack[3], pack[0] + pack[2])));
    return std::make_pair(pack[0], pack[1]);
}

template<>
inline std::tuple<double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack1[4];
    alignas(32) double pack2[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul1, mul2));
    pack2[0] = pack1[0] + pack1[2];
    pack2[1] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_mul_pd(arg3, arg3));
    pack2[2] = pack1[0] + pack1[1] + pack1[2];

    _mm256_store_pd(pack1, _mm256_sqrt_pd(_mm256_load_pd(pack2)));

    return std::make_tuple(pack1[0], pack1[1], pack1[2]);
}

template<>
inline std::tuple<double, double, double, double> length(
    const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
    const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack1[4];
    alignas(32) double pack2[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul1, mul2));
    pack2[0] = pack1[0] + pack1[2];
    pack2[1] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul3, mul4));
    pack2[2] = pack1[0] + pack1[2];
    pack2[3] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_sqrt_pd(_mm256_load_pd(pack2)));

    return std::make_tuple(pack1[0], pack1[1], pack1[2], pack1[3]);
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
    alignas(32) double pack[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack, _mm256_hadd_pd(
        _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2)));
    _mm_store_pd(pack, _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(
                _mm_set_pd(pack[1] + pack[3], pack[0] + pack[2]))));

    return std::make_pair(pack[0], pack[1]);
}
template<>
inline std::tuple<double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack1[4];
    alignas(32) double pack2[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul1, mul2));
    pack2[0] = pack1[0] + pack1[2];
    pack2[1] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_mul_pd(arg3, arg3));
    pack2[2] = pack1[0] + pack1[1] + pack1[2];

    _mm256_store_pd(pack1, _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(
                    _mm256_load_pd(pack2))));

    return std::make_tuple(pack1[0], pack1[1], pack1[2]);
}
template<>
inline std::tuple<double, double, double, double>
rlength(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
        const matrix<double, 3, 1>& v3, const matrix<double, 3, 1>& v4) noexcept
{
    alignas(32) double pack1[4];
    alignas(32) double pack2[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul1, mul2));
    pack2[0] = pack1[0] + pack1[2];
    pack2[1] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul3, mul4));
    pack2[2] = pack1[0] + pack1[2];
    pack2[3] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(
                    _mm256_load_pd(pack2))));

    return std::make_tuple(pack1[0], pack1[1], pack1[2], pack1[3]);
}

// regularize ----------------------------------------------------------------

template<>
inline std::pair<matrix<double, 3, 1>, double>
regularize(const matrix<double, 3, 1>& v) noexcept
{
    const auto l = length(v);
    return std::make_pair(v * (1.0 / l), l);
}
template<>
inline std::pair<std::pair<matrix<double, 3, 1>, double>,
                 std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2
           ) noexcept
{
    alignas(32) double pack[4];
    double l1, l2;

    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack, _mm256_hadd_pd(
        _mm256_mul_pd(arg1, arg1), _mm256_mul_pd(arg2, arg2)));
    _mm_store_pd(pack, _mm_sqrt_pd(
                 _mm_set_pd(pack[1] + pack[3], pack[0] + pack[2])));

    l1 = pack[0];
    l2 = pack[1];
    _mm_store_pd(pack, _mm_div_pd(_mm_set1_pd(1.0), _mm_load_pd(pack)));

    return std::make_pair(std::make_pair(v1 * pack[0], l1),
                          std::make_pair(v2 * pack[1], l2));
}
template<>
inline std::tuple<std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>,
                  std::pair<matrix<double, 3, 1>, double>>
regularize(const matrix<double, 3, 1>& v1, const matrix<double, 3, 1>& v2,
           const matrix<double, 3, 1>& v3) noexcept
{
    alignas(32) double pack1[4];
    alignas(32) double pack2[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul1, mul2));
    pack2[0] = pack1[0] + pack1[2];
    pack2[1] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_mul_pd(arg3, arg3));
    pack2[2] = pack1[0] + pack1[1] + pack1[2];

    _mm256_store_pd(pack1, _mm256_sqrt_pd(_mm256_load_pd(pack2)));
    _mm256_store_pd(pack2, _mm256_div_pd(
                    _mm256_set1_pd(1.0), _mm256_load_pd(pack1)));

    return std::make_tuple(std::make_pair(v1 * pack2[0], pack1[0]),
                           std::make_pair(v2 * pack2[1], pack1[1]),
                           std::make_pair(v3 * pack2[2], pack1[2]));
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
    alignas(32) double pack1[4];
    alignas(32) double pack2[4];
    const __m256d arg1 = _mm256_load_pd(v1.data());
    const __m256d arg2 = _mm256_load_pd(v2.data());
    const __m256d arg3 = _mm256_load_pd(v3.data());
    const __m256d arg4 = _mm256_load_pd(v4.data());

    const __m256d mul1 = _mm256_mul_pd(arg1, arg1);
    const __m256d mul2 = _mm256_mul_pd(arg2, arg2);
    const __m256d mul3 = _mm256_mul_pd(arg3, arg3);
    const __m256d mul4 = _mm256_mul_pd(arg4, arg4);

    // |a1|a2|a3|00| |b1|b2|b3|00|
    //  +--'  |  |    +--'  |  |
    //  |     +--'    |     |  | hadd
    //  |  .--|--.----+-----+--'
    // |aa|bb|a3|b3| pack

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul1, mul2));
    pack2[0] = pack1[0] + pack1[2];
    pack2[1] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_hadd_pd(mul3, mul4));
    pack2[2] = pack1[0] + pack1[2];
    pack2[3] = pack1[1] + pack1[3];

    _mm256_store_pd(pack1, _mm256_sqrt_pd(_mm256_load_pd(pack2)));
    _mm256_store_pd(pack2, _mm256_div_pd(
                    _mm256_set1_pd(1.0), _mm256_load_pd(pack1)));

    return std::make_tuple(std::make_pair(v1 * pack2[0], pack1[0]),
                           std::make_pair(v2 * pack2[1], pack1[1]),
                           std::make_pair(v3 * pack2[2], pack1[2]),
                           std::make_pair(v4 * pack2[3], pack1[3]));
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
