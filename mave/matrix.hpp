#ifndef MAVE_MATRIX_HPP
#define MAVE_MATRIX_HPP
#include "type_traits.hpp"
#include <utility>
#include <array>
#include <algorithm>
#include <tuple>
#include <cmath>

namespace mave
{

template<typename T, std::size_t R, std::size_t C>
struct matrix
{
    //    1   ...   C
    //  1 x00 ... x0M    N = R-1
    //  . x10 ... x1M    M = C-1
    //  .   . ...   .
    //  R xN0 ... xNM    R * C matrix
    static constexpr std::size_t row_size    = R;
    static constexpr std::size_t column_size = C;
    static constexpr std::size_t total_size  = R * C;
    using value_type      = T;
    using storage_type    = std::array<T, total_size>;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;

    template<typename ... Ts, typename std::enable_if<
        sizeof...(Ts) == total_size &&
        conjunction<std::is_convertible<Ts, T> ...>::value,
        std::nullptr_t>::type = nullptr>
    matrix(Ts&& ... args) noexcept : vs_{{static_cast<T>(args)...}}{}

    matrix(const std::array<T, total_size>& arg) noexcept
    {
        std::copy(arg.begin(), arg.end(), this->vs_.begin());
    }

    matrix(){vs_.fill(value_type(0.0));}
    ~matrix() = default;
    matrix(const matrix&) = default;
    matrix(matrix&&)      = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&)      = default;

    template<typename U>
    matrix& operator=(const matrix<U, R, C>& rhs) noexcept
    {
        for(std::size_t i=0; i<total_size; ++i) {this->vs_[i] = rhs[i];}
        return *this;
    }

    template<typename U>
    matrix& operator+=(const matrix<U, R, C>& rhs) noexcept
    {
        for(std::size_t i=0; i<total_size; ++i) {this->vs_[i] += rhs[i];}
        return *this;
    }
    template<typename U>
    matrix& operator-=(const matrix<U, R, C>& rhs) noexcept
    {
        for(std::size_t i=0; i<total_size; ++i) {this->vs_[i] -= rhs[i];}
        return *this;
    }
    template<typename U>
    typename std::enable_if<std::is_convertible<U, T>::value, matrix<T, R, C>>::type&
    operator*=(const U& rhs) noexcept
    {
        for(std::size_t i=0; i<total_size; ++i) {this->vs_[i] *= rhs;}
        return *this;
    }
    template<typename U>
    typename std::enable_if<std::is_convertible<U, T>::value, matrix<T, R, C>>::type&
    operator/=(const U& rhs) noexcept
    {
        for(std::size_t i=0; i<total_size; ++i) {this->vs_[i] /= rhs;}
        return *this;
    }

    size_type size() const noexcept {return total_size;}

    pointer       data()       noexcept {return vs_.data();}
    const_pointer data() const noexcept {return vs_.data();}

    reference               at(size_type i)       {return vs_.at(i);}
    const_reference         at(size_type i) const {return vs_.at(i);}
    reference       operator[](size_type i)       noexcept {return vs_[i];}
    const_reference operator[](size_type i) const noexcept {return vs_[i];}

    reference       at(size_type i, size_type j)       {return vs_.at(i*C+j);}
    const_reference at(size_type i, size_type j) const {return vs_.at(i*C+j);}
    reference       operator()(size_type i, size_type j)       noexcept {return vs_[i*C+j];}
    const_reference operator()(size_type i, size_type j) const noexcept {return vs_[i*C+j];}

    bool diagnosis() const noexcept {return true;}

  private:
    storage_type vs_;
};

template<typename T, std::size_t R, std::size_t C>
constexpr std::size_t matrix<T, R, C>::row_size;
template<typename T, std::size_t R, std::size_t C>
constexpr std::size_t matrix<T, R, C>::column_size;
template<typename T, std::size_t R, std::size_t C>
constexpr std::size_t matrix<T, R, C>::total_size;

// ---------------------------------------------------------------------------
// negation operator
// ---------------------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
operator-(const matrix<T, R, C>& lhs) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = -lhs[i];}
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> ms) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = -std::get<0>(ms)[i];
        r2[i] = -std::get<1>(ms)[i];
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> ms) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = -std::get<0>(ms)[i];
        r2[i] = -std::get<1>(ms)[i];
        r3[i] = -std::get<2>(ms)[i];
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> ms) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = -std::get<0>(ms)[i];
        r2[i] = -std::get<1>(ms)[i];
        r3[i] = -std::get<2>(ms)[i];
        r4[i] = -std::get<3>(ms)[i];
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// ---------------------------------------------------------------------------
// addition operator+
// ---------------------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
operator+(const matrix<T, R, C>& lhs, const matrix<T, R, C>& rhs) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] + rhs[i];}
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator+(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
          ) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] + std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs)[i] + std::get<1>(rhs)[i];
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator+(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] + std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs)[i] + std::get<1>(rhs)[i];
        r3[i] = std::get<2>(lhs)[i] + std::get<2>(rhs)[i];
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator+(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
          ) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] + std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs)[i] + std::get<1>(rhs)[i];
        r3[i] = std::get<2>(lhs)[i] + std::get<2>(rhs)[i];
        r4[i] = std::get<3>(lhs)[i] + std::get<3>(rhs)[i];
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// assignment ----------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<T, R, C>&,       matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] += std::get<0>(rhs)[i];
        std::get<1>(lhs)[i] += std::get<1>(rhs)[i];
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<T, R, C>&, matrix<T, R, C>&,
                     matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] += std::get<0>(rhs)[i];
        std::get<1>(lhs)[i] += std::get<1>(rhs)[i];
        std::get<2>(lhs)[i] += std::get<2>(rhs)[i];
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator+=(
    std::tuple<      matrix<T, R, C>&,       matrix<T, R, C>&,
                     matrix<T, R, C>&,       matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
    ) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] += std::get<0>(rhs)[i];
        std::get<1>(lhs)[i] += std::get<1>(rhs)[i];
        std::get<2>(lhs)[i] += std::get<2>(rhs)[i];
        std::get<3>(lhs)[i] += std::get<3>(rhs)[i];
    }
    return;
}

// ---------------------------------------------------------------------------
// subtraction operator-
// ---------------------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
operator-(const matrix<T, R, C>& lhs, const matrix<T, R, C>& rhs) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] - rhs[i];}
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
          ) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] - std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs)[i] - std::get<1>(rhs)[i];
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] - std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs)[i] - std::get<1>(rhs)[i];
        r3[i] = std::get<2>(lhs)[i] - std::get<2>(rhs)[i];
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
          ) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] - std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs)[i] - std::get<1>(rhs)[i];
        r3[i] = std::get<2>(lhs)[i] - std::get<2>(rhs)[i];
        r4[i] = std::get<3>(lhs)[i] - std::get<3>(rhs)[i];
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// assignment ----------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<T, R, C>&,       matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] -= std::get<0>(rhs)[i];
        std::get<1>(lhs)[i] -= std::get<1>(rhs)[i];
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<T, R, C>&, matrix<T, R, C>&,
                     matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] -= std::get<0>(rhs)[i];
        std::get<1>(lhs)[i] -= std::get<1>(rhs)[i];
        std::get<2>(lhs)[i] -= std::get<2>(rhs)[i];
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator-=(
    std::tuple<      matrix<T, R, C>&,       matrix<T, R, C>&,
                     matrix<T, R, C>&,       matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
    ) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] -= std::get<0>(rhs)[i];
        std::get<1>(lhs)[i] -= std::get<1>(rhs)[i];
        std::get<2>(lhs)[i] -= std::get<2>(rhs)[i];
        std::get<3>(lhs)[i] -= std::get<3>(rhs)[i];
    }
    return;
}

// ---------------------------------------------------------------------------
// multiplication operator*
// ---------------------------------------------------------------------------

// matrix * scalar -----------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
operator*(const matrix<T, R, C>& lhs, const T rhs) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] * rhs;}
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T> rhs) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] * std::get<0>(rhs);
        r2[i] = std::get<1>(lhs)[i] * std::get<1>(rhs);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] * std::get<0>(rhs);
        r2[i] = std::get<1>(lhs)[i] * std::get<1>(rhs);
        r3[i] = std::get<2>(lhs)[i] * std::get<2>(rhs);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>,
           matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T, T> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] * std::get<0>(rhs);
        r2[i] = std::get<1>(lhs)[i] * std::get<1>(rhs);
        r3[i] = std::get<2>(lhs)[i] * std::get<2>(rhs);
        r4[i] = std::get<3>(lhs)[i] * std::get<3>(rhs);
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// scalar * matrix -----------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
operator*(const T lhs, const matrix<T, R, C>& rhs) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs * rhs[i];}
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<T, T> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
          ) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs) * std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs) * std::get<1>(rhs)[i];
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<T, T, T> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs) * std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs) * std::get<1>(rhs)[i];
        r3[i] = std::get<2>(lhs) * std::get<2>(rhs)[i];
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator*(std::tuple<T, T, T, T> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> rhs
          ) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs) * std::get<0>(rhs)[i];
        r2[i] = std::get<1>(lhs) * std::get<1>(rhs)[i];
        r3[i] = std::get<2>(lhs) * std::get<2>(rhs)[i];
        r4[i] = std::get<3>(lhs) * std::get<3>(rhs)[i];
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// assignment ----------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator*=(
    std::tuple<matrix<T, R, C>&, matrix<T, R, C>&> lhs,
    std::tuple<T, T> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] *= std::get<0>(rhs);
        std::get<1>(lhs)[i] *= std::get<1>(rhs);
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator*=(
    std::tuple<matrix<T, R, C>&, matrix<T, R, C>&, matrix<T, R, C>&> lhs,
    std::tuple<T, T, T> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] *= std::get<0>(rhs);
        std::get<1>(lhs)[i] *= std::get<1>(rhs);
        std::get<2>(lhs)[i] *= std::get<2>(rhs);
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator*=(
    std::tuple<matrix<T, R, C>&, matrix<T, R, C>&,
               matrix<T, R, C>&, matrix<T, R, C>&> lhs,
    std::tuple<T, T, T, T> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] *= std::get<0>(rhs);
        std::get<1>(lhs)[i] *= std::get<1>(rhs);
        std::get<2>(lhs)[i] *= std::get<2>(rhs);
        std::get<3>(lhs)[i] *= std::get<3>(rhs);
    }
    return;
}

// ---------------------------------------------------------------------------
// division operator/
// ---------------------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
operator/(const matrix<T, R, C>& lhs, const T rhs) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] / rhs;}
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator/(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T> rhs) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] / std::get<0>(rhs);
        r2[i] = std::get<1>(lhs)[i] / std::get<1>(rhs);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator/(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] / std::get<0>(rhs);
        r2[i] = std::get<1>(lhs)[i] / std::get<1>(rhs);
        r3[i] = std::get<2>(lhs)[i] / std::get<2>(rhs);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator/(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T, T> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::get<0>(lhs)[i] / std::get<0>(rhs);
        r2[i] = std::get<1>(lhs)[i] / std::get<1>(rhs);
        r3[i] = std::get<2>(lhs)[i] / std::get<2>(rhs);
        r4[i] = std::get<3>(lhs)[i] / std::get<3>(rhs);
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// assignment ----------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator/=(
    std::tuple<      matrix<T, R, C>&,       matrix<T, R, C>&> lhs,
    std::tuple<T, T> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] /= std::get<0>(rhs);
        std::get<1>(lhs)[i] /= std::get<1>(rhs);
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator/=(
    std::tuple<      matrix<T, R, C>&, matrix<T, R, C>&,
                     matrix<T, R, C>&> lhs,
    std::tuple<T, T, T> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] /= std::get<0>(rhs);
        std::get<1>(lhs)[i] /= std::get<1>(rhs);
        std::get<2>(lhs)[i] /= std::get<2>(rhs);
    }
    return;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE void operator/=(
    std::tuple<      matrix<T, R, C>&,       matrix<T, R, C>&,
                     matrix<T, R, C>&,       matrix<T, R, C>&> lhs,
    std::tuple<T, T, T, T> rhs) noexcept
{
    for(std::size_t i=0; i<R*C; ++i)
    {
        std::get<0>(lhs)[i] /= std::get<0>(rhs);
        std::get<1>(lhs)[i] /= std::get<1>(rhs);
        std::get<2>(lhs)[i] /= std::get<2>(rhs);
        std::get<3>(lhs)[i] /= std::get<3>(rhs);
    }
    return;
}

// ---------------------------------------------------------------------------
// matrix multiplication operator-
// ---------------------------------------------------------------------------

template<typename T1, typename T2, std::size_t A, std::size_t B, std::size_t C>
MAVE_INLINE matrix<decltype(std::declval<T1>() * std::declval<T2>()), A, C>
operator*(const matrix<T1, A, B>& lhs, const matrix<T2, B, C>& rhs) noexcept
{
    matrix<decltype(std::declval<T1>() * std::declval<T2>()), A, C> retval;
    for(std::size_t i=0; i < A; ++i)
    {
        for(std::size_t j=0; j < C; ++j)
        {
            for(std::size_t k=0; k < B; ++k)
            {
                retval(i, j) += lhs(i, k) * rhs(k, j);
            }
        }
    }
    return retval;
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
max(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::max(v1[i], v2[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
max(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::max(std::get<0>(lhs)[i], std::get<0>(rhs)[i]);
        r2[i] = std::max(std::get<1>(lhs)[i], std::get<1>(rhs)[i]);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
max(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::max(std::get<0>(lhs)[i], std::get<0>(rhs)[i]);
        r2[i] = std::max(std::get<1>(lhs)[i], std::get<1>(rhs)[i]);
        r3[i] = std::max(std::get<2>(lhs)[i], std::get<2>(rhs)[i]);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
max(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::max(std::get<0>(lhs)[i], std::get<0>(rhs)[i]);
        r2[i] = std::max(std::get<1>(lhs)[i], std::get<1>(rhs)[i]);
        r3[i] = std::max(std::get<2>(lhs)[i], std::get<2>(rhs)[i]);
        r4[i] = std::max(std::get<3>(lhs)[i], std::get<3>(rhs)[i]);
    }
    return std::make_tuple(r1, r2, r3, r4);
}


template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
min(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::min(v1[i], v2[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
min(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::min(std::get<0>(lhs)[i], std::get<0>(rhs)[i]);
        r2[i] = std::min(std::get<1>(lhs)[i], std::get<1>(rhs)[i]);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
min(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::min(std::get<0>(lhs)[i], std::get<0>(rhs)[i]);
        r2[i] = std::min(std::get<1>(lhs)[i], std::get<1>(rhs)[i]);
        r3[i] = std::min(std::get<2>(lhs)[i], std::get<2>(rhs)[i]);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
min(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
    std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
               const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::min(std::get<0>(lhs)[i], std::get<0>(rhs)[i]);
        r2[i] = std::min(std::get<1>(lhs)[i], std::get<1>(rhs)[i]);
        r3[i] = std::min(std::get<2>(lhs)[i], std::get<2>(rhs)[i]);
        r4[i] = std::min(std::get<3>(lhs)[i], std::get<3>(rhs)[i]);
    }
    return std::make_tuple(r1, r2, r3, r4);
}


// floor ---------------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
floor(const matrix<T, R, C>& v) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::floor(v[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
floor(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept
{
    return std::make_pair(floor(v1), floor(v2));
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
floor(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
      const matrix<T, R, C>& v3) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3));
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,
                  matrix<T, R, C>, matrix<T, R, C>>
floor(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
      const matrix<T, R, C>& v3, const matrix<T, R, C>& v4) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3), floor(v4));
}

// ceil ----------------------------------------------------------------------

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
ceil(const matrix<T, R, C>& v) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::ceil(v[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>
ceil(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept
{
    return std::make_pair(ceil(v1), ceil(v2));
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
ceil(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
      const matrix<T, R, C>& v3) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3));
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,
                  matrix<T, R, C>, matrix<T, R, C>>
ceil(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
      const matrix<T, R, C>& v3, const matrix<T, R, C>& v4) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3), ceil(v4));
}

// ---------------------------------------------------------------------------
// Fused Multiply Add
// ---------------------------------------------------------------------------

// fmadd : a * b + c

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C> fmadd(
    const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::fma(a, b[i], c[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::pair<matrix<T, R, C>, matrix<T, R, C>>
fmadd(std::tuple<T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(std::get<0>(a), std::get<0>(b)[i], std::get<0>(c)[i]);
        r2[i] = std::fma(std::get<1>(a), std::get<1>(b)[i], std::get<1>(c)[i]);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fmadd(std::tuple<T, T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(std::get<0>(a), std::get<0>(b)[i], std::get<0>(c)[i]);
        r2[i] = std::fma(std::get<1>(a), std::get<1>(b)[i], std::get<1>(c)[i]);
        r3[i] = std::fma(std::get<2>(a), std::get<2>(b)[i], std::get<2>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fmadd(std::tuple<T, T, T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&, const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(std::get<0>(a), std::get<0>(b)[i], std::get<0>(c)[i]);
        r2[i] = std::fma(std::get<1>(a), std::get<1>(b)[i], std::get<1>(c)[i]);
        r3[i] = std::fma(std::get<2>(a), std::get<2>(b)[i], std::get<2>(c)[i]);
        r4[i] = std::fma(std::get<3>(a), std::get<3>(b)[i], std::get<3>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// fmsub : a * b - c

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C> fmsub(
    const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::fma(a, b[i], -c[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::pair<matrix<T, R, C>, matrix<T, R, C>>
fmsub(std::tuple<T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(std::get<0>(a), std::get<0>(b)[i], -std::get<0>(c)[i]);
        r2[i] = std::fma(std::get<1>(a), std::get<1>(b)[i], -std::get<1>(c)[i]);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fmsub(std::tuple<T, T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(std::get<0>(a), std::get<0>(b)[i], -std::get<0>(c)[i]);
        r2[i] = std::fma(std::get<1>(a), std::get<1>(b)[i], -std::get<1>(c)[i]);
        r3[i] = std::fma(std::get<2>(a), std::get<2>(b)[i], -std::get<2>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fmsub(std::tuple<T, T, T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&, const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(std::get<0>(a), std::get<0>(b)[i], -std::get<0>(c)[i]);
        r2[i] = std::fma(std::get<1>(a), std::get<1>(b)[i], -std::get<1>(c)[i]);
        r3[i] = std::fma(std::get<2>(a), std::get<2>(b)[i], -std::get<2>(c)[i]);
        r4[i] = std::fma(std::get<3>(a), std::get<3>(b)[i], -std::get<3>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// fnmadd : -a * b + c

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
fnmadd(const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::fma(-a, b[i], c[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::pair<matrix<T, R, C>, matrix<T, R, C>>
fnmadd(std::tuple<T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(-std::get<0>(a), std::get<0>(b)[i], std::get<0>(c)[i]);
        r2[i] = std::fma(-std::get<1>(a), std::get<1>(b)[i], std::get<1>(c)[i]);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fnmadd(std::tuple<T, T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(-std::get<0>(a), std::get<0>(b)[i], std::get<0>(c)[i]);
        r2[i] = std::fma(-std::get<1>(a), std::get<1>(b)[i], std::get<1>(c)[i]);
        r3[i] = std::fma(-std::get<2>(a), std::get<2>(b)[i], std::get<2>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fnmadd(std::tuple<T, T, T, T> a,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&, const matrix<T, R, C>&> b,
      std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                 const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(-std::get<0>(a), std::get<0>(b)[i], std::get<0>(c)[i]);
        r2[i] = std::fma(-std::get<1>(a), std::get<1>(b)[i], std::get<1>(c)[i]);
        r3[i] = std::fma(-std::get<2>(a), std::get<2>(b)[i], std::get<2>(c)[i]);
        r4[i] = std::fma(-std::get<3>(a), std::get<3>(b)[i], std::get<3>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3, r4);
}

// fnmsub : -a * b - c

template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE matrix<T, R, C>
fnmsub(const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept
{
    matrix<T, R, C> retval;
    for(std::size_t i=0; i<R*C; ++i)
    {
        retval[i] = std::fma(-a, b[i], -c[i]);
    }
    return retval;
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::pair<matrix<T, R, C>, matrix<T, R, C>>
fnmsub(std::tuple<T, T> a,
       std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> b,
       std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(-std::get<0>(a), std::get<0>(b)[i], -std::get<0>(c)[i]);
        r2[i] = std::fma(-std::get<1>(a), std::get<1>(b)[i], -std::get<1>(c)[i]);
    }
    return std::make_pair(r1, r2);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fnmsub(std::tuple<T, T, T> a,
       std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                  const matrix<T, R, C>&> b,
       std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                  const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(-std::get<0>(a), std::get<0>(b)[i], -std::get<0>(c)[i]);
        r2[i] = std::fma(-std::get<1>(a), std::get<1>(b)[i], -std::get<1>(c)[i]);
        r3[i] = std::fma(-std::get<2>(a), std::get<2>(b)[i], -std::get<2>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3);
}
template<typename T, std::size_t R, std::size_t C>
MAVE_INLINE
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
fnmsub(std::tuple<T, T, T, T> a,
       std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                  const matrix<T, R, C>&, const matrix<T, R, C>&> b,
       std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                  const matrix<T, R, C>&, const matrix<T, R, C>&> c) noexcept
{
    matrix<T, R, C> r1, r2, r3, r4;
    for(std::size_t i=0; i<R*C; ++i)
    {
        r1[i] = std::fma(-std::get<0>(a), std::get<0>(b)[i], -std::get<0>(c)[i]);
        r2[i] = std::fma(-std::get<1>(a), std::get<1>(b)[i], -std::get<1>(c)[i]);
        r3[i] = std::fma(-std::get<2>(a), std::get<2>(b)[i], -std::get<2>(c)[i]);
        r4[i] = std::fma(-std::get<3>(a), std::get<3>(b)[i], -std::get<3>(c)[i]);
    }
    return std::make_tuple(r1, r2, r3, r4);
}

} // mave
#endif // MAVE_MATRIX_HPP
