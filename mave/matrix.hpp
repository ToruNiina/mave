#ifndef MAVE_MATRIX_HPP
#define MAVE_MATRIX_HPP
#include "type_traits.hpp"
#include <array>
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
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;

    template<typename ... Ts, typename std::enable_if<
        sizeof...(Ts) == total_size &&
        conjunction<std::is_convertible<Ts, T> ...>::value,
        std::nullptr_t>::type = nullptr>
    matrix(Ts&& ... args) noexcept : vs_{{static_cast<T>(args)...}}{}

    matrix() = default;
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

    reference               at(size_type i)       {return vs_.at(i);}
    const_reference         at(size_type i) const {return vs_.at(i);}
    reference       operator[](size_type i)       noexcept {return vs_[i];}
    const_reference operator[](size_type i) const noexcept {return vs_[i];}

    reference       at(size_type i, size_type j)       {return vs_.at(i*C+j);}
    const_reference at(size_type i, size_type j) const {return vs_.at(i*C+j);}
    reference       operator()(size_type i, size_type j)       noexcept {return vs_[i*C+j];}
    const_reference operator()(size_type i, size_type j) const noexcept {return vs_[i*C+j];}

  private:
    storage_type vs_;
};

template<typename T, std::size_t R, std::size_t C>
constexpr std::size_t matrix<T, R, C>::row_size;
template<typename T, std::size_t R, std::size_t C>
constexpr std::size_t matrix<T, R, C>::column_size;
template<typename T, std::size_t R, std::size_t C>
constexpr std::size_t matrix<T, R, C>::total_size;

template<typename T1, typename T2, std::size_t R, std::size_t C>
inline matrix<decltype(std::declval<T1>() + std::declval<T2>()), R, C>
operator+(const matrix<T1, R, C>& lhs, const matrix<T2, R, C>& rhs) noexcept
{
    matrix<decltype(std::declval<T1>() + std::declval<T2>()), R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] + rhs[i];}
    return retval;
}
template<typename T1, typename T2, std::size_t R, std::size_t C>
inline matrix<decltype(std::declval<T1>() - std::declval<T2>()), R, C>
operator-(const matrix<T1, R, C>& lhs, const matrix<T2, R, C>& rhs) noexcept
{
    matrix<decltype(std::declval<T1>() - std::declval<T2>()), R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] - rhs[i];}
    return retval;
}
template<typename T1, typename T2, std::size_t R, std::size_t C>
inline matrix<decltype(std::declval<T1>() * std::declval<T2>()), R, C>
operator*(const matrix<T1, R, C>& lhs, const T2 rhs) noexcept
{
    matrix<decltype(std::declval<T1>() * std::declval<T2>()), R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] * rhs;}
    return retval;
}
template<typename T1, typename T2, std::size_t R, std::size_t C>
inline matrix<decltype(std::declval<T1>() * std::declval<T2>()), R, C>
operator*(const T1 lhs, const matrix<T2, R, C>& rhs) noexcept
{
    matrix<decltype(std::declval<T1>() * std::declval<T2>()), R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs * rhs[i];}
    return retval;
}
template<typename T1, typename T2, std::size_t R, std::size_t C>
inline matrix<decltype(std::declval<T1>() * std::declval<T2>()), R, C>
operator/(const matrix<T1, R, C>& lhs, const T2 rhs) noexcept
{
    matrix<decltype(std::declval<T1>() * std::declval<T2>()), R, C> retval;
    for(std::size_t i=0; i<R*C; ++i) {retval[i] = lhs[i] / rhs;}
    return retval;
}

template<typename T1, typename T2, std::size_t A, std::size_t B, std::size_t C>
inline matrix<decltype(std::declval<T1>() * std::declval<T2>()), A, C>
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

} // mave
#endif
