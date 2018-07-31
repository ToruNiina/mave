#ifndef MAVE_VECTOR_HPP
#define MAVE_VECTOR_HPP
#include "matrix.hpp"
#include <tuple>

namespace mave
{

// for usability
template<typename T, std::size_t N>
using vector = matrix<T, N, 1>;

// here, mave defines fallback functions. in this file, no SIMD operations are
// used. if there are no supported SIMD, this implementations are used. if
// SIMD operations are supported, some specialization of these functions are
// overwritten by altanative implementations that uses SIMD intrinsics.

// ---------------------------------------------------------------------------
// length
// ---------------------------------------------------------------------------

// length_sq -----------------------------------------------------------------

template<typename T>
inline T length_sq(const vector<T, 3>& v) noexcept
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}
template<typename T>
inline std::pair<T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(length_sq(v1), length_sq(v2));
}
template<typename T>
inline std::tuple<T, T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3) noexcept
{
    return std::make_tuple(length_sq(v1), length_sq(v2), length_sq(v3));
}
template<typename T>
inline std::tuple<T, T, T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(length_sq(v1), length_sq(v2),
                           length_sq(v3), length_sq(v4));
}

// length -------------------------------------------------------------------

template<typename T>
inline T length(const vector<T, 3>& v) noexcept
{
    return std::sqrt(length_sq(v));
}
template<typename T>
inline std::pair<T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(length(v1), length(v2));
}
template<typename T>
inline std::tuple<T, T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2,
       const vector<T, 3>& v3) noexcept
{
    return std::make_tuple(length(v1), length(v2), length(v3));
}
template<typename T>
inline std::tuple<T, T, T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2,
       const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(length(v1), length(v2), length(v3), length(v4));
}

// rlength -------------------------------------------------------------------

template<typename T>
inline T rlength(const vector<T, 3>& v) noexcept
{
    return 1.0 / std::sqrt(length_sq(v));
}
template<typename T>
inline std::pair<T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(1.0 / std::sqrt(length_sq(v1)),
                          1.0 / std::sqrt(length_sq(v2)));
}
template<typename T>
inline std::tuple<T, T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2,
        const vector<T, 3>& v3) noexcept
{
    return std::make_tuple(1.0 / std::sqrt(length_sq(v1)),
                           1.0 / std::sqrt(length_sq(v2)),
                           1.0 / std::sqrt(length_sq(v3)));
}
template<typename T>
inline std::tuple<T, T, T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2,
        const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(1.0 / std::sqrt(length_sq(v1)),
                           1.0 / std::sqrt(length_sq(v2)),
                           1.0 / std::sqrt(length_sq(v3)),
                           1.0 / std::sqrt(length_sq(v4)));
}

// regularize ----------------------------------------------------------------

template<typename T>
inline std::pair<vector<T, 3>, T>
regularize(const vector<T, 3>& v) noexcept
{
    const auto l = length(v);
    return std::make_pair(v * (1.0 / l), l);
}

template<typename T>
inline std::pair<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    const auto l = length(v1, v2);
    return std::make_pair(
            std::make_pair(v1 * (1.0 / std::get<0>(l)), std::get<0>(l)),
            std::make_pair(v2 * (1.0 / std::get<1>(l)), std::get<1>(l)));
}

template<typename T>
inline std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,
                  std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3) noexcept
{
    const auto l = length(v1, v2, v3);
    return std::make_tuple(
            std::make_pair(v1 * (1.0 / std::get<0>(l)), std::get<0>(l)),
            std::make_pair(v2 * (1.0 / std::get<1>(l)), std::get<1>(l)),
            std::make_pair(v3 * (1.0 / std::get<2>(l)), std::get<2>(l)));
}

template<typename T>
inline std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,
                  std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    const auto l = length(v1, v2, v3, v4);
    return std::make_tuple(
            std::make_pair(v1 * (1.0 / std::get<0>(l)), std::get<0>(l)),
            std::make_pair(v2 * (1.0 / std::get<1>(l)), std::get<1>(l)),
            std::make_pair(v3 * (1.0 / std::get<2>(l)), std::get<2>(l)),
            std::make_pair(v4 * (1.0 / std::get<3>(l)), std::get<3>(l)));
}

// ---------------------------------------------------------------------------
// 3D vector operations
// ---------------------------------------------------------------------------

template<typename T>
inline T dot_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept
{
    return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
}
template<typename T>
inline std::pair<T, T>
dot_product(std::tuple<const vector<T, 3>&, const vector<T, 3>&> lhs,
            std::tuple<const vector<T, 3>&, const vector<T, 3>&> rhs) noexcept
{
    return std::make_pair(
            std::get<0>(lhs)[0] * std::get<0>(rhs)[0] +
            std::get<0>(lhs)[1] * std::get<0>(rhs)[1] +
            std::get<0>(lhs)[2] * std::get<0>(rhs)[2],
            std::get<1>(lhs)[0] * std::get<1>(rhs)[0] +
            std::get<1>(lhs)[1] * std::get<1>(rhs)[1] +
            std::get<1>(lhs)[2] * std::get<1>(rhs)[2]);
}
template<typename T>
inline std::tuple<T, T, T>
dot_product(std::tuple<const vector<T, 3>&, const vector<T, 3>&,
                       const vector<T, 3>&> lhs,
            std::tuple<const vector<T, 3>&, const vector<T, 3>&,
                       const vector<T, 3>&> rhs) noexcept
{
    return std::make_tuple(
            std::get<0>(lhs)[0] * std::get<0>(rhs)[0] +
            std::get<0>(lhs)[1] * std::get<0>(rhs)[1] +
            std::get<0>(lhs)[2] * std::get<0>(rhs)[2],
            std::get<1>(lhs)[0] * std::get<1>(rhs)[0] +
            std::get<1>(lhs)[1] * std::get<1>(rhs)[1] +
            std::get<1>(lhs)[2] * std::get<1>(rhs)[2],
            std::get<2>(lhs)[0] * std::get<2>(rhs)[0] +
            std::get<2>(lhs)[1] * std::get<2>(rhs)[1] +
            std::get<2>(lhs)[2] * std::get<2>(rhs)[2]
            );
}
template<typename T>
inline std::tuple<T, T, T, T>
dot_product(std::tuple<const vector<T, 3>&, const vector<T, 3>&,
                       const vector<T, 3>&, const vector<T, 3>&> lhs,
            std::tuple<const vector<T, 3>&, const vector<T, 3>&,
                       const vector<T, 3>&, const vector<T, 3>&> rhs) noexcept
{
    return std::make_tuple(
            std::get<0>(lhs)[0] * std::get<0>(rhs)[0] +
            std::get<0>(lhs)[1] * std::get<0>(rhs)[1] +
            std::get<0>(lhs)[2] * std::get<0>(rhs)[2],
            std::get<1>(lhs)[0] * std::get<1>(rhs)[0] +
            std::get<1>(lhs)[1] * std::get<1>(rhs)[1] +
            std::get<1>(lhs)[2] * std::get<1>(rhs)[2],
            std::get<2>(lhs)[0] * std::get<2>(rhs)[0] +
            std::get<2>(lhs)[1] * std::get<2>(rhs)[1] +
            std::get<2>(lhs)[2] * std::get<2>(rhs)[2],
            std::get<3>(lhs)[0] * std::get<3>(rhs)[0] +
            std::get<3>(lhs)[1] * std::get<3>(rhs)[1] +
            std::get<3>(lhs)[2] * std::get<3>(rhs)[2]);
}

template<typename T>
inline vector<T, 3>
cross_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept
{
    return vector<T, 3>(lhs[1] * rhs[2] - lhs[2] * rhs[1],
                        lhs[2] * rhs[0] - lhs[0] * rhs[2],
                        lhs[0] * rhs[1] - lhs[1] * rhs[0]);
}
template<typename T>
inline T scalar_triple_product(
    const vector<T, 3>& lhs, const vector<T, 3>& mid, const vector<T, 3>& rhs
    ) noexcept
{
    return (lhs[1] * mid[2] - lhs[2] * mid[1]) * rhs[0] +
           (lhs[2] * mid[0] - lhs[0] * mid[2]) * rhs[1] +
           (lhs[0] * mid[1] - lhs[1] * mid[0]) * rhs[2];
}

} // mave
#endif // MAVE_MATH_VECTOR_HPP
