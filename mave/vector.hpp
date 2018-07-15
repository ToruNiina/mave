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
inline vector<T, 3> regularize(const vector<T, 3>& v) noexcept
{
    return v * (1.0 / length(v));
}

template<typename T>
inline std::pair<vector<T, 3>, vector<T, 3>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(v1 * (1.0 / length(v1), v2 * (1.0 / length(v2))));
}

template<typename T>
inline std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3) noexcept
{
    return std::make_tuple(v1 * (1.0 / length(v1)), v2 * (1.0 / length(v2)),
                           v3 * (1.0 / length(v3)));
}

template<typename T>
inline std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>, vector<T, 3>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(v1 * (1.0 / length(v1)), v2 * (1.0 / length(v2)),
                           v3 * (1.0 / length(v3)), v4 * (1.0 / length(v4)));
}

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

template<typename T>
inline vector<T, 3> max(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return vector<T, 3>(std::max(v1[0], v2[0]), std::max(v1[1], v2[1]),
                        std::max(v1[2], v2[2]));
}

template<typename T>
inline vector<T, 3> min(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return vector<T, 3>(std::min(v1[0], v2[0]), std::min(v1[1], v2[1]),
                        std::min(v1[2], v2[2]));
}

// floor ---------------------------------------------------------------------

template<typename T>
inline vector<T, 3> floor(const vector<T, 3>& v) noexcept
{
    return vector<T, 3>(std::floor(v[0]), std::floor(v[1]), std::floor(v[2]));
}
template<typename T>
inline std::pair<vector<T, 3>, vector<T, 3>>
floor(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(floor(v1), floor(v2));
}
template<typename T>
inline std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>>
floor(const vector<T, 3>& v1, const vector<T, 3>& v2,
      const vector<T, 3>& v3) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3));
}
template<typename T>
inline std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>, vector<T, 3>>
floor(const vector<T, 3>& v1, const vector<T, 3>& v2,
      const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(floor(v1), floor(v2), floor(v3), floor(v4));
}

// ceil ----------------------------------------------------------------------

template<typename T>
inline vector<T, 3> ceil(const vector<T, 3>& v) noexcept
{
    return vector<T, 3>(std::ceil(v[0]), std::ceil(v[1]), std::ceil(v[2]));
}
template<typename T>
inline std::pair<vector<T, 3>, vector<T, 3>>
ceil(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(ceil(v1), ceil(v2));
}
template<typename T>
inline std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>>
ceil(const vector<T, 3>& v1, const vector<T, 3>& v2,
     const vector<T, 3>& v3) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3));
}
template<typename T>
inline std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>, vector<T, 3>>
ceil(const vector<T, 3>& v1, const vector<T, 3>& v2,
     const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(ceil(v1), ceil(v2), ceil(v3), ceil(v4));
}

// ---------------------------------------------------------------------------
// Fused Multiply Add
// ---------------------------------------------------------------------------

template<typename T>
inline vector<T, 3> fmadd(
    const T a, const vector<T, 3>& b, const vector<T, 3>& c) noexcept
{
    return a * b + c;
}
template<typename T>
inline vector<T, 3> fmsub(
    const T a, const vector<T, 3>& b, const vector<T, 3>& c) noexcept
{
    return a * b - c;
}

template<typename T>
inline vector<T, 3> fnmadd(
    const T a, const vector<T, 3>& b, const vector<T, 3>& c) noexcept
{
    return -a * b + c;
}
template<typename T>
inline vector<T, 3> fnmsub(
    const T a, const vector<T, 3>& b, const vector<T, 3>& c) noexcept
{
    return -a * b - c;
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

#if defined(__AVX__)
#  include "avx/vector3d.hpp"
// #  include "avx/vector3f.hpp"
// #  include "avx/vector3i.hpp"
#endif //__AVX__

// #if __FMA__
// #include "fma/fma.hpp"
// #endif //__FMA__

#endif // MAVE_MATH_VECTOR_HPP
