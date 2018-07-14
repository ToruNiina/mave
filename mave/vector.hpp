#ifndef MAVE_VECTOR_HPP
#define MAVE_VECTOR_HPP
#include "matrix.hpp"

// #ifdef __FMA__
// #include "fma/fma.hpp"
// #else
// #include "fma/no_fma.hpp"
// #endif //__FMA__

// #if defined(__AVX2__)
// #  include "avx2/vector3d.hpp"
// #  include "avx2/vector3f.hpp"
// #  include "avx2/vector3i.hpp"
// #elif defined(__AVX__)
#  include "avx/vector3d.hpp"
// #  include "avx/vector3f.hpp"
// #  include "avx/vector3i.hpp"
// #endif //__AVX__

namespace mave
{

template<typename T, std::size_t N>
using vector = matrix<T, N, 1>;

// length --------------------------------------------------------------------

template<typename T>
inline T length_sq(const vector<T, 3>& v) noexcept
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}
template<typename T>
inline T length(const vector<T, 3>& v) noexcept
{
    return std::sqrt(length_sq(v));
}

template<typename T>
inline std::pair<T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(length_sq(v1), length_sq(v2));
}
template<typename T>
inline std::pair<T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
{
    return std::make_pair(length(v1), length(v2));
}

template<typename T>
inline std::tuple<T, T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3) noexcept
{
    return std::make_tuple(length_sq(v1), length_sq(v2), length_sq(v3));
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
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(length_sq(v1), length_sq(v2), length_sq(v3), length_sq(v4));
}
template<typename T>
inline std::tuple<T, T, T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2,
       const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
{
    return std::make_tuple(length(v1), length(v2), length(v3), length(v4));
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
    return std::make_pair(v1 * (1.0 / length(v1), v2 * (1.0 / length(v2));
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
#endif // MAVE_MATH_VECTOR_HPP
