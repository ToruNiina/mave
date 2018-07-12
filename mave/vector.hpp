#ifndef MAVE_VECTOR_HPP
#define MAVE_VECTOR_HPP
#include "matrix.hpp"

#ifdef __AVX__
#include "avx/vector3d.hpp"
#endif //__AVX__

namespace mave
{

template<typename T, std::size_t N>
using vector = matrix<T, N, 1>;

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
inline T dot_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept
{
    return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
}
template<typename T>
inline vector<T, 3> cross_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept
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

template<typename T>
inline vector<T, 3> regularize(const vector<T, 3>& lhs) noexcept
{
    const T rl = 1.0 / length(lhs);
    return vector<T, 3>(lhs[0] * rl, lhs[1] * rl, lhs[2] * rl);
}

} // mave
#endif // MAVE_MATH_VECTOR_HPP
