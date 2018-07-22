#ifndef MAVE_AVX2_VECTOR_MATRIX_MUL_HPP
#define MAVE_AVX2_VECTOR_MATRIX_MUL_HPP

#ifndef __AVX2__
#error "mave/avx2/matrix3x3d.hpp requires avx support but __AVX2__ is not defined."
#endif

#include "matrix3x3d.hpp"
#include "matrix3x3f.hpp"
#include "vector3d.hpp"
#include "vector3f.hpp"

namespace mave
{

// ---------------------------------------------------------------------------
// matrix3x3-vector3 multiplication -- double
// ---------------------------------------------------------------------------

template<>
inline matrix<double, 3, 1> operator*(
    const matrix<double, 3, 3>& m, const matrix<double, 3, 1>& v) noexcept
{
    const __m256d v0 = _mm256_load_pd(v.data());

    matrix<double, 3, 1> retval;
    alignas(32) double pack[4];
    _mm256_store_pd(pack, _mm256_mul_pd(_mm256_load_pd(m.data()  ), v0));
    retval[0] = pack[0] + pack[1] + pack[2];
    _mm256_store_pd(pack, _mm256_mul_pd(_mm256_load_pd(m.data()+4), v0));
    retval[1] = pack[0] + pack[1] + pack[2];
    _mm256_store_pd(pack, _mm256_mul_pd(_mm256_load_pd(m.data()+8), v0));
    retval[2] = pack[0] + pack[1] + pack[2];
    return retval;
}

// ---------------------------------------------------------------------------
// matrix3x3-vector3 multiplication -- float
// ---------------------------------------------------------------------------

template<>
inline matrix<float, 3, 1> operator*(
    const matrix<float, 3, 3>& m, const matrix<float, 3, 1>& v) noexcept
{
    const __m128 v128 = _mm_load_ps(v.data());
    const __m256 v256 = _mm256_insertf128_ps(_mm256_castps128_ps256(v128), v128, 1);

    alignas(32) float pack[8];
    matrix<float, 3, 1> retval;

    _mm256_store_ps(pack, _mm256_mul_ps(_mm256_load_ps(m.data()), v256));
    retval[0] = pack[0] + pack[1] + pack[2];
    retval[1] = pack[4] + pack[5] + pack[6];

    _mm_store_ps(pack, _mm_mul_ps(_mm_load_ps(m.data()+8), v128));
    retval[2] = pack[0] + pack[1] + pack[2];
    return retval;
}

} // mave
#endif // MAVE_AVX2_VECTOR_MATRIX_MUL_HPP
