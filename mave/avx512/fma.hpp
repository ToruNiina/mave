#ifndef MAVE_AVX512_FMA_HPP
#define MAVE_AVX512_FMA_HPP

#ifndef __FMA__
#error "mave/avx512/fma.hpp requires fma support but __FMA__ is not defined."
#endif

#ifndef MAVE_VECTOR_HPP
#error "do not use mave/fma/fma.hpp alone. please include mave/mave.hpp."
#endif

#include <x86intrin.h> // for *nix
#include <type_traits>
#include <array>
#include <cmath>


namespace mave
{

// vector3d ------------------------------------------------------------------

template<>
MAVE_INLINE vector<double, 3> fmadd(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fmadd_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                           _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE vector<double, 3> fmsub(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fmsub_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                           _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE vector<double, 3> fnmadd(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fnmadd_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                            _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE vector<double, 3> fnmsub(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fnmsub_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                            _mm256_load_pd(c.data()));
}

// vector3f ------------------------------------------------------------------

template<>
MAVE_INLINE vector<float, 3> fmadd(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fmadd_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                        _mm_load_ps(c.data()));
}
template<>
MAVE_INLINE vector<float, 3> fmsub(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fmsub_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                        _mm_load_ps(c.data()));
}
template<>
MAVE_INLINE vector<float, 3> fnmadd(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fnmadd_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                         _mm_load_ps(c.data()));
}
template<>
MAVE_INLINE vector<float, 3> fnmsub(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fnmsub_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                         _mm_load_ps(c.data()));
}

} // mave
#endif // MAVE_FMA_FMA_HPP
