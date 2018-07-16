#ifndef MAVE_FMA_FMA_HPP
#define MAVE_FMA_FMA_HPP

#ifndef __FMA__
#error "mave/fma/fma.hpp requires fma support but __FMA__ is not defined."
#endif
#ifndef __AVX2__
#error "mave assumes __FMA__ support includes __AVX2__"
#endif

#ifndef MAVE_VECTOR_HPP
#error "do not use mave/fma/fma.hpp alone. please include mave/vector.hpp."
#endif

#include <x86intrin.h> // for *nix
#include <type_traits>
#include <array>
#include <cmath>


namespace mave
{

// vector3d ------------------------------------------------------------------

template<>
inline vector<double, 3> fmadd(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fmadd_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                           _mm256_load_pd(c.data()));
}
template<>
inline vector<double, 3> fmsub(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fmsub_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                           _mm256_load_pd(c.data()));
}
template<>
inline vector<double, 3> fnmadd(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fnmadd_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                            _mm256_load_pd(c.data()));
}
template<>
inline vector<double, 3> fnmsub(const double a,
    const vector<double, 3>& b, const vector<double, 3>& c) noexcept
{
    return _mm256_fnmsub_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                            _mm256_load_pd(c.data()));
}

// vector3f ------------------------------------------------------------------

template<>
inline vector<float, 3> fmadd(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fmadd_ps(_mm_set1_pd(a), _mm_load_pd(b.data()),
                        _mm_load_pd(c.data()));
}
template<>
inline vector<float, 3> fmsub(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fmsub_ps(_mm_set1_pd(a), _mm_load_pd(b.data()),
                        _mm_load_pd(c.data()));
}
template<>
inline vector<float, 3> fnmadd(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fnmadd_ps(_mm_set1_pd(a), _mm_load_pd(b.data()),
                         _mm_load_pd(c.data()));
}
template<>
inline vector<float, 3> fnmsub(const float a,
    const vector<float, 3>& b, const vector<float, 3>& c) noexcept
{
    return _mm_fnmsub_ps(_mm_set1_pd(a), _mm_load_pd(b.data()),
                         _mm_load_pd(c.data()));
}

} // mave
#endif // MAVE_FMA_FMA_HPP
