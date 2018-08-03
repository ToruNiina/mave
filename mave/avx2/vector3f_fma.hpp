#ifndef MAVE_AVX2_VECTOR_3_FLOAT_FMA_HPP
#define MAVE_AVX2_VECTOR_3_FLOAT_FMA_HPP

#ifndef __FMA__
#error "mave/fma.hpp requires fma support but __FMA__ is not defined."
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

// ---------------------------------------------------------------------------
// fmadd
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> fmadd(const float a,
    const matrix<float, 3, 1>& b, const matrix<float, 3, 1>& c) noexcept
{
    return _mm_fmadd_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                        _mm_load_ps(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
fmadd(std::tuple<float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const __m256 a01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1);
    const __m256 b01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1);
    const __m256 c01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1);

    const __m256 r01 = _mm256_fmadd_ps(a01, b01, c01);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(r01)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(r01, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fmadd(std::tuple<float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fmadd(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
        fmadd(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fmadd(std::tuple<float, float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fmadd(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fmadd(
            std::tuple<float, float>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// fmsub
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> fmsub(const float a,
    const matrix<float, 3, 1>& b, const matrix<float, 3, 1>& c) noexcept
{
    return _mm_fmsub_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                        _mm_load_ps(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
fmsub(std::tuple<float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const __m256 a01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1);
    const __m256 b01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1);
    const __m256 c01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1);

    const __m256 r01 = _mm256_fmsub_ps(a01, b01, c01);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(r01)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(r01, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fmsub(std::tuple<float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fmsub(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
        fmsub(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fmsub(std::tuple<float, float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fmsub(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fmsub(
            std::tuple<float, float>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}


// ---------------------------------------------------------------------------
// fnmadd
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> fnmadd(const float a,
    const matrix<float, 3, 1>& b, const matrix<float, 3, 1>& c) noexcept
{
    return _mm_fnmadd_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                         _mm_load_ps(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
fnmadd(std::tuple<float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const __m256 a01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1);
    const __m256 b01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1);
    const __m256 c01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1);

    const __m256 r01 = _mm256_fnmadd_ps(a01, b01, c01);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(r01)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(r01, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fnmadd(std::tuple<float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fnmadd(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
        fnmadd(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fnmadd(std::tuple<float, float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fnmadd(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fnmadd(
            std::tuple<float, float>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}


// ---------------------------------------------------------------------------
// fnmsub
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<float, 3, 1> fnmsub(const float a,
    const matrix<float, 3, 1>& b, const matrix<float, 3, 1>& c) noexcept
{
    return _mm_fnmsub_ps(_mm_set1_ps(a), _mm_load_ps(b.data()),
                         _mm_load_ps(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<float, 3, 1>, matrix<float, 3, 1>>
fnmsub(std::tuple<float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const __m256 a01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1);
    const __m256 b01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1);
    const __m256 c01 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1);

    const __m256 r01 = _mm256_fnmsub_ps(a01, b01, c01);

    return std::make_pair(matrix<float, 3, 1>(_mm256_castps256_ps128(r01)),
                          matrix<float, 3, 1>(_mm256_extractf128_ps(r01, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fnmsub(std::tuple<float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fnmsub(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
        fnmsub(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>, matrix<float, 3, 1>>
fnmsub(std::tuple<float, float, float, float> a,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> b,
      std::tuple<const matrix<float, 3, 1>&, const matrix<float, 3, 1>&,
                 const matrix<float, 3, 1>&, const matrix<float, 3, 1>&> c) noexcept
{
    const auto r12 = fnmsub(
            std::tuple<float, float>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fnmsub(
            std::tuple<float, float>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}



} // mave
#endif // MAVE_FMA_FMA_HPP
