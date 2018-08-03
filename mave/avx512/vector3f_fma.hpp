#ifndef MAVE_AVX512_VECTOR_3_FLOAT_FMA_HPP
#define MAVE_AVX512_VECTOR_3_FLOAT_FMA_HPP

#if !defined(__FMA__) || !defined(__AVX512F__)
#error "mave/fma.hpp requires fma and avx512f support."
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
    const __m512 a123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2);

    const __m512 b123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2);

    const __m512 c123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2);

    const __m512 rslt = _mm512_fmadd_ps(a123, b123, c123);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)));
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
    const __m512 a1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2),
                               _mm_set1_ps(std::get<3>(a)), 3);

    const __m512 b1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2),
                               _mm_load_ps(std::get<3>(b).data()), 3);

    const __m512 c1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2),
                               _mm_load_ps(std::get<3>(c).data()), 3);

    const __m512 rslt = _mm512_fmadd_ps(a1234, b1234, c1234);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 3)));
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
    const __m512 a123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2);

    const __m512 b123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2);

    const __m512 c123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2);

    const __m512 rslt = _mm512_fmsub_ps(a123, b123, c123);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)));
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
    const __m512 a1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2),
                               _mm_set1_ps(std::get<3>(a)), 3);

    const __m512 b1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2),
                               _mm_load_ps(std::get<3>(b).data()), 3);

    const __m512 c1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2),
                               _mm_load_ps(std::get<3>(c).data()), 3);

    const __m512 rslt = _mm512_fmsub_ps(a1234, b1234, c1234);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 3)));
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
    const __m512 a123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2);

    const __m512 b123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2);

    const __m512 c123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2);

    const __m512 rslt = _mm512_fnmadd_ps(a123, b123, c123);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)));
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
    const __m512 a1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2),
                               _mm_set1_ps(std::get<3>(a)), 3);

    const __m512 b1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2),
                               _mm_load_ps(std::get<3>(b).data()), 3);

    const __m512 c1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2),
                               _mm_load_ps(std::get<3>(c).data()), 3);

    const __m512 rslt = _mm512_fnmadd_ps(a1234, b1234, c1234);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 3)));
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
    const __m512 a123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2);

    const __m512 b123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2);

    const __m512 c123 = _mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2);

    const __m512 rslt = _mm512_fnmsub_ps(a123, b123, c123);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)));
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
    const __m512 a1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_set1_ps(std::get<0>(a))),
                               _mm_set1_ps(std::get<1>(a)), 1),
                               _mm_set1_ps(std::get<2>(a)), 2),
                               _mm_set1_ps(std::get<3>(a)), 3);

    const __m512 b1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(b).data())),
                               _mm_load_ps(std::get<1>(b).data()), 1),
                               _mm_load_ps(std::get<2>(b).data()), 2),
                               _mm_load_ps(std::get<3>(b).data()), 3);

    const __m512 c1234 = _mm512_insertf32x4(_mm512_insertf32x4(_mm512_insertf32x4(
        _mm512_castps128_ps512(_mm_load_ps(std::get<0>(c).data())),
                               _mm_load_ps(std::get<1>(c).data()), 1),
                               _mm_load_ps(std::get<2>(c).data()), 2),
                               _mm_load_ps(std::get<3>(c).data()), 3);

    const __m512 rslt = _mm512_fnmsub_ps(a1234, b1234, c1234);

    return std::make_tuple(matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 0)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 1)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 2)),
                           matrix<float, 3, 1>(_mm512_extractf32x4_ps(rslt, 3)));
}

} // mave
#endif // MAVE_FMA_FMA_HPP
