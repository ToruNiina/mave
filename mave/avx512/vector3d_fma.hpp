#ifndef MAVE_AVX512_VECTOR_3_DOUBLE_FMA_HPP
#define MAVE_AVX512_VECTOR_3_DOUBLE_FMA_HPP

#if !defined(__FMA__) || !defined(__AVX512F__)
#error "mave/fma.hpp requires fma and avx512f support.
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
MAVE_INLINE matrix<double, 3, 1> fmadd(const double a,
    const matrix<double, 3, 1>& b, const matrix<double, 3, 1>& c) noexcept
{
    return _mm256_fmadd_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                           _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<double, 3, 1>, matrix<double, 3, 1>>
fmadd(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const __m512d a12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_set1_pd(std::get<0>(a))),
                               _mm256_set1_pd(std::get<1>(a)), 1);
    const __m512d b12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(b).data())),
                               _mm256_load_pd(std::get<1>(b).data()), 1);
    const __m512d c12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(c).data())),
                               _mm256_load_pd(std::get<1>(c).data()), 1);

    const __m512d rslt = _mm512_fmadd_pd(a12, b12, c12);

    return std::make_pair(matrix<double, 3, 1>(_mm512_castpd512_pd256(rslt)),
                          matrix<double, 3, 1>(_mm512_extractf64x4_pd(rslt, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fmadd(std::tuple<double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fmadd(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
            fmadd(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fmadd(std::tuple<double, double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fmadd(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fmadd(
          std::tuple<double, double>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// fmsub
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> fmsub(const double a,
    const matrix<double, 3, 1>& b, const matrix<double, 3, 1>& c) noexcept
{
    return _mm256_fmsub_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                           _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<double, 3, 1>, matrix<double, 3, 1>>
fmsub(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const __m512d a12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_set1_pd(std::get<0>(a))),
                               _mm256_set1_pd(std::get<1>(a)), 1);
    const __m512d b12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(b).data())),
                               _mm256_load_pd(std::get<1>(b).data()), 1);
    const __m512d c12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(c).data())),
                               _mm256_load_pd(std::get<1>(c).data()), 1);

    const __m512d rslt = _mm512_fmsub_pd(a12, b12, c12);

    return std::make_pair(matrix<double, 3, 1>(_mm512_castpd512_pd256(rslt)),
                          matrix<double, 3, 1>(_mm512_extractf64x4_pd(rslt, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fmsub(std::tuple<double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fmsub(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
            fmsub(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fmsub(std::tuple<double, double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fmsub(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fmsub(
          std::tuple<double, double>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// fnmadd
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> fnmadd(const double a,
    const matrix<double, 3, 1>& b, const matrix<double, 3, 1>& c) noexcept
{
    return _mm256_fnmadd_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                            _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmadd(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const __m512d a12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_set1_pd(std::get<0>(a))),
                               _mm256_set1_pd(std::get<1>(a)), 1);
    const __m512d b12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(b).data())),
                               _mm256_load_pd(std::get<1>(b).data()), 1);
    const __m512d c12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(c).data())),
                               _mm256_load_pd(std::get<1>(c).data()), 1);

    const __m512d rslt = _mm512_fnmadd_pd(a12, b12, c12);

    return std::make_pair(matrix<double, 3, 1>(_mm512_castpd512_pd256(rslt)),
                          matrix<double, 3, 1>(_mm512_extractf64x4_pd(rslt, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmadd(std::tuple<double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fnmadd(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
            fnmadd(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmadd(std::tuple<double, double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fnmadd(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fnmadd(
          std::tuple<double, double>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

// ---------------------------------------------------------------------------
// fnmsub
// ---------------------------------------------------------------------------

template<>
MAVE_INLINE matrix<double, 3, 1> fnmsub(const double a,
    const matrix<double, 3, 1>& b, const matrix<double, 3, 1>& c) noexcept
{
    return _mm256_fnmsub_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                            _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE
std::pair<matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmsub(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const __m512d a12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_set1_pd(std::get<0>(a))),
                               _mm256_set1_pd(std::get<1>(a)), 1);
    const __m512d b12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(b).data())),
                               _mm256_load_pd(std::get<1>(b).data()), 1);
    const __m512d c12 = _mm512_insertf64x4(
        _mm512_castpd256_pd512(_mm256_load_pd(std::get<0>(c).data())),
                               _mm256_load_pd(std::get<1>(c).data()), 1);

    const __m512d rslt = _mm512_fnmsub_pd(a12, b12, c12);

    return std::make_pair(matrix<double, 3, 1>(_mm512_castpd512_pd256(rslt)),
                          matrix<double, 3, 1>(_mm512_extractf64x4_pd(rslt, 1)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmsub(std::tuple<double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fnmsub(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
            fnmsub(std::get<2>(a), std::get<2>(b), std::get<2>(c)));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmsub(std::tuple<double, double, double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&,
                 const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    const auto r12 = fnmsub(
          std::tuple<double, double>(std::get<0>(a), std::get<1>(a)),
                            std::tie(std::get<0>(b), std::get<1>(b)),
                            std::tie(std::get<0>(c), std::get<1>(c)));
    const auto r34 = fnmsub(
          std::tuple<double, double>(std::get<2>(a), std::get<3>(a)),
                            std::tie(std::get<2>(b), std::get<3>(b)),
                            std::tie(std::get<2>(c), std::get<3>(c)));

    return std::make_tuple(std::get<0>(r12), std::get<1>(r12),
                           std::get<0>(r34), std::get<1>(r34));
}

} // mave
#endif // MAVE_FMA_FMA_HPP
