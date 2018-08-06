#ifndef MAVE_AVX2_VECTOR_3_DOUBLE_FMA_HPP
#define MAVE_AVX2_VECTOR_3_DOUBLE_FMA_HPP

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
MAVE_INLINE matrix<double, 3, 1> fmadd(const double a,
    const matrix<double, 3, 1>& b, const matrix<double, 3, 1>& c) noexcept
{
    return _mm256_fmadd_pd(_mm256_set1_pd(a), _mm256_load_pd(b.data()),
                           _mm256_load_pd(c.data()));
}
template<>
MAVE_INLINE
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
fmadd(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    return std::make_tuple(
        fmadd(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fmadd(std::get<1>(a), std::get<1>(b), std::get<1>(c)));
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
    return std::make_tuple(
        fmadd(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fmadd(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
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
    return std::make_tuple(
        fmadd(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fmadd(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
        fmadd(std::get<2>(a), std::get<2>(b), std::get<2>(c)),
        fmadd(std::get<3>(a), std::get<3>(b), std::get<3>(c)));
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
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
fmsub(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    return std::make_tuple(
        fmsub(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fmsub(std::get<1>(a), std::get<1>(b), std::get<1>(c)));
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
    return std::make_tuple(
        fmsub(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fmsub(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
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
    return std::make_tuple(
        fmsub(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fmsub(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
        fmsub(std::get<2>(a), std::get<2>(b), std::get<2>(c)),
        fmsub(std::get<3>(a), std::get<3>(b), std::get<3>(c)));
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
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmadd(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    return std::make_tuple(
        fnmadd(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fnmadd(std::get<1>(a), std::get<1>(b), std::get<1>(c)));
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
    return std::make_tuple(
        fnmadd(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fnmadd(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
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
    return std::make_tuple(
        fnmadd(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fnmadd(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
        fnmadd(std::get<2>(a), std::get<2>(b), std::get<2>(c)),
        fnmadd(std::get<3>(a), std::get<3>(b), std::get<3>(c)));
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
std::tuple<matrix<double, 3, 1>, matrix<double, 3, 1>>
fnmsub(std::tuple<double, double> a,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> b,
      std::tuple<const matrix<double, 3, 1>&, const matrix<double, 3, 1>&> c) noexcept
{
    return std::make_tuple(
        fnmsub(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fnmsub(std::get<1>(a), std::get<1>(b), std::get<1>(c)));
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
    return std::make_tuple(
        fnmsub(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fnmsub(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
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
    return std::make_tuple(
        fnmsub(std::get<0>(a), std::get<0>(b), std::get<0>(c)),
        fnmsub(std::get<1>(a), std::get<1>(b), std::get<1>(c)),
        fnmsub(std::get<2>(a), std::get<2>(b), std::get<2>(c)),
        fnmsub(std::get<3>(a), std::get<3>(b), std::get<3>(c)));
}

} // mave
#endif // MAVE_FMA_FMA_HPP
