#ifndef MAVE_MAVE_HPP
#define MAVE_MAVE_HPP
#include "macros.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "forward.hpp"
#include "allocator.hpp"

#if !defined(MAVE_NO_SIMD)
#  if defined(__AVX512F__)  && defined(__AVX512CD__) &&\
      defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
// Skylake-X
#    if defined(MAVE_USE_APPROXIMATION)
#      include "avx512/vector3f_approx.hpp"
#      include "avx512/vector3d_approx.hpp"
#    else
#      include "avx512/vector3f.hpp"
#      include "avx512/vector3d.hpp"
#    endif
#    include "avx512/matrix3x3f.hpp"
#    include "avx512/matrix3x3d.hpp"
#    if !defined(__FMA__)
#      error "mave avx512 implementation requires FMA instruction set."
#    endif
#    include "avx512/vector3d_fma.hpp"
#    include "avx512/vector3f_fma.hpp"
#  elif defined(__AVX2__)
// Haswell, Broadwell, Skylake-S
#    if defined(MAVE_USE_APPROXIMATION)
#      include "avx2/vector3f_approx.hpp"
#    else
#      include "avx2/vector3f.hpp"
#    endif
#    include "avx2/vector3d.hpp"
#    include "avx2/matrix3x3d.hpp"
#    include "avx2/matrix3x3f.hpp"
#    include "avx2/vector_matrix_mul.hpp"
#    if !defined(__FMA__)
#      error "mave avx2 implementation requires FMA instruction set."
#    endif
#    include "avx2/vector3f_fma.hpp"
#    include "avx2/vector3d_fma.hpp"
#  elif defined(__AVX__)
// Sandybridge
#    if defined(MAVE_USE_APPROXIMATION)
#      include "avx/vector3f_approx.hpp"
#    else
#      include "avx/vector3f.hpp"
#    endif
#    include "avx/vector3d.hpp"
#  endif
#endif

#ifndef   MAVE_VECTOR3_DOUBLE_IMPLEMENTATION
#  define MAVE_VECTOR3_DOUBLE_IMPLEMENTATION    "none"
#endif
#ifndef   MAVE_VECTOR3_FLOAT_IMPLEMENTATION
#  define MAVE_VECTOR3_FLOAT_IMPLEMENTATION     "none"
#endif
#ifndef   MAVE_MATRIX_3X3_DOUBLE_IMPLEMENTATION
#  define MAVE_MATRIX_3X3_DOUBLE_IMPLEMENTATION "none"
#endif
#ifndef   MAVE_MATRIX_3X3_FLOAT_IMPLEMENTATION
#  define MAVE_MATRIX_3X3_FLOAT_IMPLEMENTATION  "none"
#endif //__AVX__

namespace mave
{
inline constexpr const char* supported_instructions()
{
    return   "vector<double,    3>: " MAVE_VECTOR3_DOUBLE_IMPLEMENTATION
           "\nvector<float,     3>: " MAVE_VECTOR3_FLOAT_IMPLEMENTATION
           "\nmatrix<double, 3, 3>: " MAVE_MATRIX_3X3_DOUBLE_IMPLEMENTATION
           "\nmatrix<float,  3, 3>: " MAVE_MATRIX_3X3_FLOAT_IMPLEMENTATION;
}
} // mave
#endif // MAVE_MAVE_HPP
