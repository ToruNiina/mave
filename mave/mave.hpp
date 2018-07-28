#ifndef MAVE_MAVE_HPP
#define MAVE_MAVE_HPP
#include "macros.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "allocator.hpp"

// Skylake-X
#if defined(__AVX512F__)  && defined(__AVX512CD__) &&\
    defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
#  if defined(MAVE_USE_APPROXIMATION)
#    include "avx512/vector3f_approx.hpp"
#    include "avx512/vector3d_approx.hpp"
#  else
#    include "avx512/vector3f.hpp"
#    include "avx512/vector3d.hpp"
#  endif
#  include "avx512/matrix3x3f.hpp"
#  include "avx512/matrix3x3d.hpp"
//  Xeon Phi Knights Landing...
// #if defined(__AVX512F__)  && defined(__AVX512CD__) &&\
//     defined(__AVX512ER__) && defined(__AVX512PF__)
// not supported.
//
// Haswell, Broadwell, Skylake-S
#elif defined(__AVX2__)
#  if defined(MAVE_USE_APPROXIMATION)
#    include "avx2/vector3f_approx.hpp"
#  else
#    include "avx2/vector3f.hpp"
#  endif
#  include "avx2/vector3d.hpp"
#  include "avx2/matrix3x3d.hpp"
#  include "avx2/matrix3x3f.hpp"
#  include "avx2/vector_matrix_mul.hpp"
// Sandybridge
#elif defined(__AVX__)
#  include "avx/vector3d.hpp"
#  include "avx/vector3f.hpp"
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

#if __FMA__
#include "fma/fma.hpp"
#endif //__FMA__

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
