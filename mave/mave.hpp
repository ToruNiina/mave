#ifndef MAVE_MAVE_HPP
#define MAVE_MAVE_HPP
#include "matrix.hpp"
#include "vector.hpp"
#include "allocator.hpp"

#if defined(__AVX2__)
#  include "avx2/vector3d.hpp"
#  include "avx2/vector3f.hpp"
#  include "avx2/matrix3x3d.hpp"
#  include "avx2/matrix3x3f.hpp"
#elif defined(__AVX__)
#  include "avx/vector3d.hpp"
#  include "avx/vector3f.hpp"
#else
#  define MAVE_VECTOR3_DOUBLE_IMPLEMENTATION    "none"
#  define MAVE_VECTOR3_FLOAT_IMPLEMENTATION     "none"
#  define MAVE_MATRIX_3X3_DOUBLE_IMPLEMENTATION "none"
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
