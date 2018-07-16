#ifndef MAVE_TEST_TOLERANCE_HPP
#define MAVE_TEST_TOLERANCE_HPP

#include <boost/test/included/unit_test.hpp>

namespace mave
{
namespace test
{

template<typename T>
T tolerance();

template<>
double tolerance<double>(){return 1e-12;}

template<>
float tolerance<float>(){return 1e-5f;}

} // test
} // mave
#endif
