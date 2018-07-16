#ifndef MAVE_TEST_TOLERANCE_HPP
#define MAVE_TEST_TOLERANCE_HPP

#include <boost/test/included/unit_test.hpp>

namespace mave
{
namespace test
{

template<typename T>
T tolerance_val();

template<>
double tolerance_val<double>(){return 1e-12;}
template<>
float tolerance_val<float>(){return 1e-5f;}

template<typename T>
auto tolerance() -> decltype(boost::test_tools::tolerance(std::declval<T>()))
{
    return boost::test_tools::tolerance(tolerance_val<T>());
}

} // test
} // mave
#endif
