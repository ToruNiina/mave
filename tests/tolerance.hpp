#ifndef MAVE_TEST_TOLERANCE_HPP
#define MAVE_TEST_TOLERANCE_HPP

#include <boost/test/included/unit_test.hpp>

namespace mave
{
namespace test
{

template<typename T>
T tolerance_val();

#ifdef MAVE_USE_APPROXIMATION
template<> double tolerance_val<double>(){return 1.0  / 16384.0;}
template<> float  tolerance_val<float >(){return 1.5f / 4096.0f;}
#else
template<> double tolerance_val<double>(){return 1e-8;}
template<> float  tolerance_val<float >(){return 1e-4f;}
#endif

template<typename T>
auto tolerance() -> decltype(boost::test_tools::tolerance(std::declval<T>()))
{
    return boost::test_tools::tolerance(tolerance_val<T>());
}

} // test
} // mave
#endif
