#define BOOST_TEST_MODULE test_neg
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/vector.hpp>
#include <mave/allocator.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<
    mave::vector<double, 3>, mave::vector<float, 3>
    > test_targets;

constexpr std::size_t N = 120000;

BOOST_AUTO_TEST_CASE_TEMPLATE(addition, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v1 = vectors1.at(i);
        const auto v2 = -v1;
        BOOST_TEST(v2[0] == -v1[0], mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v2[1] == -v1[1], mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v2[2] == -v1[2], mave::test::tolerance<typename T::value_type>());
    }
}
