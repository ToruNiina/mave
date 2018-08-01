#define BOOST_TEST_MODULE test_cross_product
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/mave.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<
    mave::vector<double, 3>,    mave::vector<float, 3>
    > test_targets;

constexpr std::size_t N = 12000;

BOOST_AUTO_TEST_CASE_TEMPLATE(cross_product_1arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors2.at(i);

        const auto c = mave::cross_product(v1, v2);
        BOOST_TEST(c[0] == v1[1]*v2[2] - v1[2]*v2[1],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(c[1] == v1[2]*v2[0] - v1[0]*v2[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(c[2] == v1[0]*v2[1] - v1[1]*v2[0],
                   mave::test::tolerance<typename T::value_type>());

        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
        BOOST_TEST(c.diagnosis());
    }
}
