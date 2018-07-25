#define BOOST_TEST_MODULE test_mul
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/mave.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<
    mave::vector<double, 3>,    mave::vector<float, 3>,
    mave::matrix<double, 3, 3>, mave::matrix<float, 3, 3>
    > test_targets;

constexpr std::size_t N = 12000;

BOOST_AUTO_TEST_CASE_TEMPLATE(multiply, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& s  = scalars.at(i);
        const auto& v1 = vectors.at(i);

        const auto v2 = s * v1;
        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(v2[j] == s * v1[j],
                       mave::test::tolerance<typename T::value_type>());
        }

        const auto v3 = v1 * s;
        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(v3[j] == v1[j] * s,
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
        BOOST_TEST(v3.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multiply_assign, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& s  = scalars.at(i);
        const auto& v1 = vectors.at(i);

        auto v2 = v1;
        v2 *= s;
        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(v2[j] == v1[j] * s,
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
    }
}
