#define BOOST_TEST_MODULE test_fma
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/vector.hpp>
#include <mave/allocator.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<
    mave::vector<double, 3>, mave::vector<float, 3>
    > test_targets;

constexpr std::size_t N = 12000;

BOOST_AUTO_TEST_CASE_TEMPLATE(fmadd, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars  = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto  s  = scalars.at(i);
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors2.at(i);
        const auto  v3 = mave::fmadd(s, v1, v2);

        BOOST_TEST(v3[0] == std::fma(s, v1[0], v2[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[1] == std::fma(s, v1[1], v2[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[2] == std::fma(s, v1[2], v2[2]), mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fmsub, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars  = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto  s  = scalars.at(i);
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors2.at(i);
        const auto  v3 = mave::fmsub(s, v1, v2);

        BOOST_TEST(v3[0] == std::fma(s, v1[0], -v2[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[1] == std::fma(s, v1[1], -v2[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[2] == std::fma(s, v1[2], -v2[2]), mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fnmadd, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars  = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto  s  = scalars.at(i);
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors2.at(i);
        const auto  v3 = mave::fnmadd(s, v1, v2);

        BOOST_TEST(v3[0] == std::fma(-s, v1[0], v2[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[1] == std::fma(-s, v1[1], v2[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[2] == std::fma(-s, v1[2], v2[2]), mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fnmsub, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars  = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto  s  = scalars.at(i);
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors2.at(i);
        const auto  v3 = mave::fnmsub(s, v1, v2);

        BOOST_TEST(v3[0] == std::fma(-s, v1[0], -v2[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[1] == std::fma(-s, v1[1], -v2[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v3[2] == std::fma(-s, v1[2], -v2[2]), mave::test::tolerance<typename T::value_type>());
    }
}
