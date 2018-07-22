#define BOOST_TEST_MODULE test_floor
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/vector.hpp>
#include <mave/allocator.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<
    mave::vector<double, 3>,    mave::vector<float, 3>,
    mave::matrix<double, 3, 3>, mave::matrix<float, 3, 3>
    > test_targets;

constexpr std::size_t N = 12000;

BOOST_AUTO_TEST_CASE_TEMPLATE(floor_1arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v1 = vectors1.at(i);

        const auto v2 = mave::floor(v1);
        for(std::size_t j=0; j<v2.size(); ++j)
        {
            BOOST_TEST(v2[j] == std::floor(v1[j]),
                       mave::test::tolerance<typename T::value_type>());
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(floor_2arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=2)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);

        const auto vs = mave::floor(v1, v2);
        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(std::get<0>(vs)[j] == std::floor(v1[j]),
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(std::get<1>(vs)[j] == std::floor(v2[j]),
                       mave::test::tolerance<typename T::value_type>());
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(floor_3arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=3)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);
        const auto& v3 = vectors1.at(i+2);

        const auto vs = mave::floor(v1, v2, v3);
        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(std::get<0>(vs)[j] == std::floor(v1[j]),
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(std::get<1>(vs)[j] == std::floor(v2[j]),
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(std::get<2>(vs)[j] == std::floor(v3[j]),
                       mave::test::tolerance<typename T::value_type>());
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(floor_4arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=4)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);
        const auto& v3 = vectors1.at(i+2);
        const auto& v4 = vectors1.at(i+3);

        const auto vs = mave::floor(v1, v2, v3, v4);
        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(std::get<0>(vs)[j] == std::floor(v1[j]),
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(std::get<1>(vs)[j] == std::floor(v2[j]),
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(std::get<2>(vs)[j] == std::floor(v3[j]),
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(std::get<3>(vs)[j] == std::floor(v4[j]),
                       mave::test::tolerance<typename T::value_type>());
        }
    }
}
