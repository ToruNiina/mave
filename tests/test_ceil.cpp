#define BOOST_TEST_MODULE test_ceil
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ceil_1arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v1 = vectors1.at(i);

        const auto v2 = mave::ceil(v1);
        BOOST_TEST(v2[0] == std::ceil(v1[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v2[1] == std::ceil(v1[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v2[2] == std::ceil(v1[2]), mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ceil_2arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=2)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);

        const auto vs = mave::ceil(v1, v2);
        BOOST_TEST(std::get<0>(vs)[0] == std::ceil(v1[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<0>(vs)[1] == std::ceil(v1[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<0>(vs)[2] == std::ceil(v1[2]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[0] == std::ceil(v2[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[1] == std::ceil(v2[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[2] == std::ceil(v2[2]), mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ceil_3arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=3)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);
        const auto& v3 = vectors1.at(i+2);

        const auto vs = mave::ceil(v1, v2, v3);
        BOOST_TEST(std::get<0>(vs)[0] == std::ceil(v1[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<0>(vs)[1] == std::ceil(v1[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<0>(vs)[2] == std::ceil(v1[2]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[0] == std::ceil(v2[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[1] == std::ceil(v2[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[2] == std::ceil(v2[2]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<2>(vs)[0] == std::ceil(v3[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<2>(vs)[1] == std::ceil(v3[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<2>(vs)[2] == std::ceil(v3[2]), mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ceil_4arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=4)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);
        const auto& v3 = vectors1.at(i+2);
        const auto& v4 = vectors1.at(i+3);

        const auto vs = mave::ceil(v1, v2, v3, v4);
        BOOST_TEST(std::get<0>(vs)[0] == std::ceil(v1[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<0>(vs)[1] == std::ceil(v1[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<0>(vs)[2] == std::ceil(v1[2]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[0] == std::ceil(v2[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[1] == std::ceil(v2[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<1>(vs)[2] == std::ceil(v2[2]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<2>(vs)[0] == std::ceil(v3[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<2>(vs)[1] == std::ceil(v3[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<2>(vs)[2] == std::ceil(v3[2]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<3>(vs)[0] == std::ceil(v4[0]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<3>(vs)[1] == std::ceil(v4[1]), mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(std::get<3>(vs)[2] == std::ceil(v4[2]), mave::test::tolerance<typename T::value_type>());
    }
}
