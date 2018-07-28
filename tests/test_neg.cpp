#define BOOST_TEST_MODULE test_neg
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

BOOST_AUTO_TEST_CASE_TEMPLATE(negation_1arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v1 = vectors1.at(i);
        const auto v2 = -v1;
        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(v2[j] == -v1[j],
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(negation_2arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=2)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);

        const auto neg = -std::tie(v1, v2);
        const auto& n1 = std::get<0>(neg);
        const auto& n2 = std::get<1>(neg);

        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(n1[j] == -v1[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(n2[j] == -v2[j],
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
        BOOST_TEST(n1.diagnosis());
        BOOST_TEST(n2.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(negation_3arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=3)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);
        const auto& v3 = vectors1.at(i+2);

        const auto neg = -std::tie(v1, v2, v3);
        const auto& n1 = std::get<0>(neg);
        const auto& n2 = std::get<1>(neg);
        const auto& n3 = std::get<2>(neg);

        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(n1[j] == -v1[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(n2[j] == -v2[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(n3[j] == -v3[j],
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
        BOOST_TEST(v3.diagnosis());
        BOOST_TEST(n1.diagnosis());
        BOOST_TEST(n2.diagnosis());
        BOOST_TEST(n3.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(negation_4arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=4)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors1.at(i+1);
        const auto& v3 = vectors1.at(i+2);
        const auto& v4 = vectors1.at(i+3);

        const auto neg = -std::tie(v1, v2, v3, v4);
        const auto& n1 = std::get<0>(neg);
        const auto& n2 = std::get<1>(neg);
        const auto& n3 = std::get<2>(neg);
        const auto& n4 = std::get<3>(neg);

        for(std::size_t j=0; j<v1.size(); ++j)
        {
            BOOST_TEST(n1[j] == -v1[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(n2[j] == -v2[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(n3[j] == -v3[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(n4[j] == -v4[j],
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
        BOOST_TEST(v3.diagnosis());
        BOOST_TEST(v4.diagnosis());
        BOOST_TEST(n1.diagnosis());
        BOOST_TEST(n2.diagnosis());
        BOOST_TEST(n3.diagnosis());
        BOOST_TEST(n4.diagnosis());
    }
}
