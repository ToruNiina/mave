#define BOOST_TEST_MODULE test_dot_product
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/mave.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<
    mave::vector<double, 3>,    mave::vector<float, 3>
    > test_targets;

constexpr std::size_t N = 12000;

BOOST_AUTO_TEST_CASE_TEMPLATE(dot_product_1arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v1 = vectors1.at(i);
        const auto& v2 = vectors2.at(i);

        const auto d = dot_product(v1, v2);
        BOOST_TEST(d == v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dot_product_2arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=2)
    {
        const auto& v11 = vectors1.at(i);
        const auto& v12 = vectors1.at(i+1);
        const auto& v21 = vectors2.at(i);
        const auto& v22 = vectors2.at(i+1);

        const auto d = dot_product(std::tie(v11, v12), std::tie(v21, v22));

        const auto d1 = std::get<0>(d);
        const auto d2 = std::get<1>(d);

        BOOST_TEST(d1 == v11[0]*v21[0] + v11[1]*v21[1] + v11[2]*v21[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(d2 == v12[0]*v22[0] + v12[1]*v22[1] + v12[2]*v22[2],
                   mave::test::tolerance<typename T::value_type>());

        BOOST_TEST(v11.diagnosis());
        BOOST_TEST(v12.diagnosis());
        BOOST_TEST(v21.diagnosis());
        BOOST_TEST(v22.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dot_product_3arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=3)
    {
        const auto& v11 = vectors1.at(i);
        const auto& v12 = vectors1.at(i+1);
        const auto& v13 = vectors1.at(i+2);
        const auto& v21 = vectors2.at(i);
        const auto& v22 = vectors2.at(i+1);
        const auto& v23 = vectors2.at(i+2);

        const auto d = dot_product(std::tie(v11, v12, v13), std::tie(v21, v22, v23));

        const auto d1 = std::get<0>(d);
        const auto d2 = std::get<1>(d);
        const auto d3 = std::get<2>(d);

        BOOST_TEST(d1 == v11[0]*v21[0] + v11[1]*v21[1] + v11[2]*v21[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(d2 == v12[0]*v22[0] + v12[1]*v22[1] + v12[2]*v22[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(d3 == v13[0]*v23[0] + v13[1]*v23[1] + v13[2]*v23[2],
                   mave::test::tolerance<typename T::value_type>());

        BOOST_TEST(v11.diagnosis());
        BOOST_TEST(v12.diagnosis());
        BOOST_TEST(v13.diagnosis());
        BOOST_TEST(v21.diagnosis());
        BOOST_TEST(v22.diagnosis());
        BOOST_TEST(v23.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dot_product_4arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors1 = mave::test::generate_random<T>(N, mt);
    const auto vectors2 = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=4)
    {
        const auto& v11 = vectors1.at(i);
        const auto& v12 = vectors1.at(i+1);
        const auto& v13 = vectors1.at(i+2);
        const auto& v14 = vectors1.at(i+3);

        const auto& v21 = vectors2.at(i);
        const auto& v22 = vectors2.at(i+1);
        const auto& v23 = vectors2.at(i+2);
        const auto& v24 = vectors2.at(i+3);

        const auto d = dot_product(std::tie(v11, v12, v13, v14),
                                   std::tie(v21, v22, v23, v24));

        const auto d1 = std::get<0>(d);
        const auto d2 = std::get<1>(d);
        const auto d3 = std::get<2>(d);
        const auto d4 = std::get<3>(d);

        BOOST_TEST(d1 == v11[0]*v21[0] + v11[1]*v21[1] + v11[2]*v21[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(d2 == v12[0]*v22[0] + v12[1]*v22[1] + v12[2]*v22[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(d3 == v13[0]*v23[0] + v13[1]*v23[1] + v13[2]*v23[2],
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(d4 == v14[0]*v24[0] + v14[1]*v24[1] + v14[2]*v24[2],
                   mave::test::tolerance<typename T::value_type>());

        BOOST_TEST(v11.diagnosis());
        BOOST_TEST(v12.diagnosis());
        BOOST_TEST(v13.diagnosis());
        BOOST_TEST(v14.diagnosis());

        BOOST_TEST(v21.diagnosis());
        BOOST_TEST(v22.diagnosis());
        BOOST_TEST(v23.diagnosis());
        BOOST_TEST(v24.diagnosis());
    }
}
