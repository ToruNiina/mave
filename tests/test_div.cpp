#define BOOST_TEST_MODULE test_div
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

BOOST_AUTO_TEST_CASE_TEMPLATE(division, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);
    const auto scalars = mave::test::generate_random<typename T::value_type>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v = vectors.at(i);
        const auto& s = scalars.at(i);

        const auto v_ = v / s;
        for(std::size_t j=0; j<v.size(); ++j)
        {
            BOOST_TEST(v_[j] == v[j] / s,
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v.diagnosis());
        BOOST_TEST(v_.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(division_2arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=2)
    {
        const auto& s1 = scalars.at(i);
        const auto& s2 = scalars.at(i+1);
        const auto& v11 = vectors.at(i);
        const auto& v12 = vectors.at(i+1);

        const auto v2 = std::tie(v11, v12) / std::make_tuple(s1, s2);

        const auto v21 = std::get<0>(v2);
        const auto v22 = std::get<1>(v2);

        for(std::size_t j=0; j<v21.size(); ++j)
        {
            BOOST_TEST(v21[j] == v11[j] / s1,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v22[j] == v12[j] / s2,
                       mave::test::tolerance<typename T::value_type>());
        }

        BOOST_TEST(v11.diagnosis());
        BOOST_TEST(v12.diagnosis());
        BOOST_TEST(v21.diagnosis());
        BOOST_TEST(v22.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(division_3arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=3)
    {
        const auto& s1  = scalars.at(i);
        const auto& s2  = scalars.at(i+1);
        const auto& s3  = scalars.at(i+2);
        const auto& v11 = vectors.at(i);
        const auto& v12 = vectors.at(i+1);
        const auto& v13 = vectors.at(i+2);

        const auto v2 = std::tie(v11, v12, v13) / std::make_tuple(s1, s2, s3);

        const auto v21 = std::get<0>(v2);
        const auto v22 = std::get<1>(v2);
        const auto v23 = std::get<2>(v2);

        for(std::size_t j=0; j<v21.size(); ++j)
        {
            BOOST_TEST(v21[j] == v11[j] / s1,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v22[j] == v12[j] / s2,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v23[j] == v13[j] / s3,
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v11.diagnosis());
        BOOST_TEST(v12.diagnosis());
        BOOST_TEST(v13.diagnosis());

        BOOST_TEST(v21.diagnosis());
        BOOST_TEST(v22.diagnosis());
        BOOST_TEST(v23.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(division_4arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto scalars = mave::test::generate_random<typename T::value_type>(N, mt);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=4)
    {
        const auto& s1  = scalars.at(i);
        const auto& s2  = scalars.at(i+1);
        const auto& s3  = scalars.at(i+2);
        const auto& s4  = scalars.at(i+3);
        const auto& v11 = vectors.at(i);
        const auto& v12 = vectors.at(i+1);
        const auto& v13 = vectors.at(i+2);
        const auto& v14 = vectors.at(i+3);

        const auto v2 = std::tie(v11, v12, v13, v14) / std::make_tuple(s1, s2, s3, s4);

        const auto v21 = std::get<0>(v2);
        const auto v22 = std::get<1>(v2);
        const auto v23 = std::get<2>(v2);
        const auto v24 = std::get<3>(v2);

        for(std::size_t j=0; j<v21.size(); ++j)
        {
            BOOST_TEST(v21[j] == v11[j] / s1,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v22[j] == v12[j] / s2,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v23[j] == v13[j] / s3,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v24[j] == v14[j] / s4,
                       mave::test::tolerance<typename T::value_type>());
        }

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

BOOST_AUTO_TEST_CASE_TEMPLATE(div_assignment, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);
    const auto scalars = mave::test::generate_random<typename T::value_type>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v = vectors.at(i);
        const auto& s = scalars.at(i);

        auto v_(v);
        v_ /= s;
        for(std::size_t j=0; j<v.size(); ++j)
        {
            BOOST_TEST(v_[j] == v[j] / s,
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v.diagnosis());
        BOOST_TEST(v_.diagnosis());
    }
}
