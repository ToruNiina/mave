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

BOOST_AUTO_TEST_CASE_TEMPLATE(multiplication_1arg, T, test_targets)
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

BOOST_AUTO_TEST_CASE_TEMPLATE(multiplication_2arg, T, test_targets)
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

        const auto v2 = std::make_tuple(s1, s2) * std::tie(v11, v12);

        const auto v21 = std::get<0>(v2);
        const auto v22 = std::get<1>(v2);

        for(std::size_t j=0; j<v21.size(); ++j)
        {
            BOOST_TEST(v21[j] == s1 * v11[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v22[j] == s2 * v12[j],
                       mave::test::tolerance<typename T::value_type>());
        }

        const auto v3 = std::tie(v11, v12) * std::make_tuple(s1, s2);

        const auto v31 = std::get<0>(v3);
        const auto v32 = std::get<1>(v3);

        for(std::size_t j=0; j<v31.size(); ++j)
        {
            BOOST_TEST(v31[j] == v11[j] * s1,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v32[j] == v12[j] * s2,
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v11.diagnosis());
        BOOST_TEST(v12.diagnosis());
        BOOST_TEST(v21.diagnosis());
        BOOST_TEST(v22.diagnosis());
        BOOST_TEST(v31.diagnosis());
        BOOST_TEST(v32.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multiplication_3arg, T, test_targets)
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

        const auto v2 = std::make_tuple(s1, s2, s3) * std::tie(v11, v12, v13);

        const auto v21 = std::get<0>(v2);
        const auto v22 = std::get<1>(v2);
        const auto v23 = std::get<2>(v2);

        for(std::size_t j=0; j<v21.size(); ++j)
        {
            BOOST_TEST(v21[j] == s1 * v11[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v22[j] == s2 * v12[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v23[j] == s3 * v13[j],
                       mave::test::tolerance<typename T::value_type>());
        }

        const auto v3 = std::tie(v11, v12, v13) * std::make_tuple(s1, s2, s3);

        const auto v31 = std::get<0>(v3);
        const auto v32 = std::get<1>(v3);
        const auto v33 = std::get<2>(v3);

        for(std::size_t j=0; j<v31.size(); ++j)
        {
            BOOST_TEST(v31[j] == v11[j] * s1,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v32[j] == v12[j] * s2,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v33[j] == v13[j] * s3,
                       mave::test::tolerance<typename T::value_type>());
        }
        BOOST_TEST(v11.diagnosis());
        BOOST_TEST(v12.diagnosis());
        BOOST_TEST(v13.diagnosis());

        BOOST_TEST(v21.diagnosis());
        BOOST_TEST(v22.diagnosis());
        BOOST_TEST(v23.diagnosis());

        BOOST_TEST(v31.diagnosis());
        BOOST_TEST(v32.diagnosis());
        BOOST_TEST(v33.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multiplication_4arg, T, test_targets)
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

        const auto v2 = std::make_tuple(s1, s2, s3, s4) * std::tie(v11, v12, v13, v14);

        const auto v21 = std::get<0>(v2);
        const auto v22 = std::get<1>(v2);
        const auto v23 = std::get<2>(v2);
        const auto v24 = std::get<3>(v2);

        for(std::size_t j=0; j<v21.size(); ++j)
        {
            BOOST_TEST(v21[j] == s1 * v11[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v22[j] == s2 * v12[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v23[j] == s3 * v13[j],
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v24[j] == s4 * v14[j],
                       mave::test::tolerance<typename T::value_type>());
        }

        const auto v3 = std::tie(v11, v12, v13, v14) * std::make_tuple(s1, s2, s3, s4);

        const auto v31 = std::get<0>(v3);
        const auto v32 = std::get<1>(v3);
        const auto v33 = std::get<2>(v3);
        const auto v34 = std::get<3>(v3);

        for(std::size_t j=0; j<v31.size(); ++j)
        {
            BOOST_TEST(v31[j] == v11[j] * s1,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v32[j] == v12[j] * s2,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v33[j] == v13[j] * s3,
                       mave::test::tolerance<typename T::value_type>());
            BOOST_TEST(v34[j] == v14[j] * s4,
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

        BOOST_TEST(v31.diagnosis());
        BOOST_TEST(v32.diagnosis());
        BOOST_TEST(v33.diagnosis());
        BOOST_TEST(v34.diagnosis());
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
