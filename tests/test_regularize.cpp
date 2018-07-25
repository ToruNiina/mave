#define BOOST_TEST_MODULE test_regularize
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/mave.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<
    mave::vector<double, 3>, mave::vector<float, 3>
    > test_targets;

constexpr std::size_t N = 12000;

template<typename T, std::size_t N>
T length_ref(const mave::vector<T, N>& v)
{
    T retval(0);
    for(std::size_t i=0; i<v.size(); ++i)
    {
        retval += v[i] * v[i];
    }
    return std::sqrt(retval);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(regularize_1arg, T, test_targets)
{
    using real = typename T::value_type;
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v = vectors.at(i);

        const auto ref = length_ref(v);
        const auto val = mave::regularize(v);

        BOOST_TEST(ref == std::get<1>(val), mave::test::tolerance<real>());
        BOOST_TEST(length_ref(std::get<0>(val)) == real(1.0),
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val)[0] * std::get<1>(val) == v[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val)[1] * std::get<1>(val) == v[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val)[2] * std::get<1>(val) == v[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(v.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(regularize_2arg, T, test_targets)
{
    using real = typename T::value_type;
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=2)
    {
        const auto& v1 = vectors.at(i);
        const auto& v2 = vectors.at(i+1);

        const auto ref1 = length_ref(v1);
        const auto ref2 = length_ref(v2);

        const auto val = mave::regularize(v1, v2);
        const auto& val1 = std::get<0>(val);
        const auto& val2 = std::get<1>(val);

        BOOST_TEST(ref1 == std::get<1>(val1), mave::test::tolerance<real>());
        BOOST_TEST(ref2 == std::get<1>(val2), mave::test::tolerance<real>());

        BOOST_TEST(length_ref(std::get<0>(val1)) == real(1.0),
                   mave::test::tolerance<real>());
        BOOST_TEST(length_ref(std::get<0>(val2)) == real(1.0),
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val1)[0] * std::get<1>(val1) == v1[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val1)[1] * std::get<1>(val1) == v1[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val1)[2] * std::get<1>(val1) == v1[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val2)[0] * std::get<1>(val2) == v2[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val2)[1] * std::get<1>(val2) == v2[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val2)[2] * std::get<1>(val2) == v2[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(regularize_3arg, T, test_targets)
{
    using real = typename T::value_type;
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=3)
    {
        const auto& v1 = vectors.at(i);
        const auto& v2 = vectors.at(i+1);
        const auto& v3 = vectors.at(i+2);

        const auto ref1 = length_ref(v1);
        const auto ref2 = length_ref(v2);
        const auto ref3 = length_ref(v3);

        const auto val = mave::regularize(v1, v2, v3);
        const auto& val1 = std::get<0>(val);
        const auto& val2 = std::get<1>(val);
        const auto& val3 = std::get<2>(val);

        BOOST_TEST(ref1 == std::get<1>(val1), mave::test::tolerance<real>());
        BOOST_TEST(ref2 == std::get<1>(val2), mave::test::tolerance<real>());
        BOOST_TEST(ref3 == std::get<1>(val3), mave::test::tolerance<real>());

        BOOST_TEST(length_ref(std::get<0>(val1)) == real(1.0),
                   mave::test::tolerance<real>());
        BOOST_TEST(length_ref(std::get<0>(val2)) == real(1.0),
                   mave::test::tolerance<real>());
        BOOST_TEST(length_ref(std::get<0>(val3)) == real(1.0),
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val1)[0] * std::get<1>(val1) == v1[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val1)[1] * std::get<1>(val1) == v1[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val1)[2] * std::get<1>(val1) == v1[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val2)[0] * std::get<1>(val2) == v2[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val2)[1] * std::get<1>(val2) == v2[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val2)[2] * std::get<1>(val2) == v2[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val3)[0] * std::get<1>(val3) == v3[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val3)[1] * std::get<1>(val3) == v3[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val3)[2] * std::get<1>(val3) == v3[2],
                   mave::test::tolerance<real>());
        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
        BOOST_TEST(v3.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(regularize_4arg, T, test_targets)
{
    using real = typename T::value_type;
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=4)
    {
        const auto& v1 = vectors.at(i);
        const auto& v2 = vectors.at(i+1);
        const auto& v3 = vectors.at(i+2);
        const auto& v4 = vectors.at(i+3);

        const auto ref1 = length_ref(v1);
        const auto ref2 = length_ref(v2);
        const auto ref3 = length_ref(v3);
        const auto ref4 = length_ref(v4);

        const auto val = mave::regularize(v1, v2, v3, v4);
        const auto& val1 = std::get<0>(val);
        const auto& val2 = std::get<1>(val);
        const auto& val3 = std::get<2>(val);
        const auto& val4 = std::get<3>(val);

        BOOST_TEST(ref1 == std::get<1>(val1), mave::test::tolerance<real>());
        BOOST_TEST(ref2 == std::get<1>(val2), mave::test::tolerance<real>());
        BOOST_TEST(ref3 == std::get<1>(val3), mave::test::tolerance<real>());
        BOOST_TEST(ref4 == std::get<1>(val4), mave::test::tolerance<real>());

        BOOST_TEST(length_ref(std::get<0>(val1)) == real(1.0),
                   mave::test::tolerance<real>());
        BOOST_TEST(length_ref(std::get<0>(val2)) == real(1.0),
                   mave::test::tolerance<real>());
        BOOST_TEST(length_ref(std::get<0>(val3)) == real(1.0),
                   mave::test::tolerance<real>());
        BOOST_TEST(length_ref(std::get<0>(val4)) == real(1.0),
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val1)[0] * std::get<1>(val1) == v1[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val1)[1] * std::get<1>(val1) == v1[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val1)[2] * std::get<1>(val1) == v1[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val2)[0] * std::get<1>(val2) == v2[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val2)[1] * std::get<1>(val2) == v2[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val2)[2] * std::get<1>(val2) == v2[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val3)[0] * std::get<1>(val3) == v3[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val3)[1] * std::get<1>(val3) == v3[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val3)[2] * std::get<1>(val3) == v3[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(std::get<0>(val4)[0] * std::get<1>(val4) == v4[0],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val4)[1] * std::get<1>(val4) == v4[1],
                   mave::test::tolerance<real>());
        BOOST_TEST(std::get<0>(val4)[2] * std::get<1>(val4) == v4[2],
                   mave::test::tolerance<real>());

        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
        BOOST_TEST(v3.diagnosis());
        BOOST_TEST(v4.diagnosis());
    }
}
