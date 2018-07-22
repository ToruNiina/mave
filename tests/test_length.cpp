#define BOOST_TEST_MODULE test_length
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

BOOST_AUTO_TEST_CASE_TEMPLATE(length_1arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& v = vectors.at(i);

        const auto ref = length_ref(v);
        const auto val = mave::length(v);
        BOOST_TEST(ref == val, mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(length_2arg, T, test_targets)
{
    std::mt19937 mt(123456789);
    const auto vectors = mave::test::generate_random<T>(N, mt);

    for(std::size_t i=0; i<N; i+=2)
    {
        const auto& v1 = vectors.at(i);
        const auto& v2 = vectors.at(i+1);

        const auto ref1 = length_ref(v1);
        const auto ref2 = length_ref(v2);

        const auto val = mave::length(v1, v2);
        BOOST_TEST(ref1 == std::get<0>(val),
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(ref2 == std::get<1>(val),
                   mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(length_3arg, T, test_targets)
{
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

        const auto val = mave::length(v1, v2, v3);
        BOOST_TEST(ref1 == std::get<0>(val),
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(ref2 == std::get<1>(val),
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(ref3 == std::get<2>(val),
                   mave::test::tolerance<typename T::value_type>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(length_4arg, T, test_targets)
{
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

        const auto val = mave::length(v1, v2, v3, v4);
        BOOST_TEST(ref1 == std::get<0>(val),
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(ref2 == std::get<1>(val),
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(ref3 == std::get<2>(val),
                   mave::test::tolerance<typename T::value_type>());
        BOOST_TEST(ref4 == std::get<3>(val),
                   mave::test::tolerance<typename T::value_type>());
    }
}
