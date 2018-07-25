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
