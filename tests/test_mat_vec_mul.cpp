#define BOOST_TEST_MODULE test_mat_vec_mul
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/mave.hpp>
#include <tests/generate_random_matrices.hpp>
#include <tests/tolerance.hpp>

typedef boost::mpl::list<double, float> test_targets;

constexpr std::size_t N = 12000;

BOOST_AUTO_TEST_CASE_TEMPLATE(mat3x3_vec3, T, test_targets)
{
    std::mt19937 mt(123456789);

    const auto matrices = mave::test::generate_random_positive<mave::matrix<T, 3, 3>>(N, mt);
    const auto vectors  = mave::test::generate_random_positive<mave::vector<T, 3>>(N, mt);

    for(std::size_t i=0; i<N; ++i)
    {
        const auto& m = matrices.at(i);
        const auto& v = vectors.at(i);

        const auto v2 = m * v;

        BOOST_TEST(v2[0] == m(0,0)*v[0] + m(0,1)*v[1] + m(0,2)*v[2], mave::test::tolerance<T>());
        BOOST_TEST(v2[1] == m(1,0)*v[0] + m(1,1)*v[1] + m(1,2)*v[2], mave::test::tolerance<T>());
        BOOST_TEST(v2[2] == m(2,0)*v[0] + m(2,1)*v[1] + m(2,2)*v[2], mave::test::tolerance<T>());

        BOOST_TEST(v.diagnosis());
        BOOST_TEST(v2.diagnosis());
    }
}
