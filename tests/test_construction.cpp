#define BOOST_TEST_MODULE test_construction
#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <mave/mave.hpp>
#include <random>

constexpr std::size_t N = 12000;

typedef boost::mpl::list<float, double> floating_points;

BOOST_AUTO_TEST_CASE(test_construct_defualt)
{
    mave::vector<float,  3> v3f;
    mave::vector<double, 3> v3d;
    mave::vector<float,  4> v4f;
    mave::vector<double, 4> v4d;

    mave::matrix<float,  3, 3> m3x3d;
    mave::matrix<double, 3, 3> m3x3f;
    mave::matrix<float,  4, 4> m4x4d;
    mave::matrix<double, 4, 4> m4x4f;

    BOOST_TEST(v3f.diagnosis());
    BOOST_TEST(v3d.diagnosis());
    BOOST_TEST(v4f.diagnosis());
    BOOST_TEST(v4d.diagnosis());

    BOOST_TEST(m3x3d.diagnosis());
    BOOST_TEST(m3x3f.diagnosis());
    BOOST_TEST(m4x4d.diagnosis());
    BOOST_TEST(m4x4f.diagnosis());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_construct_vector3, T, floating_points)
{
    std::mt19937 mt(123456789);
    std::uniform_real_distribution<T> uni(-1000, 1000);

    for(std::size_t i=0; i<N; ++i)
    {
        const T x1 = uni(mt);
        const T x2 = uni(mt);
        const T x3 = uni(mt);

        const mave::vector<T, 3> v(x1, x2, x3);

        BOOST_TEST(v[0] == x1);
        BOOST_TEST(v[1] == x2);
        BOOST_TEST(v[2] == x3);

        BOOST_TEST(v(0,0) == x1);
        BOOST_TEST(v(1,0) == x2);
        BOOST_TEST(v(2,0) == x3);

        BOOST_TEST(v.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_construct_vector3_from_array, T, floating_points)
{
    std::mt19937 mt(123456789);
    std::uniform_real_distribution<T> uni(-1000, 1000);

    for(std::size_t i=0; i<N; ++i)
    {
        const std::array<T, 3> a = [&mt, &uni]() {
            std::array<T, 3> a_;
            a_[0] = uni(mt);
            a_[1] = uni(mt);
            a_[2] = uni(mt);
            return a_;
        }();

        const mave::vector<T, 3> v1(a);
        const mave::vector<T, 3> v2 = a;

        BOOST_TEST(v1[0] == a[0]);
        BOOST_TEST(v1[1] == a[1]);
        BOOST_TEST(v1[2] == a[2]);
        BOOST_TEST(v1(0,0) == a[0]);
        BOOST_TEST(v1(1,0) == a[1]);
        BOOST_TEST(v1(2,0) == a[2]);

        BOOST_TEST(v2[0] == a[0]);
        BOOST_TEST(v2[1] == a[1]);
        BOOST_TEST(v2[2] == a[2]);
        BOOST_TEST(v2(0,0) == a[0]);
        BOOST_TEST(v2(1,0) == a[1]);
        BOOST_TEST(v2(2,0) == a[2]);

        BOOST_TEST(v1.diagnosis());
        BOOST_TEST(v2.diagnosis());
    }
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_construct_matrix3x3, T, floating_points)
{
    std::mt19937 mt(123456789);
    std::uniform_real_distribution<T> uni(-1000, 1000);

    for(std::size_t i=0; i<N; ++i)
    {
        const T x11 = uni(mt);
        const T x12 = uni(mt);
        const T x13 = uni(mt);

        const T x21 = uni(mt);
        const T x22 = uni(mt);
        const T x23 = uni(mt);

        const T x31 = uni(mt);
        const T x32 = uni(mt);
        const T x33 = uni(mt);

        const mave::matrix<T, 3, 3> m(x11, x12, x13,
                                      x21, x22, x23,
                                      x31, x32, x33);

        BOOST_TEST(m[0] == x11);
        BOOST_TEST(m[1] == x12);
        BOOST_TEST(m[2] == x13);

        BOOST_TEST(m[3] == x21);
        BOOST_TEST(m[4] == x22);
        BOOST_TEST(m[5] == x23);

        BOOST_TEST(m[6] == x31);
        BOOST_TEST(m[7] == x32);
        BOOST_TEST(m[8] == x33);

        BOOST_TEST(m(0,0) == x11);
        BOOST_TEST(m(0,1) == x12);
        BOOST_TEST(m(0,2) == x13);

        BOOST_TEST(m(1,0) == x21);
        BOOST_TEST(m(1,1) == x22);
        BOOST_TEST(m(1,2) == x23);

        BOOST_TEST(m(2,0) == x31);
        BOOST_TEST(m(2,1) == x32);
        BOOST_TEST(m(2,2) == x33);

        BOOST_TEST(m.diagnosis());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_construct_matrix3x3_from_array, T, floating_points)
{
    std::mt19937 mt(123456789);
    std::uniform_real_distribution<T> uni(-1000, 1000);

    for(std::size_t i=0; i<N; ++i)
    {
        const std::array<T, 9> a = [&mt, &uni]() {
            std::array<T, 9> a_;
            for(std::size_t j=0; j<9; ++j)
            {
                a_[j] = uni(mt);
            }
            return a_;
        }();

        const mave::matrix<T, 3, 3> m1(a);
        const mave::matrix<T, 3, 3> m2 = a;

        for(std::size_t j=0; j<9; ++j)
        {
            BOOST_TEST(m1[j] == a[j]);
            BOOST_TEST(m2[j] == a[j]);
        }

        BOOST_TEST(m1(0,0) == a[0]);
        BOOST_TEST(m1(0,1) == a[1]);
        BOOST_TEST(m1(0,2) == a[2]);

        BOOST_TEST(m1(1,0) == a[3]);
        BOOST_TEST(m1(1,1) == a[4]);
        BOOST_TEST(m1(1,2) == a[5]);

        BOOST_TEST(m1(2,0) == a[6]);
        BOOST_TEST(m1(2,1) == a[7]);
        BOOST_TEST(m1(2,2) == a[8]);

        BOOST_TEST(m2(0,0) == a[0]);
        BOOST_TEST(m2(0,1) == a[1]);
        BOOST_TEST(m2(0,2) == a[2]);

        BOOST_TEST(m2(1,0) == a[3]);
        BOOST_TEST(m2(1,1) == a[4]);
        BOOST_TEST(m2(1,2) == a[5]);

        BOOST_TEST(m2(2,0) == a[6]);
        BOOST_TEST(m2(2,1) == a[7]);
        BOOST_TEST(m2(2,2) == a[8]);

        BOOST_TEST(m1.diagnosis());
        BOOST_TEST(m2.diagnosis());
    }
}
