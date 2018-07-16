#ifndef MAVE_TEST_GENERATE_RANDOM_MATRICES_HPP
#define MAVE_TEST_GENERATE_RANDOM_MATRICES_HPP
#include <mave/matrix.hpp>
#include <mave/vector.hpp>
#include <random>
#include <vector>

namespace mave
{
namespace test
{

template<typename T, std::size_t R, std::size_t C, typename RNG>
mave::matrix<T, R, C> generate_random_matrix(RNG&& rng)
{
    std::uniform_real_distribution<T> uni(T(-1000.0), T(1000.0));

    mave::matrix<T, R, C> m;
    for(std::size_t i=0; i<R*C; ++i) {m[i] = uni(rng);}
    return m;
}

template<typename T, std::size_t R, std::size_t C, typename RNG>
std::vector<mave::matrix<T, R, C>,
            mave::aligned_allocator<mave::matrix<T, R, C>>>
generate_random_matrices(std::size_t N, RNG&& rng)
{
    std::uniform_real_distribution<T> uni(T(-1000.0), T(1000.0));

    std::vector<mave::matrix<T, R, C>,
                mave::aligned_allocator<mave::matrix<T, R, C>>> vec(N);
    for(std::size_t i=0; i<N; ++i)
    {
        auto& v = vec[i];
        for(std::size_t j=0; j<R*C; ++j)
        {
            v[j] = uni(rng);
        }
    }
    return vec;
}

template<typename T, typename RNG>
std::vector<T, mave::aligned_allocator<T>>
generate_random(std::size_t N, RNG&& rng)
{
    return generate_random_matrices<
        typename T::value_type, T::row_size, T::column_size>(N, rng);
}

} // test
} // mave
#endif // MAVE_TEST_GENERATE_RANDOM_MATRICES_HPP
