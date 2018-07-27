mave
====

SIMD-oriented small matrix and vector library

It focuses on 3D vector operations (e.g. physical simulation, graphics, etc).

## usage

after including `mave/mave.hpp`, compile your code with
`-march=native -mtune=native` to turn simd flags (e.g. `__AVX2__`) on.

```cpp
mave::matrix<double, 3, 3> m(1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0);
mave::vector<double, 3>    v(1.0, 2.0, 3.0);

mave::vector<double, 3>    w = m * v;

std::cout << w[0] << ' ' << w[1] << ' ' << w[2] << '\n';
```

it automatically uses SIMD instructions for the minimal width to store
vector/matrix (e.g. for `mave::vector<float, 3>`, use `SSE` with `__m128`).

if `AVX` is available, you can use `__m256` to calculate 2 vector addition at
once.

```cpp
mave::vector<double, 3> v1(1.0, 2.0, 3.0), v2(4.0, 5.0, 6.0);
mave::vector<double, 3> w1(1.0, 2.0, 3.0), w2(4.0, 5.0, 6.0);
mave::vector<double, 3> u1, u2;

std::tie(u1, u2) = std::tie(v1, v2) + std::tie(w1, w2);
```

you need to use `std::make_tuple` instead of `std::tie` to do scalar mul/div.

```cpp
float s1 = 1.0, s2 = 2.0, s3 = 3.0, s4 = 4.0;
mave::vector<float, 3> v1(1.0, 2.0, 3.0), v2( 4.0,  5.0,  6.0),
                       v3(7.0, 8.0, 9.0), v4(10.0, 11.0, 12.0);

mave::vector<float, 3> w1, w2, w3, w4;

std::tie(w1, w2, w3, w4) = std::tie(v1, v2, v3, v4) + std::make_tuple(s1, s2, s3, s4);
```

When you want to make `mave` faster by using `rcp` and `rsqrt`, define
`MAVE_USE_APPROXIMATION` before including `mave/mave.hpp`.

## installation

This library is header-only.
The only thing that you need to do is add this library to your include path.

## building test codes

mave recommend you to build test codes and check it runs.
Because mave uses many architecture-dependent instructions, you need to verify
whether the codes would be compiled and run on your architecture.

```sh
$ git clone https://github.com/ToruNiina/mave.git
$ cd mave
$ mkdir build
$ cd build
$ cmake ..
$ make
$ make test
```

If you want to use a compiler that is different from system's defualt, you
need to pass it to cmake.

```cpp
$ cmake .. -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8
```


## reference

### matrix

```cpp
template<typename T, std::size_t R, std::size_t C>
struct matrix
{
    //    1   ...   C
    //  1 x00 ... x0M    N = R-1
    //  . x10 ... x1M    M = C-1
    //  .   . ...   .
    //  R xN0 ... xNM    R * C matrix
    static constexpr std::size_t row_size    = R;
    static constexpr std::size_t column_size = C;
    static constexpr std::size_t total_size  = R * C;
    using value_type      = T;
    using storage_type    = std::array<T, total_size>;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using size_type       = std::size_t;

    template<typename ... Ts> // sizeof...(Ts) should be equal to `total_size`.
    matrix(Ts&& ... args) noexcept;

    matrix(){}; // fill storage with 0.
    ~matrix() = default;
    matrix(const matrix&) = default;
    matrix(matrix&&)      = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&)      = default;

    template<typename U>
    matrix& operator=(const matrix<U, R, C>& rhs) noexcept;

    template<typename U>
    matrix& operator+=(const matrix<U, R, C>& rhs) noexcept;

    template<typename U>
    matrix& operator-=(const matrix<U, R, C>& rhs) noexcept;

    template<typename U>
    matrix& operator*=(const U& rhs) noexcept
    template<typename U>
    matrix& operator/=(const U& rhs) noexcept

    size_type size() const noexcept;

    reference               at(size_type i);
    const_reference         at(size_type i) const;
    reference       operator[](size_type i)       noexcept;
    const_reference operator[](size_type i) const noexcept;

    reference       at(size_type i, size_type j);
    const_reference at(size_type i, size_type j) const;
    reference       operator()(size_type i, size_type j)       noexcept;
    const_reference operator()(size_type i, size_type j) const noexcept;
};

// negation
template<typename T, std::size_t R, std::size_t C>
matrix<T, R, C> operator-(const matrix<T, R, C>& lhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> ms) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> ms) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> ms) noexcept;

// addition
template<typename T, std::size_t R, std::size_t C>
matrix<T, R, C>
operator+(const matrix<T, R, C>& lhs, const matrix<T, R, C>& rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator+(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator+(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator+(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept;

// subtraction
template<typename T, std::size_t R, std::size_t C>
matrix<T, R, C>
operator-(const matrix<T, R, C>& lhs, const matrix<T, R, C>& rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator-(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept;

// scalar multiplication
template<typename T, std::size_t R, std::size_t C>
matrix<T, R, C>
operator*(const matrix<T, R, C>& lhs, const T rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator*(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T, T> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
matrix<T, R, C>
operator*(const T lhs, const matrix<T, R, C>& rhs) noexcept

template<typename T, std::size_t R, std::size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<T, T> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator*(std::tuple<T, T, T> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> rhs) noexcept

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator*(std::tuple<T, T, T, T> lhs,
          std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> rhs) noexcept

// scalar division
template<typename T, std::size_t R, std::size_t C>
matrix<T, R, C>
operator/(const matrix<T, R, C>& lhs, const T rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
operator/(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
operator/(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T> rhs) noexcept;

template<typename T, std::size_t R, std::size_t C>
std::tuple<matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>, matrix<T,R,C>>
operator/(std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,
                     const matrix<T, R, C>&, const matrix<T, R, C>&> lhs,
          std::tuple<T, T, T, T> rhs) noexcept;

// matrix multiplication
template<typename T1, typename T2, std::size_t A, std::size_t B, std::size_t C>
matrix<U, A, C>
operator*(const matrix<T1, A, B>& lhs, const matrix<T2, B, C>& rhs) noexcept;
```

### vector


```cpp
template<typename T, std::size_t N>
using vector = matrix<T, N, 1>;
```

`mave::vector` is a column vector.

so you can multiply `mave::vector` and `mave::matrix` in the following way.

```cpp
mave::matrix<double, 3, 3> m(/*...*/);
mave::vector<double, 3>    v(/*...*/);
mave::vector<double, 3>    w = m * v;
```

#### `length_sq`

returns length(s) squared

```cpp
template<typename T>
T
length_sq(const vector<T, 3>& v) noexcept;

template<typename T>
std::pair<T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept;

template<typename T>
std::tuple<T, T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3) noexcept;

template<typename T>
std::tuple<T, T, T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept;
```

#### `length`

returns length(s)

```cpp
template<typename T>
T
length(const vector<T, 3>& v) noexcept;

template<typename T>
std::pair<T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept;

template<typename T>
std::tuple<T, T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2,
       const vector<T, 3>& v3) noexcept;

template<typename T>
std::tuple<T, T, T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2,
       const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept;

```

#### `rlength`

returns reciprocal length(s)

```cpp
template<typename T>
T
rlength(const vector<T, 3>& v) noexcept;

template<typename T>
std::pair<T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept;

template<typename T>
std::tuple<T, T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2,
        const vector<T, 3>& v3) noexcept;

template<typename T>
std::tuple<T, T, T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2,
        const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept;
```

#### `regularize`

returns regularized vector and original length

```cpp
template<typename T>
std::pair<vector<T, 3>, T>
regularize(const vector<T, 3>& v) noexcept;

template<typename T>
std::pair<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept;

template<typename T>
std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,
           std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3) noexcept;

template<typename T>
std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,
           std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept;
```

#### vector products


```cpp
template<typename T>
T dot_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept;

template<typename T>
vector<T, 3>
cross_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept;

template<typename T>
T scalar_triple_product(const vector<T, 3>& v1, const vector<T, 3>& v2,
                        const vector<T, 3>& v3) noexcept;
```

### math functions

#### `min` and `max`

applies min/max for each elements

```cpp
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
min(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept;
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
max(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept;
```

#### `floor`

applies floor for each elements

```cpp
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
floor(const matrix<T, R, C>& v) noexcept;

template<typename T, std::size_t R, std:size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
floor(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept;

template<typename T, std::size_t R, std:size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
floor(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
      const matrix<T, R, C>& v3) noexcept;

template<typename T, std::size_t R, std:size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
floor(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
      const matrix<T, R, C>& v3, const matrix<T, R, C>& v4) noexcept;
```

#### `ceil`

applies ceil for each elements

```cpp
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
ceil(const matrix<T, R, C>& v) noexcept;

template<typename T, std::size_t R, std:size_t C>
std::pair<matrix<T, R, C>, matrix<T, R, C>>
ceil(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept;

template<typename T, std::size_t R, std:size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
ceil(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
     const matrix<T, R, C>& v3) noexcept;

template<typename T, std::size_t R, std:size_t C>
std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>
ceil(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2,
     const matrix<T, R, C>& v3, const matrix<T, R, C>& v4) noexcept;
```

#### `fma`

fused multiply add

```cpp
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
fmadd(const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept;

template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
fmsub(const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept;

template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
fnmadd(const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept;
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
fnmsub(const T a, const matrix<T, R, C>& b, const matrix<T, R, C>& c) noexcept;
```

### utility

```cpp
template<typename T>
struct is_matrix;
template<typename T>
struct is_vector;
```

meta-function that inherits `std::true_type` if `T` is a `matrix` or `vector`,
respectively. otherwise, they inherits `std::false_type`.

## Licensing terms

This product is licensed under the terms of the [MIT License](LICENSE).

- Copyright (c) 2018 Toru Niina

All rights reserved.
