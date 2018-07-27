# synopsis

## matrix

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

## vector


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

### `length_sq`

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

### `length`

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

### `rlength`

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

### `regularize`

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

### vector products


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

## math functions

### `min` and `max`

applies min/max for each elements

```cpp
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
min(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept;
template<typename T, std::size_t R, std:size_t C>
matrix<T, R, C>
max(const matrix<T, R, C>& v1, const matrix<T, R, C>& v2) noexcept;
```

### `floor`

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

### `ceil`

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

### `fma`

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

## utility

```cpp
template<typename T>
struct is_matrix;
template<typename T>
struct is_vector;
```

meta-function that inherits `std::true_type` if `T` is a `matrix` or `vector`,
respectively. otherwise, they inherits `std::false_type`.


