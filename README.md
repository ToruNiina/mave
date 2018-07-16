mave
====

SIMD-oriented small matrix and vector library

It focuses on 3D vector operations (e.g. physical simulation, graphics, etc).

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

### `vector<T, 3>`

#### `length_sq`

returns length(s) squared

```cpp
template<typename T>
T
length_sq(const vector<T, 3>& v) noexcept

template<typename T>
std::pair<T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept

template<typename T>
std::tuple<T, T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3) noexcept

template<typename T>
std::tuple<T, T, T, T>
length_sq(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept

```

#### `length`

returns length(s)

```cpp
template<typename T>
T
length(const vector<T, 3>& v) noexcept

template<typename T>
std::pair<T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept

template<typename T>
std::tuple<T, T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3) noexcept

template<typename T>
std::tuple<T, T, T, T>
length(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept

```

#### `rlength`

returns reciprocal length(s)

```cpp
template<typename T>
T
rlength(const vector<T, 3>& v) noexcept

template<typename T>
std::pair<T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept

template<typename T>
std::tuple<T, T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3) noexcept

template<typename T>
std::tuple<T, T, T, T>
rlength(const vector<T, 3>& v1, const vector<T, 3>& v2,
          const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept

```

#### `regularize`

returns regularized vector and original length

```cpp
template<typename T>
std::pair<vector<T, 3>, T>
regularize(const vector<T, 3>& v) noexcept

template<typename T>
std::pair<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept

template<typename T>
std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,
           std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3) noexcept

template<typename T>
std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,
           std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>
regularize(const vector<T, 3>& v1, const vector<T, 3>& v2,
           const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
```

#### `min` and `max`

applies min/max for each elements

```cpp
template<typename T>
vector<T, 3> min(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
template<typename T>
vector<T, 3> max(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
```

#### `floor`

applies floor for each elements

```cpp
template<typename T>
vector<T, 3> floor(const vector<T, 3>& v) noexcept
template<typename T>
std::pair<vector<T, 3>, vector<T, 3>>
floor(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
template<typename T>
std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>>
floor(const vector<T, 3>& v1, const vector<T, 3>& v2,
      const vector<T, 3>& v3) noexcept
template<typename T>
std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>, vector<T, 3>>
floor(const vector<T, 3>& v1, const vector<T, 3>& v2,
      const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
```

#### `ceil`

applies ceil for each elements

```cpp
template<typename T>
vector<T, 3> ceil(const vector<T, 3>& v) noexcept
template<typename T>
std::pair<vector<T, 3>, vector<T, 3>>
ceil(const vector<T, 3>& v1, const vector<T, 3>& v2) noexcept
template<typename T>
std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>>
ceil(const vector<T, 3>& v1, const vector<T, 3>& v2,
     const vector<T, 3>& v3) noexcept
template<typename T>
std::tuple<vector<T, 3>, vector<T, 3>, vector<T, 3>, vector<T, 3>>
ceil(const vector<T, 3>& v1, const vector<T, 3>& v2,
     const vector<T, 3>& v3, const vector<T, 3>& v4) noexcept
```

#### `fma`

fused multiply add

```cpp
template<typename T>
vector<T, 3> fmadd(const T a,
                   const vector<T, 3>& b, const vector<T, 3>& c) noexcept

template<typename T>
vector<T, 3> fmsub(const T a,
                   const vector<T, 3>& b, const vector<T, 3>& c) noexcept

template<typename T>
vector<T, 3> fnmadd(const T a,
                    const vector<T, 3>& b, const vector<T, 3>& c) noexcept
template<typename T>
vector<T, 3> fnmsub(const T a,
                    const vector<T, 3>& b, const vector<T, 3>& c) noexcept
```

#### vector operations


```cpp
template<typename T>
T dot_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept

template<typename T>
vector<T, 3>
cross_product(const vector<T, 3>& lhs, const vector<T, 3>& rhs) noexcept

template<typename T>
T scalar_triple_product(const vector<T, 3>& v1, const vector<T, 3>& v2,
                        const vector<T, 3>& v3) noexcept
```

## Licensing terms

This product is licensed under the terms of the [MIT License](LICENSE).

- Copyright (c) 2018 Toru Niina

All rights reserved.
