mave
====

[![Build Status](https://travis-ci.com/ToruNiina/mave.svg?branch=master)](https://travis-ci.com/ToruNiina/mave)

SIMD-oriented small matrix and vector library

It focuses on 3D vector operations (e.g. physical simulation, graphics, etc).

|              |    AVX                   |  AVX2                         |  AVX512           |
|:-------------|:-------------------------|:------------------------------|:------------------|
|  CPU         | Sandy Bridge, Ivy Bridge | Haswell, Broadwell, Skylake   | Skylake-SP,X,W,DE |
|              |                          | Kaby Lake, Coffee Lake        |                   |
|  matrix3x3d  | --                       | OK                            | OK                |
|  matrix3x3f  | --                       | OK                            | OK                |
|  vector3d    | OK                       | OK                            | OK                |
|  vector3f    | OK                       | OK                            | OK                |

`matrix4x4x` and `vector4x` are also planned, but not implemented now.

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

std::tie(w1, w2, w3, w4) = std::tie(v1, v2, v3, v4) * std::make_tuple(s1, s2, s3, s4);
```

When you want to make `mave` faster by using `rcp` and `rsqrt` instructions,
define `MAVE_USE_APPROXIMATION` before including `mave/mave.hpp`.

`mave::matrix` / `mave::vector` requires larger alignment than normal types.
to store `mave::matrix` / `mave::vector` in `std::vector`, you need to use
custom allocator `mave::aligned_allocator<T>`.

details are in [synopsis](SYNOPSIS.md).

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

## Licensing terms

This product is licensed under the terms of the [MIT License](LICENSE).

- Copyright (c) 2018 Toru Niina

All rights reserved.
