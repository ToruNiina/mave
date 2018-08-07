mave
====

[![Build Status](https://travis-ci.com/ToruNiina/mave.svg?branch=master)](https://travis-ci.com/ToruNiina/mave)

SIMD-oriented small matrix and vector library

It focuses on 3D vector operations (e.g. physical simulation, graphics, etc).

![logo](https://github.com/ToruNiina/mave/blob/imgs/misc/logo1.png)

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

In `mave`, since `matrix` is templatized, you can use matrix of any dimension.
With matrices listed above, `mave` automatically use SIMD instructions internally.

after including `mave/mave.hpp`, compile your code with
`-march=native -mtune=native` to turn simd flags (e.g. `__AVX2__`) on.

Basic `vector/matrix` arithmetic operators are supported.

```cpp
const mave::vector<double, 3> v1(1, 2, 3);
const mave::vector<double, 3> v2(2, 3, 4);

const mave::vector<double, 3> v3 = v1 + v2;
```

You know, with AVX instructions, you can add 8x`float` values at once with `vaddps`.
With `mave`, you can add 2 `vector3f` at once with `std::tie`.

```cpp
const mave::vector<float, 3> v1(/*...*/), v2(/*...*/);
const mave::vector<float, 3> w1(/*...*/), w2(/*...*/);
mave::vector<float, 3> u1, u2;

std::tie(u1, u2) = std::tie(v1, v2) + std::tie(w1, w2); // use vaddps
```

C++17 structured binding makes it more convenient.

```cpp
const mave::vector<float, 3> v1(/*...*/), v2(/*...*/);
const mave::vector<float, 3> w1(/*...*/), w2(/*...*/);

const auto [u1, u2] = std::tie(v1, v2) + std::tie(w1, w2);
```

You can chain this expressions as you want.

```cpp
const auto [u1, u2] = length(std::tie(v1, v2) - std::tie(w1, w2)) *
                      std::tie(v1, v2) / std::make_tuple(2.0, 4.0);
// this is equivalent to
const auto u1 = length(v1 - w1) * v1 / 2.0;
const auto u2 = length(v2 - w2) * v2 / 4.0;
```

`mave` also supports small matrix.

```cpp
mave::matrix<double, 3, 3> m(1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0);
mave::vector<double, 3>    v(1.0, 2.0, 3.0);

mave::vector<double, 3>    w = m * v;
```

When you want to make `mave` faster by using `rcp` and `rsqrt` instructions
which approximates `1/x` and `1/sqrt(x)`, define `MAVE_USE_APPROXIMATION`
before including `mave/mave.hpp`.

`mave::matrix` / `mave::vector` requires larger alignment than normal types.
to store `mave::matrix` / `mave::vector` in `std::vector` or some other
containers, you need to use custom allocator `mave::aligned_allocator<T>`.

details are in [synopsis](SYNOPSIS.md).

## installation

This library is header-only.
The only thing that you need to do is add this library to your include path.

## building test codes

`mave` recommend you to run test codes.
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

If you want to use a non-default compiler, you need to pass it to cmake.

```cpp
$ cmake .. -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8
```

## What is missing?

- [ ] optimize `cross_product`
- [ ] optimize matrix-vector multiplication
- [ ] optimize matrix-matrix multiplication
- [ ] approximate `operator/` by `rcp` for `matrix`
- [ ] specialize `vector<float,  4>`
- [ ] specialize `vector<dobule, 4>`
- [ ] specialize `matrix<float,  4, 4>`
- [ ] specialize `matrix<dobule, 4, 4>`

## Licensing terms

This product is licensed under the terms of the [MIT License](LICENSE).

- Copyright (c) 2018 Toru Niina

All rights reserved.
