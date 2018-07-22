set(TEST_NAMES
    test_length_sq
    test_length
    test_rlength
    test_regularize
    test_add
    test_sub
    test_mul
    test_div
    test_fma
    test_neg
    test_ceil
    test_floor
    test_max
    test_min
    test_mat_vec_mul
    )

add_definitions("-std=c++11 -O3 -march=native -mtune=native")

foreach(TEST_NAME ${TEST_NAMES})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach(TEST_NAME)
