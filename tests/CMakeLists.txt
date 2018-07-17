set(TEST_NAMES
    test_length_sq
    test_length
    test_rlength
    test_regularize
    )

add_definitions("-std=c++11 -O3 -march=native -mtune=native")

foreach(TEST_NAME ${TEST_NAMES})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach(TEST_NAME)
