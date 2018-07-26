cmake_minimum_required(VERSION 3.0)
project(mave)
enable_testing()

include_directories("${PROJECT_SOURCE_DIR}")

if(EXISTS "${PROJECT_SOURCE_DIR}/extlib/boost_1_67_0/boost/version.hpp")
    message(STATUS "boost 1.67.0 exists.")
else()
    message(STATUS "downloading Boost 1.67.0 to ./extlib/ ...")
    execute_process(COMMAND wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.bz2
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib"
        RESULT_VARIABLE DOWNLOAD_BOOST_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NOT ${DOWNLOAD_BOOST_RESULT} EQUAL "0")
        file(REMOVE ${PROJECT_SOURCE_DIR}/extlib/boost_1_67_0.tar.bz2)
        message(FATAL_ERROR "failed to download Boost 1.67.0. please confirm network connections.")
    endif()

    execute_process(COMMAND which sha256sum RESULT_VARIABLE SHA256SUM_EXISTS OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND which shasum    RESULT_VARIABLE SHASUM_EXISTS    OUTPUT_QUIET ERROR_QUIET)
    if(${SHA256SUM_EXISTS} EQUAL "0")
        execute_process(COMMAND sha256sum --check boost_1_67_0_tar_bz2_sha256sum.dat
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib"
            RESULT_VARIABLE   DOWNLOADED_SHA256SUM_RESULT)
    elseif(${SHASUM_EXISTS} EQUAL "0")
        execute_process(COMMAND shasum --algorithm 256 --check boost_1_67_0_tar_bz2_sha256sum.dat
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib"
            RESULT_VARIABLE   DOWNLOADED_SHA256SUM_RESULT)
    endif()

    if(NOT ${DOWNLOADED_SHA256SUM_RESULT} EQUAL "0")
        file(REMOVE ${PROJECT_SOURCE_DIR}/extlib/boost_1_67_0.tar.bz2)
        message(FATAL_ERROR "boost_1_67_0.tar.bz2 has invalid sha256sum!")
    endif()

    execute_process(COMMAND tar xf boost_1_67_0.tar.bz2
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extlib"
        OUTPUT_QUIET ERROR_QUIET)

    message(STATUS "done.")
endif()
include_directories(${PROJECT_SOURCE_DIR}/extlib/boost_1_67_0)

OPTION(MAVE_USE_APPROX "use approximation functions" OFF)
if(MAVE_USE_APPROX)
    add_definitions(-DMAVE_USE_APPROXIMATION)
endif(MAVE_USE_APPROX)

add_subdirectory(tests)
