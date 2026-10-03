cmake_minimum_required(VERSION 3.25)
project(reflection_ad CXX)

# Building the tests and benchmarks reads their directives with Python 3.7 or
# newer (see test_simple/test_simple_cmake.cmake); turn this off to configure
# without them.
option(BUILD_TESTING "Build the tests under tests/ and benchmarks/" ON)
if(NOT BUILD_TESTING)
    return()
endif()

# ---------------------------------------------------------------------------
# Detect compiler family from the kit-selected CMAKE_CXX_COMPILER
# ---------------------------------------------------------------------------
# _compiler is the name test directives use for it (`// TEST-FLAGS-CLANG:`).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_compiler clang)
    set(_is_clang TRUE)
    set(_is_gcc   FALSE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_compiler gcc)
    set(_is_clang FALSE)
    set(_is_gcc   TRUE)
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. "
                        "Select a clang-p2996 or gcc-reflect kit.")
endif()

include("${CMAKE_SOURCE_DIR}/test_simple/test_simple_cmake.cmake")

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
