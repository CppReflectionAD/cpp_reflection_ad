cmake_minimum_required(VERSION 3.25)
project(reflection_ad CXX)

# The tests read their directives with Python 3.7 or newer at configure time
# (see test_simple/test_simple_cmake.cmake); turn this off to build without.
option(BUILD_TESTING "Build the tests under tests/" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
add_subdirectory(benchmarks)
