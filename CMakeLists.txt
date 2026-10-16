cmake_minimum_required(VERSION 3.25)
project(reflection_ad CXX)

include(cmake/reflect_flags.cmake)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
