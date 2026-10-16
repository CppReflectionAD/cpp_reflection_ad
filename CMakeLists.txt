cmake_minimum_required(VERSION 3.25)
project(reflection_ad CXX)

include(cmake/reflect_flags.cmake)

# ---------------------------------------------------------------------------
# Header-only libraries under include/. Each is an INTERFACE target that
# carries the flag profile; include its headers by their path under include/,
# e.g. <ad/autograd.h> or <cx_std/cx_exp.hpp>.
# ---------------------------------------------------------------------------
# constexpr maths (exp, log, sqrt, erfc, sin, cos) usable in constant evaluation
add_library(reflection_ad_cx_std INTERFACE)
add_library(reflection_ad::cx_std ALIAS reflection_ad_cx_std)
target_include_directories(reflection_ad_cx_std INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(reflection_ad_cx_std INTERFACE reflection_ad::flags)

# The AD engine. The headers in include/ad/clang_only/ need the clang fork.
add_library(reflection_ad_ad INTERFACE)
add_library(reflection_ad::ad ALIAS reflection_ad_ad)
target_include_directories(reflection_ad_ad INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(reflection_ad_ad INTERFACE reflection_ad::cx_std)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
