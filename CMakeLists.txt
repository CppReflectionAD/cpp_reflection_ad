cmake_minimum_required(VERSION 3.25)
project(reflection_ad VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(cmake/reflect_flags.cmake)

# ---------------------------------------------------------------------------
# Header-only libraries under include/. Each is an INTERFACE target that
# carries the flag profile; include its headers by their path under include/,
# e.g. <ad/autograd.h> or <cx_std/cx_exp.hpp>.
#
# Only the build tree links the profile here: its flags hold paths into this
# machine's compilers. In the installed package, reflection_adConfig.cmake
# works it out for the consumer's compiler and links it to cx_std.
# ---------------------------------------------------------------------------
# constexpr maths (exp, log, sqrt, erfc, sin, cos) usable in constant evaluation
add_library(reflection_ad_cx_std INTERFACE)
add_library(reflection_ad::cx_std ALIAS reflection_ad_cx_std)
set_target_properties(reflection_ad_cx_std PROPERTIES EXPORT_NAME cx_std)
target_include_directories(reflection_ad_cx_std INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(reflection_ad_cx_std INTERFACE
    "$<BUILD_INTERFACE:reflection_ad::flags>")

# The AD engine. The headers in include/ad/clang_only/ need the clang fork.
add_library(reflection_ad_ad INTERFACE)
add_library(reflection_ad::ad ALIAS reflection_ad_ad)
set_target_properties(reflection_ad_ad PROPERTIES EXPORT_NAME ad)
target_include_directories(reflection_ad_ad INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(reflection_ad_ad INTERFACE reflection_ad::cx_std)

# ---------------------------------------------------------------------------
# Install: the headers, the targets above as reflection_ad::cx_std and
# reflection_ad::ad, and a package config for find_package(reflection_ad).
# Before 1.0 a minor version may break the API, so find_package(reflection_ad
# 0.1) accepts any 0.1.x and nothing else. The headers fit any architecture.
# tests/install/ checks that a separate project can use it.
# ---------------------------------------------------------------------------
set(REFLECTION_AD_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/reflection_ad")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS reflection_ad_cx_std reflection_ad_ad EXPORT reflection_adTargets)
install(EXPORT reflection_adTargets
    NAMESPACE reflection_ad::
    DESTINATION "${REFLECTION_AD_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/reflection_adConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion
    ARCH_INDEPENDENT)
install(FILES
    cmake/reflection_adConfig.cmake
    "${PROJECT_BINARY_DIR}/reflection_adConfigVersion.cmake"
    cmake/reflection_ad_flags.cmake
    DESTINATION "${REFLECTION_AD_INSTALL_CMAKEDIR}")

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
