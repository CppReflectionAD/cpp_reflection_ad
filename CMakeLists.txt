cmake_minimum_required(VERSION 3.25)
project(reflection_ad CXX)

# Building the tests and benchmarks reads their directives with Python 3.7 or
# newer (see test_simple/test_simple_cmake.cmake); turn this off to configure
# without them. Off by default when this project is a subproject of another.
option(REFLECTION_AD_BUILD_TESTING "Build the tests under tests/ and benchmarks/"
       ${PROJECT_IS_TOP_LEVEL})
if(NOT REFLECTION_AD_BUILD_TESTING)
    return()
endif()

# ---------------------------------------------------------------------------
# Detect compiler family from the kit-selected CMAKE_CXX_COMPILER
# ---------------------------------------------------------------------------
# _compiler is the name test directives use for it (`// TEST-FLAGS-CLANG:`).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_compiler clang)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_compiler gcc)
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. "
                        "Select a clang-p2996 or gcc-reflect kit.")
endif()

# ---------------------------------------------------------------------------
# Reflection compile flags, for the tests and the benchmarks
# ---------------------------------------------------------------------------
if(_compiler STREQUAL "clang")
    # Derive the compiler root from the executable path (bin/../)
    get_filename_component(_clang_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_clang_root "${_clang_bin}" DIRECTORY)

    set(_reflect_flags
        -freflection
        -fparameter-reflection
        -fexpansion-statements
        -stdlib=libc++
        -isystem "${_clang_root}/include/c++/v1"
    )

    if(APPLE)
        execute_process(COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE _sdk OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_sdk)
            list(APPEND _reflect_flags -isysroot "${_sdk}")
        endif()
    endif()

else() # GCC
    set(_reflect_flags
        -freflection
    )
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/test_simple/test_simple_cmake.cmake")

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
