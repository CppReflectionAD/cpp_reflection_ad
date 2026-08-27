# ---------------------------------------------------------------------------
# Benchmarks — built as executables but NOT registered with CTest.
# Files here are intentionally excluded from add_test() so they never
# appear in `ctest` runs or the tests/ discovery glob.
# ---------------------------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_reflect_flags -freflection)
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# ---------------------------------------------------------------------------
# Discover and register benchmarks
# ---------------------------------------------------------------------------
file(GLOB _benchmarks CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(_src IN LISTS _benchmarks)
    get_filename_component(_name "${_src}" NAME_WE)
    add_executable(${_name} "${_src}")
    target_compile_options(${_name} PRIVATE
        -std=c++2c
        ${_reflect_flags}
    )
    target_include_directories(${_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/tests"
    )
    # Intentionally no add_test() — benchmarks are run manually.
endforeach()
