# ---------------------------------------------------------------------------
# Benchmarks — built as executables but NOT registered with CTest.
# Files here are intentionally excluded from add_test() so they never
# appear in `ctest` runs or the tests/ discovery glob.
# ---------------------------------------------------------------------------

# _is_clang / _is_gcc and test_flags() come from the top-level CMakeLists.txt.
# A benchmark declares the flags it needs (-O2, a constant-evaluation budget)
# in `// TEST-FLAGS:` comments, which test_flags() reads, as run_tests.py does.

if(_is_clang)
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
elseif(_is_gcc)
    set(_reflect_flags -freflection)
endif()

# ---------------------------------------------------------------------------
# Helper: register one benchmark executable
# ---------------------------------------------------------------------------
function(add_reflect_benchmark name source)
    add_executable(${name} ${source})
    test_flags(${name} "${source}" FALSE)
    target_compile_options(${name} PRIVATE
        -std=c++2c
        ${_reflect_flags}
    )
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/tests"
    )
    # Intentionally no add_test() — benchmarks are run manually.
endfunction()

# ---------------------------------------------------------------------------
# Discover and register benchmarks
# ---------------------------------------------------------------------------
file(GLOB _common_benchmarks CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB _clang_only_benchmarks CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/clang_only/*.cpp")

foreach(_src IN LISTS _common_benchmarks)
    get_filename_component(_name "${_src}" NAME_WE)
    add_reflect_benchmark(${_name} "${_src}")
endforeach()

if(_is_clang)
    foreach(_src IN LISTS _clang_only_benchmarks)
        get_filename_component(_name "${_src}" NAME_WE)
        add_reflect_benchmark(${_name} "${_src}")
    endforeach()
endif()
