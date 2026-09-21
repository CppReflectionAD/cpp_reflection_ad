# ---------------------------------------------------------------------------
# Benchmarks — built as executables but NOT registered with CTest.
# Files here are intentionally excluded from add_test() so they never
# appear in `ctest` runs or the tests/ discovery glob.
# ---------------------------------------------------------------------------

# Detect compiler family
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_is_clang TRUE)
    set(_is_gcc FALSE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_is_clang FALSE)
    set(_is_gcc TRUE)
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

if(_is_clang)
    get_filename_component(_clang_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_clang_root "${_clang_bin}" DIRECTORY)
    set(_reflect_flags
        -freflection
        -fparameter-reflection
        -fexpansion-statements
        -stdlib=libc++
        -isystem "${_clang_root}/include/c++/v1"
        # High-order Taylor derivatives build a large DAG at compile time, past
        # the default 1M constant-evaluation step budget.
        -fconstexpr-steps=16000000
    )
    if(APPLE)
        execute_process(COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE _sdk OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_sdk)
            list(APPEND _reflect_flags -isysroot "${_sdk}")
        endif()
    endif()
elseif(_is_gcc)
    set(_reflect_flags
        -freflection
        -fconstexpr-ops-limit=96000000
        # Work around a gcc-reflection bug where -O2 combined with -g breaks
        # consteval statement reflection in the shared DAG/reflection code.
        -g0
    )
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# ---------------------------------------------------------------------------
# Helper: register one benchmark executable
# ---------------------------------------------------------------------------
function(add_reflect_benchmark name source)
    add_executable(${name} ${source})
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
