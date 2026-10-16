# ---------------------------------------------------------------------------
# Benchmarks — built as executables but NOT registered with CTest.
# Files here are intentionally excluded from add_test() so they never
# appear in `ctest` runs or the tests/ discovery glob.
# ---------------------------------------------------------------------------

# The flag profile (reflection_ad::flags) and _is_clang / _is_gcc come from
# cmake/reflect_flags.cmake, included by the top-level CMakeLists.txt.

# High-order Taylor derivatives build a large DAG at compile time, past the
# default constant-evaluation budget.
add_library(reflection_ad_benchmark_flags INTERFACE)
if(_is_clang)
    target_compile_options(reflection_ad_benchmark_flags INTERFACE
        -fconstexpr-steps=16000000)
else()
    target_compile_options(reflection_ad_benchmark_flags INTERFACE
        -fconstexpr-ops-limit=96000000)
endif()

# ---------------------------------------------------------------------------
# Helper: register one benchmark executable
# ---------------------------------------------------------------------------
function(add_reflect_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE
        reflection_ad::flags
        reflection_ad_benchmark_flags
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
        # clang-only engine headers (tensor.h, autograd_tensor.h, ...)
        target_include_directories(${_name} PRIVATE "${CMAKE_SOURCE_DIR}/tests/clang_only")
    endforeach()
endif()
