# ---------------------------------------------------------------------------
# Benchmarks — each is an executable outside the tests/ discovery glob. CTest
# runs each one only as a smoke test with a tiny workload (#94 T5), so a
# benchmark that stops running fails CI. For timings, see "Benchmarks" in
# CONTRIBUTING.md.
# ---------------------------------------------------------------------------

# The engine (reflection_ad::ad, which carries the flag profile) is defined in
# the top-level CMakeLists.txt; _is_clang / _is_gcc come from
# cmake/reflect_flags.cmake.

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

# Each smoke test's arguments, shrinking its workload to well under a second in
# Release. black_scholes_timing reads ITERATIONS, which every smoke test gets. A
# new benchmark needs an entry here unless it reads ITERATIONS too: otherwise its
# full workload runs, up to the smoke test's TIMEOUT.
set(_smoke_args_ad_bench 1000)
set(_smoke_args_ad_tensor_bench 10)
set(_smoke_args_black_scholes_simulated --paths 1000 --convergence-max-paths 1000)

# ---------------------------------------------------------------------------
# Helper: register one benchmark executable
# ---------------------------------------------------------------------------
function(add_reflect_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE
        reflection_ad::ad
        reflection_ad::warnings
        reflection_ad_benchmark_flags
    )
    set_target_properties(${name} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    # The Black-Scholes model (mc_sim/) is shared with the tests.
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/tests/fixtures")
    add_test(NAME benchmark_smoke.${name} COMMAND ${name} ${_smoke_args_${name}})
    set_tests_properties(benchmark_smoke.${name} PROPERTIES
        ENVIRONMENT ITERATIONS=100
        TIMEOUT 120
    )
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
