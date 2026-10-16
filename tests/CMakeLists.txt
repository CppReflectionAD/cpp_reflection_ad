# The engine (reflection_ad::ad, which carries the flag profile) is defined in
# the top-level CMakeLists.txt; _is_clang / _is_gcc come from
# cmake/reflect_flags.cmake. See README.md here for where each kind of test goes.
include("${CMAKE_SOURCE_DIR}/test_simple/test_simple_cmake.cmake")

# Every target here, including compile_check's static_fail/ and known_gap/
# ones, sees the shared fixtures and the test_simple checks.
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    "${CMAKE_SOURCE_DIR}/test_simple"
)

# ---------------------------------------------------------------------------
# Helper: register one test executable
# ---------------------------------------------------------------------------
function(add_reflect_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE reflection_ad::ad reflection_ad::warnings)
    set_target_properties(${name} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Helper: register, as ordinary tests on `compiler` (clang or gcc), the known_gap
# files in `sources` that say they pass there with a `// TEST-PASSES-ON:` line
# at the top. One source then checks the same asserts on both compilers.
function(add_known_gap_passing_tests compiler sources)
    foreach(_src IN LISTS sources)
        # The first 10 lines, as run_tests.py reads them.
        file(STRINGS "${_src}" _head LIMIT_COUNT 10)
        list(FILTER _head INCLUDE
             REGEX "^// TEST-PASSES-ON:.*[ ,]${compiler}([ ,]|$)")
        if(_head)
            get_filename_component(_name "${_src}" NAME_WE)
            add_reflect_test(${_name} "${_src}")
        endif()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# Discover and register tests
# ---------------------------------------------------------------------------
file(GLOB _common   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp")
file(GLOB _clang_only CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/clang_only/*.cpp")
file(GLOB _gcc_only   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gcc_only/*.cpp")
file(GLOB _static_fail CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/static_fail/*.cpp")
# Known compiler gaps: code a compiler should accept but rejects because of a
# bug or missing feature in its reflection fork. Like static_fail, each file's
# expected error is pinned by a .fail.txt snapshot, so the test also fails if
# the file starts compiling (the gap is fixed) or fails for another reason.
file(GLOB _known_gap_clang CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/known_gap/clang/*.cpp")
file(GLOB _known_gap_gcc CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/known_gap/gcc/*.cpp")
# run_tests.py compiles every .cpp under tests/, so a test anywhere else (e.g.
# directly in tests/ or in fixtures/) would build there but not here.
file(GLOB_RECURSE _stray CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER _stray EXCLUDE REGEX
    "^(common|clang_only|gcc_only|static_fail|known_gap/clang|known_gap/gcc)/[^/]+\\.cpp$")
if(_stray)
    message(FATAL_ERROR "Tests outside the test directories: ${_stray}. "
                        "Move each into one of them (see tests/README.md).")
endif()

# The gap only shows with this UBSan check enabled.
set_source_files_properties(known_gap/gcc/ubsan_reflection_repro.cpp
    PROPERTIES COMPILE_OPTIONS -fsanitize=float-divide-by-zero)

# GCC's statements_of() for this file's d_return_call depends on the flags:
# optimization (any -O but -O0) together with -g turns the gap's <init_expr>
# back into a return statement. Appending -O0 pins the result its .fail.txt
# records, whatever the build type. clang builds it with the usual flags.
if(_is_gcc)
    set_source_files_properties(known_gap/gcc/nontrivial_destructor_reflection.cpp
        PROPERTIES COMPILE_OPTIONS -O0)
endif()

foreach(_src IN LISTS _common)
    get_filename_component(_name "${_src}" NAME_WE)
    add_reflect_test(${_name} "${_src}")
endforeach()

if(_is_clang)
    foreach(_src IN LISTS _clang_only)
        get_filename_component(_name "${_src}" NAME_WE)
        add_reflect_test(${_name} "${_src}")
    endforeach()
endif()

if(_is_gcc)
    foreach(_src IN LISTS _gcc_only)
        get_filename_component(_name "${_src}" NAME_WE)
        add_reflect_test(${_name} "${_src}")
    endforeach()
endif()

compile_check(reflection_ad "${_static_fail}")
if(_is_clang)
    compile_check(reflection_ad "${_known_gap_clang}")
    add_known_gap_passing_tests(clang "${_known_gap_gcc}")
endif()
if(_is_gcc)
    compile_check(reflection_ad "${_known_gap_gcc}")
    add_known_gap_passing_tests(gcc "${_known_gap_clang}")
endif()

message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Static fail tests: ${_static_fail}")
if(_is_clang)
    message(STATUS "Known gap tests: ${_known_gap_clang}")
else()
    message(STATUS "Known gap tests: ${_known_gap_gcc}")
endif()
