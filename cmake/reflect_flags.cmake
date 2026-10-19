# ---------------------------------------------------------------------------
# The reflection flag profile, defined once for every directory.
#
# Included from the top-level CMakeLists.txt. It sets _is_clang / _is_gcc and
# defines the INTERFACE target reflection_ad::flags, which carries the language
# standard, the reflection flags and the fork's standard library: libc++ for
# clang (compile and link), and an rpath to libstdc++ for GCC, so test binaries
# run with no LD_LIBRARY_PATH. The flags themselves come from
# reflection_ad_set_flags() in reflection_ad_flags.cmake, which the installed
# package also uses. Targets get the profile by linking it:
#
#     target_link_libraries(<target> PRIVATE reflection_ad::flags)
#
# It also defines reflection_ad::warnings, the warning flags for tests and
# benchmarks (see the end of this file).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Detect compiler family from the kit-selected CMAKE_CXX_COMPILER
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_is_clang TRUE)
    set(_is_gcc   FALSE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_is_clang FALSE)
    set(_is_gcc   TRUE)
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. "
                        "Select a clang-p2996 or gcc-reflect kit.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/reflection_ad_flags.cmake")

add_library(reflection_ad_flags INTERFACE)
add_library(reflection_ad::flags ALIAS reflection_ad_flags)
reflection_ad_set_flags(reflection_ad_flags)

# ---------------------------------------------------------------------------
# Sanitizers (opt-in): AddressSanitizer and UndefinedBehaviorSanitizer for
# every target, via the profile above. The *-sanitize presets turn this on.
# A UBSan report aborts the test, so ctest sees it as a failure.
# ---------------------------------------------------------------------------
option(REFLECTION_AD_SANITIZE
    "Build tests and benchmarks with AddressSanitizer and UBSan" OFF)
if(REFLECTION_AD_SANITIZE)
    set(_sanitize_flags
        -fsanitize=address,undefined
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer
    )
    if(_is_gcc)
        # GCC's UBSan instrumentation leaks into the expression trees that
        # std::meta sees (tests/known_gap/gcc/ubsan_reflection_repro.cpp), and
        # each of these checks on its own breaks some tests' static_asserts.
        # Drop this once that known gap is fixed.
        list(APPEND _sanitize_flags
            -fno-sanitize=shift,integer-divide-by-zero,signed-integer-overflow,null,alignment,vptr)
    endif()
    set(_sanitize_link_flags -fsanitize=address,undefined)

    # Check that the runtimes are there, so a toolchain without them fails
    # here rather than at the first link (or, worse, builds without them).
    get_target_property(_profile_flags reflection_ad_flags INTERFACE_COMPILE_OPTIONS)
    get_target_property(_profile_link_flags reflection_ad_flags INTERFACE_LINK_OPTIONS)
    if(NOT _profile_link_flags) # GCC on macOS needs none
        set(_profile_link_flags)
    endif()
    set(_sanitize_probe_key
        "${REFLECTION_AD_PROBED};${_sanitize_flags};${_sanitize_link_flags}")
    if(NOT "${_sanitize_probe_key}" STREQUAL "${REFLECTION_AD_SANITIZE_PROBED}")
        try_compile(_sanitize_ok
            SOURCE_FROM_CONTENT sanitize_probe.cpp [=[
#include <vector>
int main(int argc, char**) { return std::vector<int>(argc).size() == 0; }
]=]
            COMPILE_DEFINITIONS ${_profile_flags} ${_sanitize_flags}
            LINK_OPTIONS ${_profile_link_flags} ${_sanitize_link_flags}
            OUTPUT_VARIABLE _sanitize_output)
        if(NOT _sanitize_ok)
            string(REGEX MATCHALL "[^\n]*(error|cannot find)[^\n]*"
                _sanitize_errors "${_sanitize_output}")
            list(JOIN _sanitize_errors "\n  " _sanitize_errors)
            list(JOIN _sanitize_flags " " _sanitize_flags_text)
            message(FATAL_ERROR
                "REFLECTION_AD_SANITIZE is ON, but ${CMAKE_CXX_COMPILER} can't "
                "build and link a program with ${_sanitize_flags_text}.\n"
                "For clang, the toolchain needs compiler-rt: rebuild it with "
                "`python3 run_tests.py --build-compilers --compiler clang`. "
                "For GCC, the toolchain needs libsanitizer (libasan, libubsan), "
                "which the fork builds by default on Linux and macOS.\n"
                "Compiler errors:\n  ${_sanitize_errors}")
        endif()
        set(REFLECTION_AD_SANITIZE_PROBED "${_sanitize_probe_key}" CACHE INTERNAL
            "Compiler and flags that passed the sanitizer probe")
    endif()

    target_compile_options(reflection_ad_flags INTERFACE ${_sanitize_flags})
    target_link_options(reflection_ad_flags INTERFACE ${_sanitize_link_flags})
    unset(_sanitize_flags)
    unset(_sanitize_link_flags)
endif()

# ---------------------------------------------------------------------------
# Warnings. Tests and benchmarks link reflection_ad::warnings and set
# COMPILE_WARNING_AS_ERROR, so a warning fails their build; configure with
# `cmake --compile-no-warning-as-error` to see them as warnings instead.
# compile_check's targets don't use it: they are expected to fail, and a
# warning turned into an error would change their .fail.txt snapshot.
# ---------------------------------------------------------------------------
add_library(reflection_ad_warnings INTERFACE)
add_library(reflection_ad::warnings ALIAS reflection_ad_warnings)
target_compile_options(reflection_ad_warnings INTERFACE -Wall -Wextra)
