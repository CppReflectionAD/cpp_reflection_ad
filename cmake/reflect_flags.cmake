# ---------------------------------------------------------------------------
# The reflection flag profile, defined once for every directory.
#
# Included from the top-level CMakeLists.txt. It sets _is_clang / _is_gcc and
# defines the INTERFACE target reflection_ad::flags, which carries the language
# standard, the reflection flags and the fork's standard library: libc++ for
# clang (compile and link), and an rpath to libstdc++ for GCC, so test binaries
# run with no LD_LIBRARY_PATH. Targets get the profile by linking it:
#
#     target_link_libraries(<target> PRIVATE reflection_ad::flags)
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

# ---------------------------------------------------------------------------
# Reflection compile and link flags
# ---------------------------------------------------------------------------
set(_reflect_link_flags)
if(_is_clang)
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

    # Link and load the fork's libc++ rather than the system one, as
    # run_tests.py does.
    set(_reflect_link_flags
        -stdlib=libc++
        "-L${_clang_root}/lib"
        "-Wl,-rpath,${_clang_root}/lib"
    )
    # On Linux the runtimes build puts libc++ in a per-target subdirectory.
    file(GLOB _clang_target_lib_dirs LIST_DIRECTORIES true
        "${_clang_root}/lib/*linux*")
    foreach(_dir IN LISTS _clang_target_lib_dirs)
        if(IS_DIRECTORY "${_dir}")
            list(APPEND _reflect_link_flags "-L${_dir}" "-Wl,-rpath,${_dir}")
        endif()
    endforeach()

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

    # The fork's libstdc++ is newer than the system one, and the tests use
    # symbols only it exports (GLIBCXX_3.4.35), so binaries must load the
    # fork's copy, as run_tests.py arranges. The compiler says where its
    # runtime libraries are; a bare name back means it has none of its own.
    foreach(_lib libstdc++.so libgcc_s.so.1)
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=${_lib}
            OUTPUT_VARIABLE _path OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(IS_ABSOLUTE "${_path}")
            file(REAL_PATH "${_path}" _path)
            get_filename_component(_dir "${_path}" DIRECTORY)
            if(NOT "-Wl,-rpath,${_dir}" IN_LIST _reflect_link_flags)
                list(APPEND _reflect_link_flags "-Wl,-rpath,${_dir}")
            endif()
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Check that the compiler supports reflection, so that a wrong compiler gives
# one message here instead of a wall of errors in the first test. The result
# is cached for this compiler and flag set.
# ---------------------------------------------------------------------------
set(_probe_key "${CMAKE_CXX_COMPILER};${_reflect_flags};${_reflect_link_flags}")
if(NOT "${_probe_key}" STREQUAL "${REFLECTION_AD_PROBED}")
    try_compile(_reflection_ok
        SOURCE_FROM_CONTENT reflection_probe.cpp [=[
#include <meta>
constexpr std::meta::info r = ^^int;
static_assert(r == ^^int);
int main() {}
]=]
        COMPILE_DEFINITIONS -std=c++2c ${_reflect_flags}
        LINK_OPTIONS ${_reflect_link_flags}
        OUTPUT_VARIABLE _probe_output)
    if(NOT _reflection_ok)
        string(REGEX MATCHALL "[^\n]*error[^\n]*" _probe_errors "${_probe_output}")
        list(JOIN _probe_errors "\n  " _probe_errors)
        message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER} can't compile a minimal reflection program "
            "(#include <meta>, ^^int) with this project's flags.\n"
            "This project needs the clang-p2996 or the GCC reflection fork. "
            "Build both with `python3 run_tests.py --build-compilers` (into "
            "build/clang-p2996/bin/clang++ and build/gcc-p2996/artifacts/bin/g++), "
            "then configure a fresh build directory with "
            "-DCMAKE_CXX_COMPILER=<one of them>.\n"
            "Compiler errors:\n  ${_probe_errors}")
    endif()
    set(REFLECTION_AD_PROBED "${_probe_key}" CACHE INTERNAL
        "Compiler and flags that passed the reflection probe")
endif()

add_library(reflection_ad_flags INTERFACE)
add_library(reflection_ad::flags ALIAS reflection_ad_flags)
target_compile_options(reflection_ad_flags INTERFACE
    -std=c++2c
    ${_reflect_flags}
)
target_link_options(reflection_ad_flags INTERFACE ${_reflect_link_flags})

# The target is the one way to get the profile.
unset(_reflect_flags)
unset(_reflect_link_flags)
