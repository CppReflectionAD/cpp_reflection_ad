# ---------------------------------------------------------------------------
# The reflection flag profile, defined once for every directory.
#
# Included from the top-level CMakeLists.txt. It sets _is_clang / _is_gcc and
# defines the INTERFACE target reflection_ad::flags, which carries the language
# standard, the reflection flags and, for clang, the fork's libc++ (compile and
# link). Targets get the profile by linking it:
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
