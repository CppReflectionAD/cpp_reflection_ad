# _is_clang / _is_gcc and test_flags() come from the top-level CMakeLists.txt.

# ---------------------------------------------------------------------------
# Reflection compile flags
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helper: register one test executable
# ---------------------------------------------------------------------------
function(add_reflect_test name source)
    add_executable(${name} ${source})
    test_flags(${name} "${source}" FALSE)
    target_compile_options(${name} PRIVATE
        -std=c++2c
        ${_reflect_flags}
    )
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/test_simple"
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ---------------------------------------------------------------------------
# Discover and register tests
# ---------------------------------------------------------------------------
file(GLOB _common   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB _clang_only CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/clang_only/*.cpp")
file(GLOB _gcc_only   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gcc_only/*.cpp")
file(GLOB_RECURSE _static_fail CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/static_fail/*.cpp")
file(GLOB_RECURSE _clang_only_static_fail CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/clang_only/static_fail/*.cpp")
file(GLOB_RECURSE _gcc_only_static_fail CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/gcc_only/static_fail/*.cpp")

set(_must_compile ${_common})
if(_is_clang)
    list(APPEND _must_compile ${_clang_only})
    list(APPEND _static_fail ${_clang_only_static_fail})
endif()
if(_is_gcc)
    list(APPEND _must_compile ${_gcc_only})
    list(APPEND _static_fail ${_gcc_only_static_fail})
endif()

foreach(_src IN LISTS _must_compile)
    get_filename_component(_name "${_src}" NAME_WE)
    add_reflect_test(${_name} "${_src}")
endforeach()

compile_check(reflection_ad "${_static_fail}" TRUE)

message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Static fail tests: ${_static_fail}")
