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

include("${CMAKE_SOURCE_DIR}/test_simple/test_simple_cmake.cmake")

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
        # The fork's libc++/libc++abi depend on @rpath/libunwind.1.dylib; ld
        # treats that non-public indirect dylib as implicitly re-exported and
        # warns when it can't tie it to a parent. Linking it directly avoids
        # the warning.
        list(APPEND _reflect_link_flags -lunwind)
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
    target_compile_options(${name} PRIVATE
        -std=c++2c
        ${_reflect_flags}
    )
    target_link_options(${name} PRIVATE ${_reflect_link_flags})
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
file(GLOB _cx_std   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cx_std/*.cpp")
file(GLOB _clang_only CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/clang_only/*.cpp")
file(GLOB _gcc_only   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gcc_only/*.cpp")
file(GLOB _static_fail CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/static_fail/*.cpp")

foreach(_src IN LISTS _common _cx_std)
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

message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Static fail tests: ${_static_fail}")
