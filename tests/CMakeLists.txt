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
    target_compile_options(${name} PRIVATE
        -std=c++2c
        ${_reflect_flags}
    )
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ---------------------------------------------------------------------------
# Discover and register tests
# ---------------------------------------------------------------------------
file(GLOB _common   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB _clang_only CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/clang_only/*.cpp")
file(GLOB _gcc_only   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gcc_only/*.cpp")

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

message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
