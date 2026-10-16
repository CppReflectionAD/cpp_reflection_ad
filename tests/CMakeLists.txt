# The flag profile (reflection_ad::flags) and _is_clang / _is_gcc come from
# cmake/reflect_flags.cmake, included by the top-level CMakeLists.txt.
include("${CMAKE_SOURCE_DIR}/test_simple/test_simple_cmake.cmake")

# ---------------------------------------------------------------------------
# Helper: register one test executable
# ---------------------------------------------------------------------------
function(add_reflect_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE reflection_ad::flags)
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/include"
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
# Known compiler gaps: code a compiler should accept but rejects because of a
# bug or missing feature in its reflection fork. Like static_fail, each file's
# expected error is pinned by a .fail.txt snapshot, so the test also fails if
# the file starts compiling (the gap is fixed) or fails for another reason.
file(GLOB _known_gap_clang CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/known_gap/clang/*.cpp")
file(GLOB _known_gap_gcc CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/known_gap/gcc/*.cpp")
# The gap only shows with this UBSan check enabled.
set_source_files_properties(known_gap/gcc/ubsan_reflection_repro.cpp
    PROPERTIES COMPILE_OPTIONS -fsanitize=float-divide-by-zero)

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
if(_is_clang)
    compile_check(reflection_ad "${_known_gap_clang}")
endif()
if(_is_gcc)
    compile_check(reflection_ad "${_known_gap_gcc}")
endif()

message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Static fail tests: ${_static_fail}")
if(_is_clang)
    message(STATUS "Known gap tests: ${_known_gap_clang}")
else()
    message(STATUS "Known gap tests: ${_known_gap_gcc}")
endif()
