# A test declares its extra flags (`// TEST-FLAGS:`) and, if it must fail to
# compile, the error(s) it must fail with (`// EXPECT-ERROR:`) in comments at
# the top of the file. test_simple/source_directives.py reads them, for CTest
# here as for run_tests.py, when a test or benchmark is built, so building
# them needs Python 3.7 or newer (configure with -DBUILD_TESTING=OFF to skip
# them).
find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter)
set(_test_simple_dir "${CMAKE_CURRENT_LIST_DIR}")
# Every Python script runs with -B, so that it writes no bytecode into the
# source tree.
set(_test_python "${Python3_EXECUTABLE}" -B)
# The build's compiler, as test directives name it (`// TEST-FLAGS-CLANG:`),
# which the including file (the top-level CMakeLists.txt) sets as _compiler.
if(NOT _compiler MATCHES "^(clang|gcc)$")
    message(FATAL_ERROR "Set _compiler to clang or gcc before including "
                        "test_simple_cmake.cmake")
endif()
set(_test_directives_compiler ${_compiler})

# test_flags(<target> <source> <must_fail>)
#
# Builds <source>, in <target>, with the flags its directives give it for the
# build's compiler: its TEST-FLAGS and, if <must_fail>, the diagnostic flags
# compile_fail_check.py parses. test_directives_cmake.py writes them to a
# response file, which the compiler reads after CMAKE_CXX_FLAGS and the
# target's own options, before <source> is compiled and again whenever it
# changes; so editing a test doesn't re-run CMake. The script rewrites the
# response file only when the flags change (Ninja then restats it), so a
# change that leaves them alone recompiles nothing else. If the directives of
# <source> are invalid, building <target> fails with the reason.
function(test_flags target source must_fail)
    set(_rsp "${CMAKE_CURRENT_BINARY_DIR}/test_flags/${target}.rsp")
    if(must_fail)
        set(_must_fail --must-fail)
    endif()
    add_custom_command(OUTPUT "${_rsp}"
        COMMAND ${_test_python} "${_test_simple_dir}/test_directives_cmake.py"
            --compiler ${_test_directives_compiler} --source "${source}"
            --output "${_rsp}" ${_must_fail}
        DEPENDS "${source}"
            "${_test_simple_dir}/test_directives_cmake.py"
            "${_test_simple_dir}/source_directives.py"
            "${_test_simple_dir}/compile_fail_check.py"
        COMMENT "Reading the test directives of ${source}"
        VERBATIM)
    # A source of <target>, so that its build has a rule for the response
    # file; and a dependency of the object, so that it is rebuilt when the
    # flags change.
    target_sources(${target} PRIVATE "${_rsp}")
    set_property(SOURCE "${source}" APPEND PROPERTY OBJECT_DEPENDS "${_rsp}")
    set_property(SOURCE "${source}" APPEND PROPERTY COMPILE_OPTIONS "@${_rsp}")
endfunction()

function(compile_check group filelist fail)
    foreach(testfile IN LISTS filelist)
        if(IS_ABSOLUTE "${testfile}")
            set(_source "${testfile}")
            file(RELATIVE_PATH target "${CMAKE_CURRENT_SOURCE_DIR}" "${testfile}")
        else()
            set(_source "${CMAKE_CURRENT_SOURCE_DIR}/${testfile}")
            set(target "${testfile}")
        endif()
        string(REPLACE .cpp "" target ${target})
        string(REPLACE / "." target ${target})
        set(target_name "${group}.static.${target}")
        set(test_name "${target}")
        # Compiled, never linked, so that only the compiler's diagnostics
        # reach compile_fail_check.py.
        add_library(${target_name} OBJECT "${_source}")
        test_flags(${target_name} "${_source}" ${fail})
        set_target_properties(${target_name} PROPERTIES EXCLUDE_FROM_ALL true EXCLUDE_FROM_DEFAULT_BUILD true)
        target_compile_options(${target_name} PRIVATE
            -std=c++2c
            ${_reflect_flags}
        )
        target_include_directories(${target_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_SOURCE_DIR}/test_simple"
        )
        if (fail)
            # Must fail with the error(s) its `// EXPECT-ERROR:` comments name.
            add_test(NAME ${test_name} COMMAND ${_test_python}
                "${_test_simple_dir}/compile_fail_check.py"
                --source "${_source}" --compiler ${_test_directives_compiler}
                -- ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${target_name})
        else()
            add_test(NAME ${test_name} COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${target_name})
        endif()
    endforeach()
endfunction()

function(run_check group filelist)
    foreach(testfile IN LISTS filelist)
        if(IS_ABSOLUTE "${testfile}")
            set(_source "${testfile}")
            file(RELATIVE_PATH target "${CMAKE_CURRENT_SOURCE_DIR}" "${testfile}")
        else()
            set(_source "${CMAKE_CURRENT_SOURCE_DIR}/${testfile}")
            set(target "${testfile}")
        endif()
        string(REPLACE .cpp "" target ${target})
        string(REPLACE / "." target ${target})
        set(test_name "${group}.dynamic.${target}")
        add_executable(${test_name} "${_source}")
        test_flags(${test_name} "${_source}" FALSE)
        target_compile_options(${test_name} PRIVATE
            -std=c++2c
            ${_reflect_flags}
        )
        target_include_directories(${test_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_SOURCE_DIR}/test_simple"
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endfunction()
