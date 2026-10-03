# A test declares its extra flags (`// TEST-FLAGS:`) and, if it must fail to
# compile, the error(s) it must fail with (`// EXPECT-ERROR:`) in comments at
# the top of the file. test_simple/source_directives.py reads them, for CTest
# here as for run_tests.py, so reading them needs Python 3.7 or newer at
# configure time (configure with -DBUILD_TESTING=OFF to skip the tests).
find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter)
set(_test_simple_dir "${CMAKE_CURRENT_LIST_DIR}")

# read_test_directives(<clang|gcc> [MUST_COMPILE <file>...] [MUST_FAIL <file>...])
#
# Reads, once, the directives of every test this directory registers, for
# the compiler (clang or gcc) the build uses. Call it in the directory that
# creates the tests' targets, before compile_check, run_check or
# directive_error_test. Each file gets its TEST-FLAGS (and, for a MUST_FAIL
# file, the diagnostic flags compile_fail_check.py parses) as source file
# COMPILE_FLAGS, and any error in its directives as TEST_DIRECTIVE_ERROR;
# see test_directives_cmake.py. Editing a file re-runs CMake, so they stay
# current.
function(read_test_directives compiler)
    cmake_parse_arguments(PARSE_ARGV 1 _arg "" "" "MUST_COMPILE;MUST_FAIL")
    foreach(_list IN ITEMS MUST_COMPILE MUST_FAIL)
        set(_absolute)
        foreach(_file IN LISTS _arg_${_list})
            cmake_path(ABSOLUTE_PATH _file NORMALIZE)
            list(APPEND _absolute "${_file}")
        endforeach()
        set(_${_list} ${_absolute})
    endforeach()
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/test_directives.cmake")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${_test_simple_dir}/test_directives_cmake.py"
            write --compiler ${compiler} --output "${_output}"
            --must-compile ${_MUST_COMPILE} --must-fail ${_MUST_FAIL}
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _log
        ERROR_VARIABLE _log)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "Reading the test directives failed (${_result}):\n${_log}")
    endif()
    include("${_output}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${_MUST_COMPILE} ${_MUST_FAIL}
        "${_test_simple_dir}/test_directives_cmake.py"
        "${_test_simple_dir}/source_directives.py"
        "${_test_simple_dir}/compile_fail_check.py")
    set(_test_directives_compiler ${compiler} PARENT_SCOPE)
endfunction()

# directive_error_test(<name> <source> <must_fail> <out-var>)
#
# If the directives of <source> (read by read_test_directives) are invalid,
# registers test <name> as one that fails with the reason, and sets
# <out-var> to TRUE, so the caller doesn't build it; otherwise sets it to
# FALSE.
function(directive_error_test name source must_fail out)
    get_property(_read SOURCE "${source}" PROPERTY TEST_DIRECTIVE_ERROR SET)
    if(NOT _read)
        message(FATAL_ERROR "${source}: its directives were not read; pass it "
                            "to read_test_directives() first")
    endif()
    get_property(_error SOURCE "${source}" PROPERTY TEST_DIRECTIVE_ERROR)
    if(_error STREQUAL "")
        set(${out} FALSE PARENT_SCOPE)
        return()
    endif()
    if(must_fail)
        set(_must_fail --must-fail)
    endif()
    add_test(NAME ${name} COMMAND "${Python3_EXECUTABLE}"
        "${_test_simple_dir}/test_directives_cmake.py" report
        --compiler ${_test_directives_compiler} --source "${source}" ${_must_fail})
    set(${out} TRUE PARENT_SCOPE)
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
        directive_error_test(${test_name} "${_source}" ${fail} _invalid)
        if(_invalid)
            continue()
        endif()
        add_executable(${target_name} "${_source}")
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
            add_test(NAME ${test_name} COMMAND "${Python3_EXECUTABLE}"
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
        directive_error_test(${test_name} "${_source}" FALSE _invalid)
        if(_invalid)
            continue()
        endif()
        add_executable(${test_name} "${_source}")
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
