# Run by ctest (via `cmake -P`) for each compile_check(... TRUE) test.
#
#   cmake -DBUILD_DIR=<dir> -DTARGET=<target> -DSOURCE=<file.cpp> -P compile_fail_check.cmake
#
# Builds TARGET and passes only if the build fails *and* every expected error
# declared in SOURCE appears on an `error:` line of the compiler output.
# Expected errors are declared with one or more
#
#   // EXPECT-ERROR: <text>
#
# comments anywhere in SOURCE. <text> is matched literally (not as a regex),
# so it can contain any characters. At least one directive is required, so a
# test that fails for an unrelated reason (a broken include, a typo) does not
# pass by accident.

foreach(_var BUILD_DIR TARGET SOURCE)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "compile_fail_check.cmake: ${_var} is not set")
    endif()
endforeach()

# CMake lists are `;`-separated and treat `[`/`]` specially, but expected
# errors and compiler output can contain all three. Swap them for control
# characters before splitting text into lines, and back again afterwards.
string(ASCII 1 _semi)
string(ASCII 2 _lbracket)
string(ASCII 3 _rbracket)

function(_split_lines out_var text)
    string(REPLACE ";" "${_semi}" text "${text}")
    string(REPLACE "[" "${_lbracket}" text "${text}")
    string(REPLACE "]" "${_rbracket}" text "${text}")
    string(REPLACE "\r" "" text "${text}")
    string(REPLACE "\n" ";" text "${text}")
    set(${out_var} "${text}" PARENT_SCOPE)
endfunction()

function(_restore out_var text)
    string(REPLACE "${_semi}" ";" text "${text}")
    string(REPLACE "${_lbracket}" "[" text "${text}")
    string(REPLACE "${_rbracket}" "]" text "${text}")
    set(${out_var} "${text}" PARENT_SCOPE)
endfunction()

# Collect the expected errors (still encoded, like the output lines below).
file(READ "${SOURCE}" _source_text)
_split_lines(_source_lines "${_source_text}")
set(_expected)
foreach(_line IN LISTS _source_lines)
    if(_line MATCHES "^[ \t]*// EXPECT-ERROR:(.*)$")
        string(STRIP "${CMAKE_MATCH_1}" _text)
        if(_text STREQUAL "")
            message(FATAL_ERROR "${SOURCE}: empty `// EXPECT-ERROR:` directive")
        endif()
        list(APPEND _expected "${_text}")
    endif()
endforeach()
list(LENGTH _expected _expected_count)
if(_expected_count EQUAL 0)
    message(FATAL_ERROR
        "${SOURCE}: no `// EXPECT-ERROR: <text>` directive. A compile-fail "
        "test must say which error it expects.")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${TARGET}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _output)

# Drop ANSI color codes in case colored diagnostics are forced on.
string(ASCII 27 _esc)
string(REGEX REPLACE "${_esc}\\[[0-9;]*[A-Za-z]" "" _output "${_output}")

if(_rc EQUAL 0)
    message(FATAL_ERROR
        "${SOURCE}: expected a compile error, but it compiled.\n${_output}")
endif()

# Only search `error:` lines: the compiler also echoes source snippets, and
# the expected text usually appears in the source as a static_assert message.
_split_lines(_output_lines "${_output}")
set(_missing)
foreach(_text IN LISTS _expected)
    set(_found FALSE)
    foreach(_line IN LISTS _output_lines)
        if(_line MATCHES "error:")
            string(FIND "${_line}" "${_text}" _pos)
            if(NOT _pos EQUAL -1)
                set(_found TRUE)
                break()
            endif()
        endif()
    endforeach()
    if(NOT _found)
        _restore(_text "${_text}")
        string(APPEND _missing "\n  ${_text}")
    endif()
endforeach()

if(NOT "${_missing}" STREQUAL "")
    message(FATAL_ERROR
        "${SOURCE}: compilation failed, but not with the expected error(s):"
        "${_missing}\n\nCompiler output:\n${_output}")
endif()
