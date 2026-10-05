# validate_compile_output.cmake
# Validates that a test's compilation output matches the expected snapshot
# Supports multiple compilers: extracts and compares only current compiler's section
#
# Usage (called by CMake test):
#   cmake -DBINARY_DIR=... -DTARGET_NAME=... -DEXPECTED_OUTPUT_FILE=... -DCXX_COMPILER_ID=... -P validate_compile_output.cmake

# Check required parameters
if(NOT DEFINED BINARY_DIR OR NOT DEFINED TARGET_NAME OR NOT DEFINED EXPECTED_OUTPUT_FILE OR NOT DEFINED CXX_COMPILER_ID)
    message(FATAL_ERROR "Missing required parameters")
endif()

# Try to build the target and capture output
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${BINARY_DIR}" --target "${TARGET_NAME}"
    OUTPUT_VARIABLE BUILD_OUTPUT
    ERROR_VARIABLE BUILD_ERROR
    RESULT_VARIABLE BUILD_RESULT
)

# Combine stderr and stdout
set(ACTUAL_OUTPUT "${BUILD_ERROR}${BUILD_OUTPUT}")

# Check if expected output file exists
if(NOT EXISTS "${EXPECTED_OUTPUT_FILE}")
    message(FATAL_ERROR
"Expected output file not found: ${EXPECTED_OUTPUT_FILE}

This is the first time this test is being run. To create the baseline:
  cmake --build . --target rebase-${TARGET_NAME}

Then commit the .fail.txt file to version control.")
endif()

# Read expected output
file(READ "${EXPECTED_OUTPUT_FILE}" EXPECTED_OUTPUT)

# Determine which compiler section to extract
if(CXX_COMPILER_ID STREQUAL "Clang")
    set(COMPILER_HEADER "=== Clang ===")
elseif(CXX_COMPILER_ID STREQUAL "GNU")
    set(COMPILER_HEADER "=== GCC ===")
else()
    # Fallback: use as-is if no compiler-specific section
    set(COMPILER_HEADER "")
endif()

# Extract only the section for the current compiler
if(COMPILER_HEADER)
    # Match from header to next section (=== ...) or end of string
    string(REGEX MATCH "${COMPILER_HEADER}\n([^=]*)" compiler_section "${EXPECTED_OUTPUT}")
    # Extract just the content part (after the header)
    string(REGEX REPLACE "${COMPILER_HEADER}\n" "" EXPECTED_OUTPUT "${compiler_section}")
else()
    # Strip any compiler headers (backward compat: use all content if no specific section found)
    string(REGEX REPLACE "=== (Clang|GCC|GNU) ===\n" "" EXPECTED_OUTPUT "${EXPECTED_OUTPUT}")
endif()

# Helper function to extract error/warning lines (same as extract_error_output.cmake)
function(extract_errors input output_var)
    # Extract error blocks: error line followed by continuation lines (starting with space or tab)
    string(REGEX MATCHALL "[^\n]*error:[^\n]*(\n[ \t][^\n]*)*" error_lines "${input}")
    string(REGEX MATCHALL "[^\n]*warning:[^\n]*(\n[ \t][^\n]*)*" warning_lines "${input}")

    set(extracted "")
    foreach(line IN LISTS error_lines)
        if(extracted)
            string(APPEND extracted "\n${line}")
        else()
            set(extracted "${line}")
        endif()
    endforeach()

    foreach(line IN LISTS warning_lines)
        if(extracted)
            string(APPEND extracted "\n${line}")
        else()
            set(extracted "${line}")
        endif()
    endforeach()

    set(${output_var} "${extracted}" PARENT_SCOPE)
endfunction()

# Extract error lines from actual output
extract_errors("${ACTUAL_OUTPUT}" extracted_actual)

# Normalize both for comparison
function(normalize_output input output_var)
    # Keep only the core error message, strip all paths, line numbers, formatting noise
    # Extract just the error: ... part and continue lines (starting with space/tab)
    # This is robust to compiler version, line number changes, path changes

    # First, extract just error/warning lines (strip build system noise)
    string(REGEX MATCHALL "[^\n]*error:[^\n]*(\n[ \t][^\n]*)*" errors "${input}")

    set(normalized "")
    foreach(error IN LISTS errors)
        # Remove file paths (everything up to the filename)
        string(REGEX REPLACE ".*/([^/]+):[0-9]+:[0-9]+:" "\\1: error:" cleaned "${error}")
        # Collapse multiple spaces
        string(REGEX REPLACE "[ \t]+" " " cleaned "${cleaned}")
        # Strip trailing whitespace
        string(REGEX REPLACE "[ \t]+\n" "\n" cleaned "${cleaned}")
        # Normalize line number references (in error context)
        string(REGEX REPLACE " [0-9]+ \\|" " N |" cleaned "${cleaned}")
        string(REGEX REPLACE "\\| +\\^" "| ^" cleaned "${cleaned}")

        if(normalized)
            string(APPEND normalized "\n${cleaned}")
        else()
            set(normalized "${cleaned}")
        endif()
    endforeach()

    string(STRIP normalized "${normalized}")
    set(${output_var} "${normalized}" PARENT_SCOPE)
endfunction()

normalize_output("${extracted_actual}" normalized_actual)
normalize_output("${EXPECTED_OUTPUT}" normalized_expected)

# Debug output
string(LENGTH "${normalized_expected}" exp_len)
string(LENGTH "${normalized_actual}" act_len)
message(STATUS "Expected bytes: ${exp_len}, Actual bytes: ${act_len}")

# Debug: show what we're comparing
if(NOT normalized_actual STREQUAL normalized_expected)
    message(FATAL_ERROR
"Error output changed for ${TARGET_NAME}!

Expected (${CMAKE_CURRENT_LIST_LINE}):
[${normalized_expected}]

Actual:
[${normalized_actual}]

To update the baseline (rebase):
  cmake --build . --target rebase-${TARGET_NAME}
  git diff ${EXPECTED_OUTPUT_FILE}  # review changes
  git add ${EXPECTED_OUTPUT_FILE}")
endif()

message(STATUS "✓ ${TARGET_NAME}: Compilation failed as expected")

