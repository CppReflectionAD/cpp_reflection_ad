# extract_error_output.cmake
# Builds a target that is expected to fail and saves its compiler errors to
# .fail.txt for rebasing
# Supports multiple compilers: appends/updates current compiler's section
#
# Usage (called by rebase-* target):
#   cmake -DBINARY_DIR=... -DTARGET_NAME=... -DEXPECTED_FILE=... -DCOMPILER=... -DTEST_SIMPLE_DIR=... -P extract_error_output.cmake

if(NOT DEFINED TEST_SIMPLE_DIR)
    message(FATAL_ERROR "TEST_SIMPLE_DIR must be provided")
endif()

# Include shared extraction functions
include("${TEST_SIMPLE_DIR}/shared_extract.cmake")

# Compute CMAKE_SOURCE_DIR from TEST_SIMPLE_DIR (its parent)
get_filename_component(CMAKE_SOURCE_DIR "${TEST_SIMPLE_DIR}" DIRECTORY)

if(NOT DEFINED BINARY_DIR OR NOT DEFINED TARGET_NAME OR NOT DEFINED EXPECTED_FILE OR NOT DEFINED COMPILER)
    message(FATAL_ERROR "Missing required parameters: BINARY_DIR, TARGET_NAME, EXPECTED_FILE, COMPILER")
endif()

# Build the target and capture its output, the same way the test does
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${BINARY_DIR}" --target "${TARGET_NAME}"
    OUTPUT_VARIABLE BUILD_OUTPUT
    ERROR_VARIABLE BUILD_ERROR
    RESULT_VARIABLE BUILD_RESULT
)
set(BUILD_OUTPUT "${BUILD_ERROR}${BUILD_OUTPUT}")

if(BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "${TARGET_NAME} compiled successfully but is expected to fail; not writing a baseline")
endif()

# Extract error lines
extract_errors("${BUILD_OUTPUT}" EXTRACTED)

# Refuse to write an empty section: the build failed, but not with a compiler error
if(EXTRACTED STREQUAL "")
    message(FATAL_ERROR "${TARGET_NAME} failed to build, but no compiler errors were found in its output:\n${BUILD_OUTPUT}")
endif()

# Normalize paths: strip CMAKE_SOURCE_DIR to make baselines portable.
# Plain REPLACE, not REGEX REPLACE: the path is literal text, and characters
# like + ( [ . in it would otherwise be read as regex syntax.
string(REPLACE "${CMAKE_SOURCE_DIR}/" "" EXTRACTED "${EXTRACTED}")

# Determine compiler header using shared helper
get_compiler_header("${COMPILER}" COMPILER_HEADER)

set(NEW_SECTION "${COMPILER_HEADER}\n${EXTRACTED}")

# Split the existing file into: sections before ours, ours, sections after ours.
# If our section doesn't exist yet, everything goes in 'before' and ours is
# appended.
set(before_sections "")
set(after_sections "")
if(EXISTS "${EXPECTED_FILE}")
    file(READ "${EXPECTED_FILE}" EXISTING_OUTPUT)

    string(FIND "${EXISTING_OUTPUT}" "${COMPILER_HEADER}" header_pos)
    if(header_pos EQUAL -1)
        set(before_sections "${EXISTING_OUTPUT}")
    else()
        string(SUBSTRING "${EXISTING_OUTPUT}" 0 ${header_pos} before_sections)

        # Our section runs until the next header (\n===) or the end of the file
        string(LENGTH "${COMPILER_HEADER}" header_len)
        math(EXPR section_start "${header_pos} + ${header_len}")
        string(SUBSTRING "${EXISTING_OUTPUT}" ${section_start} -1 after_header)
        string(FIND "${after_header}" "\n===" next_section_pos)
        if(NOT next_section_pos EQUAL -1)
            string(SUBSTRING "${after_header}" ${next_section_pos} -1 after_sections)
        endif()
    endif()
endif()

# Write one layout whichever path was taken, so repeated rebases don't churn
# whitespace: no leading blank line, exactly one blank line between sections,
# and a single newline at the end of the file.
set(REBASED_OUTPUT "")
foreach(part_var IN ITEMS before_sections NEW_SECTION after_sections)
    string(REGEX REPLACE "^\n+" "" part "${${part_var}}")
    string(REGEX REPLACE "\n+$" "" part "${part}")
    if(part STREQUAL "")
        continue()
    endif()
    if(NOT REBASED_OUTPUT STREQUAL "")
        string(APPEND REBASED_OUTPUT "\n\n")
    endif()
    string(APPEND REBASED_OUTPUT "${part}")
endforeach()
string(APPEND REBASED_OUTPUT "\n")

# Write to expected output file
file(WRITE "${EXPECTED_FILE}" "${REBASED_OUTPUT}")

message(STATUS "Wrote baseline to: ${EXPECTED_FILE}")
message(STATUS "Compiler: ${COMPILER}")
message(STATUS "Next: review the file, then run:")
message(STATUS "  git add $(dirname ${EXPECTED_FILE})/*.fail.txt")

