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

# Read existing file and update current compiler's section in place
set(REBASED_OUTPUT "")
if(EXISTS "${EXPECTED_FILE}")
    file(READ "${EXPECTED_FILE}" EXISTING_OUTPUT)

    # Check if our section already exists in the file
    string(FIND "${EXISTING_OUTPUT}" "${COMPILER_HEADER}" header_pos)
    if(header_pos EQUAL -1)
        # Our section doesn't exist; append it to the end
        set(REBASED_OUTPUT "${EXISTING_OUTPUT}")
        if(REBASED_OUTPUT AND NOT REBASED_OUTPUT MATCHES "\n$")
            string(APPEND REBASED_OUTPUT "\n")
        endif()
        string(APPEND REBASED_OUTPUT "\n${NEW_SECTION}\n")
    else()
        # Our section exists; replace it in place
        # Find the end of our section (start of next "===" or end of file)
        string(LENGTH "${COMPILER_HEADER}" header_len)
        math(EXPR section_start "${header_pos} + ${header_len}")
        string(SUBSTRING "${EXISTING_OUTPUT}" ${section_start} -1 after_header)
        string(FIND "${after_header}" "\n===" next_section_pos)

        # Extract the part before our section
        string(SUBSTRING "${EXISTING_OUTPUT}" 0 ${header_pos} before_section)

        # Extract the part after our section (if any)
        if(next_section_pos EQUAL -1)
            # Our section extends to end of file
            set(after_section "")
        else()
            # There's a next section; extract it
            math(EXPR after_pos "${section_start} + ${next_section_pos}")
            string(SUBSTRING "${EXISTING_OUTPUT}" ${after_pos} -1 after_section)
        endif()

        # Rebuild: before + new section + after
        # Trim trailing newline from before_section to avoid double newlines
        string(REGEX REPLACE "\n+$" "" before_section "${before_section}")
        set(REBASED_OUTPUT "${before_section}\n${NEW_SECTION}")

        # Add after_section if it exists
        if(after_section)
            string(REGEX REPLACE "^\n+" "" after_section "${after_section}")
            string(APPEND REBASED_OUTPUT "\n${after_section}")
        else()
            string(APPEND REBASED_OUTPUT "\n")
        endif()
    endif()
else()
    # File doesn't exist; create with our section
    set(REBASED_OUTPUT "${NEW_SECTION}\n")
endif()

# Write to expected output file
file(WRITE "${EXPECTED_FILE}" "${REBASED_OUTPUT}")

message(STATUS "Wrote baseline to: ${EXPECTED_FILE}")
message(STATUS "Compiler: ${COMPILER}")
message(STATUS "Next: review the file, then run:")
message(STATUS "  git add $(dirname ${EXPECTED_FILE})/*.fail.txt")

