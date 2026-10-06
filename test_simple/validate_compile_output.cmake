# validate_compile_output.cmake
# Validates that a test's compilation output matches the expected snapshot
# Supports multiple compilers: extracts and compares only current compiler's section
#
# Usage (called by CMake test):
#   cmake -DBINARY_DIR=... -DTARGET_NAME=... -DEXPECTED_OUTPUT_FILE=... -DCXX_COMPILER_ID=... -DTEST_SIMPLE_DIR=... -P validate_compile_output.cmake

if(NOT DEFINED TEST_SIMPLE_DIR)
    message(FATAL_ERROR "TEST_SIMPLE_DIR must be provided")
endif()

# Include shared extraction functions
include("${TEST_SIMPLE_DIR}/shared_extract.cmake")

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

# Check that compilation failed
if(BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "${TARGET_NAME} compiled successfully but is expected to fail")
endif()

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

# Extract only the section for the current compiler
extract_compiler_section("${EXPECTED_OUTPUT}" "${CXX_COMPILER_ID}" EXPECTED_OUTPUT)

# Extract error lines from actual output
extract_errors_and_warnings("${ACTUAL_OUTPUT}" extracted_actual)

# Normalize both for comparison
normalize_error_output("${extracted_actual}" normalized_actual)
normalize_error_output("${EXPECTED_OUTPUT}" normalized_expected)

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

