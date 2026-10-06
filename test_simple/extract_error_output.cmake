# extract_error_output.cmake
# Extracts compiler error output and saves it to .fail.txt for rebasing
# Supports multiple compilers: appends/updates current compiler's section
#
# Usage (called by rebase-* target):
#   cmake -DOUTPUT_FILE=... -DEXPECTED_FILE=... -DCOMPILER=... -DTEST_SIMPLE_DIR=... -P extract_error_output.cmake

if(NOT DEFINED TEST_SIMPLE_DIR)
    message(FATAL_ERROR "TEST_SIMPLE_DIR must be provided")
endif()

# Include shared extraction functions
include("${TEST_SIMPLE_DIR}/shared_extract.cmake")

# Compute CMAKE_SOURCE_DIR from TEST_SIMPLE_DIR (its parent)
get_filename_component(CMAKE_SOURCE_DIR "${TEST_SIMPLE_DIR}" DIRECTORY)

if(NOT DEFINED OUTPUT_FILE OR NOT DEFINED EXPECTED_FILE OR NOT DEFINED COMPILER)
    message(FATAL_ERROR "Missing required parameters: OUTPUT_FILE, EXPECTED_FILE, COMPILER")
endif()

# Read the build output
if(NOT EXISTS "${OUTPUT_FILE}")
    message(FATAL_ERROR "Output file not found: ${OUTPUT_FILE}")
endif()

file(READ "${OUTPUT_FILE}" BUILD_OUTPUT)

# Extract error and warning lines
extract_errors_and_warnings("${BUILD_OUTPUT}" EXTRACTED)

# Refuse to write empty section—if no errors/warnings found, something went wrong
if(NOT EXTRACTED)
    message(FATAL_ERROR "No compilation errors or warnings found in build output. The target may have compiled successfully or output may be in an unexpected format.")
endif()

# Normalize paths: strip CMAKE_SOURCE_DIR to make baselines portable
string(REGEX REPLACE "${CMAKE_SOURCE_DIR}/" "" EXTRACTED "${EXTRACTED}")

# Determine compiler header
if(COMPILER STREQUAL "Clang")
    set(COMPILER_HEADER "=== Clang ===")
elseif(COMPILER STREQUAL "GNU")
    set(COMPILER_HEADER "=== GCC ===")
else()
    set(COMPILER_HEADER "=== ${COMPILER} ===")
endif()

set(NEW_SECTION "${COMPILER_HEADER}\n${EXTRACTED}")

# Read existing file (if present) and preserve other compilers' sections
set(REBASED_OUTPUT "")
if(EXISTS "${EXPECTED_FILE}")
    file(READ "${EXPECTED_FILE}" EXISTING_OUTPUT)

    # Extract sections for other compilers using the helper function
    # For Clang being rebased: preserve GCC section
    if(COMPILER_HEADER STREQUAL "=== Clang ===")
        extract_compiler_section("${EXISTING_OUTPUT}" "GNU" gcc_section)
        if(gcc_section)
            set(REBASED_OUTPUT "=== GCC ===\n${gcc_section}\n\n")
        endif()
    # For GCC being rebased: preserve Clang section
    elseif(COMPILER_HEADER STREQUAL "=== GCC ===")
        extract_compiler_section("${EXISTING_OUTPUT}" "Clang" clang_section)
        if(clang_section)
            set(REBASED_OUTPUT "=== Clang ===\n${clang_section}\n\n")
        endif()
    endif()
endif()

# Append/update current compiler section
string(APPEND REBASED_OUTPUT "${NEW_SECTION}\n")

# Write to expected output file
file(WRITE "${EXPECTED_FILE}" "${REBASED_OUTPUT}")

message(STATUS "Wrote baseline to: ${EXPECTED_FILE}")
message(STATUS "Compiler: ${COMPILER}")
message(STATUS "Next: review the file, then run:")
message(STATUS "  git add $(dirname ${EXPECTED_FILE})/*.fail.txt")

