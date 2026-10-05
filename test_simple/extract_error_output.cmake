# extract_error_output.cmake
# Extracts compiler error output and saves it to .fail.txt for rebasing
# Supports multiple compilers: appends/updates current compiler's section
#
# Usage (called by rebase-* target):
#   cmake -DOUTPUT_FILE=... -DEXPECTED_FILE=... -DCOMPILER=... -P extract_error_output.cmake

if(NOT DEFINED OUTPUT_FILE OR NOT DEFINED EXPECTED_FILE OR NOT DEFINED COMPILER)
    message(FATAL_ERROR "Missing required parameters: OUTPUT_FILE, EXPECTED_FILE, COMPILER")
endif()

# Read the build output
if(NOT EXISTS "${OUTPUT_FILE}")
    message(FATAL_ERROR "Output file not found: ${OUTPUT_FILE}")
endif()

file(READ "${OUTPUT_FILE}" BUILD_OUTPUT)

# Extract only error and warning lines with continuations (preserve compiler output format)
# Error/warning blocks: error/warning line followed by continuation lines (starting with space or tab)
string(REGEX MATCHALL "[^\n]*error:[^\n]*(\n[ \t][^\n]*)*" error_lines "${BUILD_OUTPUT}")
string(REGEX MATCHALL "[^\n]*warning:[^\n]*(\n[ \t][^\n]*)*" warning_lines "${BUILD_OUTPUT}")

# Combine errors and warnings, preserving newlines
set(EXTRACTED "")
foreach(line IN LISTS error_lines)
    if(EXTRACTED)
        string(APPEND EXTRACTED "\n${line}")
    else()
        set(EXTRACTED "${line}")
    endif()
endforeach()

foreach(line IN LISTS warning_lines)
    if(EXTRACTED)
        string(APPEND EXTRACTED "\n${line}")
    else()
        set(EXTRACTED "${line}")
    endif()
endforeach()

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

    # Extract sections for other compilers
    if(COMPILER_HEADER STREQUAL "=== Clang ===")
        # Keep GCC section if present
        string(REGEX MATCH "=== GCC ===\n[^=]*" gcc_section "${EXISTING_OUTPUT}")
        if(gcc_section)
            set(REBASED_OUTPUT "${gcc_section}\n\n")
        endif()
    elseif(COMPILER_HEADER STREQUAL "=== GCC ===")
        # Keep Clang section if present
        string(REGEX MATCH "=== Clang ===\n[^=]*" clang_section "${EXISTING_OUTPUT}")
        if(clang_section)
            set(REBASED_OUTPUT "${clang_section}\n\n")
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

