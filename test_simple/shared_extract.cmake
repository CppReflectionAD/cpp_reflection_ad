# shared_extract.cmake
# Common functions for extracting and normalizing compiler errors
# Included by: validate_compile_output.cmake, extract_error_output.cmake

# Get the section header for a compiler ID
# Maps compiler IDs to their section headers (e.g. "Clang" → "=== Clang ===")
function(get_compiler_header compiler_id output_var)
    if(compiler_id STREQUAL "Clang")
        set(header "=== Clang ===")
    elseif(compiler_id STREQUAL "GNU")
        set(header "=== GCC ===")
    else()
        set(header "=== ${compiler_id} ===")
    endif()
    set(${output_var} "${header}" PARENT_SCOPE)
endfunction()

# Extract every error block from compiler output, joined by newlines.
# A block is a line containing "error:" plus the continuation lines after it
# (lines starting with a space or tab: source excerpt, caret, GCC notes).
# Blocks are consumed one at a time with REGEX MATCH rather than collected
# with REGEX MATCHALL: MATCHALL returns a ;-separated list, which splits
# diagnostics at any ';' they contain (e.g. "expected ';'").
function(extract_errors input output_var)
    set(block_regex "[^\n]*error:[^\n]*(\n[ \t][^\n]*)*")
    set(rest "${input}")
    set(extracted "")
    # Loop on the match itself, not while(TRUE): these functions run under
    # cmake -P with no policies set, where TRUE is treated as a variable name.
    string(REGEX MATCH "${block_regex}" block "${rest}")
    while(NOT block STREQUAL "")
        if(extracted STREQUAL "")
            set(extracted "${block}")
        else()
            string(APPEND extracted "\n${block}")
        endif()
        # Continue after this block. The match is the leftmost one, so FIND
        # locates the same occurrence.
        string(FIND "${rest}" "${block}" pos)
        string(LENGTH "${block}" len)
        math(EXPR pos "${pos} + ${len}")
        string(SUBSTRING "${rest}" ${pos} -1 rest)
        string(REGEX MATCH "${block_regex}" block "${rest}")
    endwhile()

    set(${output_var} "${extracted}" PARENT_SCOPE)
endfunction()

# Normalize error messages for robust comparison
# Strips paths, line numbers, and formatting noise
# Keeps semantic content of error messages
# Uses string matching instead of lists to avoid semicolon delimiter issues
function(normalize_error_output input output_var)
    extract_errors("${input}" all_errors)

    # Apply all transformations to the entire block
    # Remove file paths (keep only filename). In CMake regex '.' also matches
    # newlines, so stay within one line: otherwise the match runs to the last
    # path in the output and deletes every error before it.
    string(REGEX REPLACE "[^\n]*/([^/\n]+):[0-9]+:[0-9]+:" "\\1: error:" cleaned "${all_errors}")
    # Collapse multiple spaces
    string(REGEX REPLACE "[ \t]+" " " cleaned "${cleaned}")
    # Strip trailing whitespace
    string(REGEX REPLACE "[ \t]+\n" "\n" cleaned "${cleaned}")
    # Normalize line number references (in error context)
    string(REGEX REPLACE " [0-9]+ \\|" " N |" cleaned "${cleaned}")
    string(REGEX REPLACE "\\| +\\^" "| ^" cleaned "${cleaned}")

    string(STRIP cleaned "${cleaned}")
    set(${output_var} "${cleaned}" PARENT_SCOPE)
endfunction()

# Extract a specific compiler's section from multi-compiler baseline
# All baselines must use multi-section format with headers like "=== Clang ==="
# If the section doesn't exist, fails with instruction to run rebase
function(extract_compiler_section baseline compiler_id output_var)
    get_compiler_header("${compiler_id}" header)

    # Check if this header exists in baseline
    if(baseline MATCHES "${header}")
        # Multi-compiler format: extract this compiler's section
        # Strategy: find header, then extract until next section or end

        # Find the position after the header
        string(FIND "${baseline}" "${header}" header_pos)
        if(header_pos EQUAL -1)
            # Shouldn't happen due to MATCHES check above
            set(${output_var} "" PARENT_SCOPE)
            return()
        endif()

        # Move past the header line (header + \n)
        string(LENGTH "${header}" header_len)
        math(EXPR start_pos "${header_pos} + ${header_len} + 1")

        # Find the next section header (looks for \n===)
        string(SUBSTRING "${baseline}" ${start_pos} -1 rest)
        string(FIND "${rest}" "\n===" next_section_pos)

        if(next_section_pos EQUAL -1)
            # No next section, take everything until end
            set(content "${rest}")
        else()
            # Extract until next section
            string(SUBSTRING "${rest}" 0 ${next_section_pos} content)
        endif()

        # Remove leading/trailing newlines
        string(REGEX REPLACE "^\n+" "" content "${content}")
        string(REGEX REPLACE "\n+$" "" content "${content}")

        set(${output_var} "${content}" PARENT_SCOPE)
    else()
        # Section not found. Check if baseline has any headers (multi-section format)
        if(baseline MATCHES "===")
            # Baseline has headers but not ours; this is an error
            # Pass empty output to trigger validation failure with clear message
            set(${output_var} "" PARENT_SCOPE)
        else()
            # No headers in baseline; this shouldn't happen with new code
            # but treating as single-section for backwards compat
            set(${output_var} "${baseline}" PARENT_SCOPE)
        endif()
    endif()

endfunction()
