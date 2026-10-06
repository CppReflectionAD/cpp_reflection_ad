# shared_extract.cmake
# Common functions for extracting and normalizing compiler errors
# Included by: validate_compile_output.cmake, extract_error_output.cmake

# Extract error and warning lines from compiler output
# Captures multi-line errors (continuation lines starting with space/tab)
# Uses string matching instead of lists to avoid semicolon delimiter issues
function(extract_errors_and_warnings input output_var)
    # Extract all errors and warnings as single strings (not lists)
    string(REGEX MATCH "([^\n]*error:[^\n]*(\n[ \t][^\n]*)*)+" all_errors "${input}")
    string(REGEX MATCH "([^\n]*warning:[^\n]*(\n[ \t][^\n]*)*)+" all_warnings "${input}")

    set(extracted "")
    if(all_errors)
        set(extracted "${all_errors}")
    endif()
    if(all_warnings)
        if(extracted)
            string(APPEND extracted "\n${all_warnings}")
        else()
            set(extracted "${all_warnings}")
        endif()
    endif()

    set(${output_var} "${extracted}" PARENT_SCOPE)
endfunction()

# Normalize error messages for robust comparison
# Strips paths, line numbers, and formatting noise
# Keeps semantic content of error messages
# Uses string matching instead of lists to avoid semicolon delimiter issues
function(normalize_error_output input output_var)
    # Extract all errors as a single string (not a list)
    string(REGEX MATCH "([^\n]*error:[^\n]*(\n[ \t][^\n]*)*)+" all_errors "${input}")

    # Apply all transformations to the entire block
    # Remove file paths (keep only filename)
    string(REGEX REPLACE ".*/([^/]+):[0-9]+:[0-9]+:" "\\1: error:" cleaned "${all_errors}")
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
# Handles both old (single section) and new (multi-section) formats
function(extract_compiler_section baseline compiler_id output_var)
    # Determine the section header for this compiler
    if(compiler_id STREQUAL "Clang")
        set(header "=== Clang ===")
    elseif(compiler_id STREQUAL "GNU")
        set(header "=== GCC ===")
    else()
        set(header "=== ${compiler_id} ===")
    endif()

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
    else()
        # Old single-compiler format: use entire baseline
        # Strip any headers that might be present for backwards compat
        string(REGEX REPLACE "=== (Clang|GCC|GNU) ===\n" "" content "${baseline}")
        string(STRIP content "${content}")
    endif()

    set(${output_var} "${content}" PARENT_SCOPE)
endfunction()
