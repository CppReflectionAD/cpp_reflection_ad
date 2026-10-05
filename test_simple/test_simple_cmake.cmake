function(compile_check group filelist)
    # Snapshot-based compilation failure testing
    # Validates that code fails to compile for the correct reason
    # Get the test_simple directory path
    # This function is included from tests/CMakeLists.txt with an absolute include path
    set(test_simple_dir "${CMAKE_SOURCE_DIR}/test_simple")

    foreach(testfile IN LISTS filelist)
        # Set up test file paths and executable
        if(IS_ABSOLUTE "${testfile}")
            set(_source "${testfile}")
            file(RELATIVE_PATH target "${CMAKE_CURRENT_SOURCE_DIR}" "${testfile}")
        else()
            set(_source "${CMAKE_CURRENT_SOURCE_DIR}/${testfile}")
            set(target "${testfile}")
        endif()

        # Transform target path to test naming format
        string(REPLACE .cpp "" target "${target}")
        string(REPLACE / "." target "${target}")
        set(target_name "${group}.static.${target}")
        set(test_name "${group}.${target}")

        # Create the executable
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

        # Keep directory structure for the .fail.txt file location
        set(source_path "${target}")
        string(REPLACE .cpp "" source_path ${source_path})
        set(expected_output_file "${CMAKE_CURRENT_SOURCE_DIR}/${source_path}.fail.txt")

        # Create a test that validates compilation output against snapshot
        add_test(
            NAME ${test_name}
            COMMAND ${CMAKE_COMMAND}
                -DBINARY_DIR=${CMAKE_BINARY_DIR}
                -DTARGET_NAME=${target_name}
                -DEXPECTED_OUTPUT_FILE=${expected_output_file}
                -DCXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
                -DTEST_SIMPLE_DIR=${test_simple_dir}
                -P ${test_simple_dir}/validate_compile_output.cmake
        )

        # Create rebase target for this test
        set(rebase_target "rebase-${test_name}")
        add_custom_target(${rebase_target}
            COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${target_name} 2>&1
                | tee /tmp/${target_name}_output.txt || true
            COMMAND ${CMAKE_COMMAND}
                -DOUTPUT_FILE=/tmp/${target_name}_output.txt
                -DEXPECTED_FILE=${expected_output_file}
                -DCOMPILER=${CMAKE_CXX_COMPILER_ID}
                -DTEST_SIMPLE_DIR=${test_simple_dir}
                -P ${test_simple_dir}/extract_error_output.cmake
            COMMAND ${CMAKE_COMMAND} -E echo "Rebased ${expected_output_file}"
            COMMENT "Rebasing expected output for ${rebase_target}"
            VERBATIM
        )

        # Track for global rebase target
        list(APPEND ALL_REBASE_TARGETS ${rebase_target})
        set(ALL_REBASE_TARGETS "${ALL_REBASE_TARGETS}" PARENT_SCOPE)
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

