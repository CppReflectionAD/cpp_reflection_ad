function(compile_check group filelist fail)
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
        set(target_name "${group}.static.${target}")
        set(test_name "${target}")
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
        add_test(NAME ${test_name} COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${target_name})
        if (fail)
            set_tests_properties(${test_name} PROPERTIES WILL_FAIL true)
        endif()
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
