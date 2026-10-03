# A test declares its extra flags (`// TEST-FLAGS:`) and, if it must fail to
# compile, the error(s) it must fail with (`// EXPECT-ERROR:`) in comments at
# the top of the file. test_simple/source_directives.py reads them, for CTest
# here as for run_tests.py, when a test or benchmark is built, so building
# them needs Python 3.7 or newer (configure with
# -DREFLECTION_AD_BUILD_TESTING=OFF to skip them).
find_package(Python3 3.7 REQUIRED COMPONENTS Interpreter)
set(_test_simple_dir "${CMAKE_CURRENT_LIST_DIR}")
# Every Python script runs with -B, so that it writes no bytecode into the
# source tree.
set(_test_python "${Python3_EXECUTABLE}" -B)
# _compiler is the build's compiler, as test directives name it
# (`// TEST-FLAGS-CLANG:`), which the including file (the top-level
# CMakeLists.txt) sets.
if(NOT _compiler MATCHES "^(clang|gcc)$")
    message(FATAL_ERROR "Set _compiler to clang or gcc before including "
                        "test_simple_cmake.cmake")
endif()

# The user's compile flags: CMAKE_CXX_FLAGS and those of the build type.
# CMake would put them first, where a test's TEST-FLAGS override them
# (-DCMAKE_CXX_FLAGS=-O0 wouldn't undo a benchmark's -O2), so they are
# cleared for the targets of this directory and those below it, and
# test_flags() gives them to each test after its TEST-FLAGS instead, as
# run_tests.py does --extra-cxxflag. They reach the build through a file,
# written only when they change, so that a test is rebuilt when they do,
# whatever the generator. A multi-config generator builds every config from
# that one file (OBJECT_DEPENDS can't name a file per config), so there the
# build type's flags stay where CMake puts them.
set(_user_cxxflags "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "")
get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_multi_config)
    message(WARNING "With a multi-config generator, a test's TEST-FLAGS come "
        "after CMAKE_CXX_FLAGS_<CONFIG>, so they override those flags; only "
        "CMAKE_CXX_FLAGS come after them.")
else()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)
    string(APPEND _user_cxxflags " ${CMAKE_CXX_FLAGS_${_build_type}}")
    set(CMAKE_CXX_FLAGS_${_build_type} "")
endif()
set(_user_cxxflags_file "${CMAKE_CURRENT_BINARY_DIR}/test_flags/user_cxxflags")
file(GENERATE OUTPUT "${_user_cxxflags_file}" CONTENT "${_user_cxxflags}\n")

# test_flags(<target> <source> <must_fail>)
#
# Builds <source>, in <target>, with the flags compile_fail_check.py's
# compile_flags gives it, as run_tests.py does: the TEST-FLAGS its directives
# give it for the build's compiler, then the user's flags (above), then, if
# <must_fail>, the diagnostic flags compile_fail_check.py parses.
# test_directives_cmake.py writes them to a response file, which the
# compiler reads after the target's own options, before <source> is compiled
# and again whenever it or the user's flags change; so editing a test doesn't
# re-run CMake. An executable <target> is linked with them too, as
# run_tests.py compiles and links in one command. The script rewrites the
# response file only when the flags change, and the command's output is a
# stamp file beside it, so a change that leaves them alone recompiles
# nothing, and the command runs once per change, under Ninja and Make alike.
# If the directives of <source> are invalid, building <target> fails with
# the reason.
#
# The flags are properties of <source> (as a target's own options would come
# before its -std and reflection flags, and a target has no OBJECT_DEPENDS),
# which every target of the directory that builds it shares; so a source may
# be built by one such target only.
function(test_flags target source must_fail)
    get_source_file_property(_owner "${source}" TEST_FLAGS_TARGET)
    if(_owner)
        message(FATAL_ERROR "${source} is a test of both ${_owner} and "
            "${target}; test_flags() gives a source the flags of one target")
    endif()
    set_source_files_properties("${source}" PROPERTIES TEST_FLAGS_TARGET ${target})
    set(_rsp "${CMAKE_CURRENT_BINARY_DIR}/test_flags/${target}.rsp")
    set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/test_flags/${target}.stamp")
    if(must_fail)
        set(_must_fail --must-fail)
    endif()
    add_custom_command(OUTPUT "${_stamp}"
        BYPRODUCTS "${_rsp}"
        COMMAND ${_test_python} "${_test_simple_dir}/test_directives_cmake.py"
            --compiler ${_compiler} --source "${source}"
            --user-flags-file "${_user_cxxflags_file}"
            --output "${_rsp}" ${_must_fail}
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${source}" "${_user_cxxflags_file}"
            "${_test_simple_dir}/test_directives_cmake.py"
            "${_test_simple_dir}/source_directives.py"
            "${_test_simple_dir}/compile_fail_check.py"
        COMMENT "Reading the test directives of ${source}"
        VERBATIM)
    # The stamp is a source of <target>, so that its build runs the command;
    # the response file is a dependency of the object, so that it is rebuilt
    # when the flags change.
    target_sources(${target} PRIVATE "${_stamp}")
    set_property(SOURCE "${source}" APPEND PROPERTY OBJECT_DEPENDS "${_rsp}")
    set_property(SOURCE "${source}" APPEND PROPERTY COMPILE_OPTIONS "@${_rsp}")
    get_target_property(_type ${target} TYPE)
    if(_type STREQUAL "EXECUTABLE")
        target_link_options(${target} PRIVATE "@${_rsp}")
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${_rsp}")
    endif()
endfunction()

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
        # Compiled, never linked, so that only the compiler's diagnostics
        # reach compile_fail_check.py.
        add_library(${target_name} OBJECT "${_source}")
        test_flags(${target_name} "${_source}" ${fail})
        set_target_properties(${target_name} PROPERTIES EXCLUDE_FROM_ALL true EXCLUDE_FROM_DEFAULT_BUILD true)
        target_compile_options(${target_name} PRIVATE
            -std=c++2c
            ${_reflect_flags}
        )
        target_include_directories(${target_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${_test_simple_dir}"
        )
        if (fail)
            # Must fail with the error(s) its `// EXPECT-ERROR:` comments name.
            add_test(NAME ${test_name} COMMAND ${_test_python}
                "${_test_simple_dir}/compile_fail_check.py"
                --source "${_source}" --compiler ${_compiler}
                -- ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${target_name})
        else()
            add_test(NAME ${test_name} COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${target_name})
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
        test_flags(${test_name} "${_source}" FALSE)
        target_compile_options(${test_name} PRIVATE
            -std=c++2c
            ${_reflect_flags}
        )
        target_include_directories(${test_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${_test_simple_dir}"
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endfunction()
