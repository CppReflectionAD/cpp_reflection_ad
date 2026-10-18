# The install_consumer test (#94 B5): install this build into a scratch prefix,
# check the installed package names none of this machine's paths, then
# configure, build and run tests/install/consumer/ against it with the same
# compiler and generator.
#
#   cmake -DBUILD_DIR=... -DSCRATCH_DIR=... -DCONSUMER_DIR=... -DCXX=...
#         -DGENERATOR=... -DMAKE_PROGRAM=... -DBUILD_TYPE=...
#         -P run_consumer_test.cmake
cmake_minimum_required(VERSION 3.25)

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        list(JOIN ARGN " " _command)
        message(FATAL_ERROR "Failed (${_result}): ${_command}")
    endif()
endfunction()

set(_prefix "${SCRATCH_DIR}/prefix")
set(_build "${SCRATCH_DIR}/build")
file(REMOVE_RECURSE "${SCRATCH_DIR}")

run("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${_prefix}")

# The flags hold absolute paths into the compiler, so the package must work
# them out for the consumer rather than store this build's.
get_filename_component(_compiler_root "${CXX}" DIRECTORY)
get_filename_component(_compiler_root "${_compiler_root}" DIRECTORY)
file(GLOB_RECURSE _package_files "${_prefix}/*.cmake")
foreach(_file IN LISTS _package_files)
    file(READ "${_file}" _text)
    foreach(_path IN ITEMS "${_compiler_root}" "${BUILD_DIR}")
        string(FIND "${_text}" "${_path}" _at)
        if(NOT _at EQUAL -1)
            message(FATAL_ERROR "${_file} names ${_path}")
        endif()
    endforeach()
endforeach()

run("${CMAKE_COMMAND}" -S "${CONSUMER_DIR}" -B "${_build}"
    -G "${GENERATOR}"
    "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}"
    "-DCMAKE_CXX_COMPILER=${CXX}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_PREFIX_PATH=${_prefix}")
run("${CMAKE_COMMAND}" --build "${_build}")
run("${_build}/consumer")
