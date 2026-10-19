# The install_consumer test (#94 B5): install this build into a scratch prefix,
# check the installed package names none of this machine's paths, then
# configure, build and run tests/install/consumer/ against it with the same
# compiler and generator. The consumer asks for this build's VERSION; asking
# for the next minor version must fail.
#
#   cmake -DBUILD_DIR=... -DSCRATCH_DIR=... -DCONSUMER_DIR=... -DCXX=...
#         -DGENERATOR=... -DMAKE_PROGRAM=... -DBUILD_TYPE=... -DVERSION=...
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

set(_configure "${CMAKE_COMMAND}" -S "${CONSUMER_DIR}"
    -G "${GENERATOR}"
    "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}"
    "-DCMAKE_CXX_COMPILER=${CXX}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_PREFIX_PATH=${_prefix}")

# Before 1.0 only the same minor version is compatible.
string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\1;\\2" _major_minor "${VERSION}")
list(GET _major_minor 0 _major)
list(GET _major_minor 1 _minor)
math(EXPR _next_minor "${_minor} + 1")
execute_process(
    COMMAND ${_configure} -B "${SCRATCH_DIR}/wrong_version"
        "-DREFLECTION_AD_VERSION=${_major}.${_next_minor}"
    RESULT_VARIABLE _result OUTPUT_QUIET ERROR_VARIABLE _error)
string(FIND "${_error}" "compatible with requested version" _at)
if(_result EQUAL 0 OR _at EQUAL -1)
    message(FATAL_ERROR "find_package(reflection_ad ${_major}.${_next_minor}) "
                        "should reject version ${VERSION}:\n${_error}")
endif()

run(${_configure} -B "${_build}" "-DREFLECTION_AD_VERSION=${VERSION}")
run("${CMAKE_COMMAND}" --build "${_build}")
run("${_build}/consumer")
