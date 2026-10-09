cmake_minimum_required(VERSION 3.25)
project(reflection_ad CXX)

set(REFLECTION_AD_SANITIZE_FLAGS
    -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # These UBSan checks break GCC's std::meta expression reflection; see
  # tests/gcc_only/ubsan_reflection_repro.cpp.
  list(APPEND REFLECTION_AD_SANITIZE_FLAGS
       -fno-sanitize=shift,integer-divide-by-zero,signed-integer-overflow,float-divide-by-zero,null,alignment,vptr)
endif()
add_compile_options("$<$<CONFIG:Debug>:${REFLECTION_AD_SANITIZE_FLAGS}>")
add_link_options("$<$<CONFIG:Debug>:${REFLECTION_AD_SANITIZE_FLAGS}>")

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
