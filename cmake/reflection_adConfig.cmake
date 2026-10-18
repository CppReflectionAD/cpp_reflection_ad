# Package config for find_package(reflection_ad), installed with the headers.
#
#     project(my_project CXX)
#     find_package(reflection_ad CONFIG REQUIRED)
#     target_link_libraries(my_target PRIVATE reflection_ad::ad)
#
# reflection_ad::ad is the AD engine, and reflection_ad::cx_std the constexpr
# maths it uses. Both carry reflection_ad::flags, the flag profile, which is
# worked out here for the consumer's compiler (one of the reflection forks)
# rather than stored: it holds absolute paths into that compiler's install.

if(CMAKE_VERSION VERSION_LESS 3.25)
    set(reflection_ad_FOUND FALSE)
    set(reflection_ad_NOT_FOUND_MESSAGE "reflection_ad needs CMake 3.25 or later.")
    return()
endif()
if(NOT CMAKE_CXX_COMPILER_ID)
    set(reflection_ad_FOUND FALSE)
    set(reflection_ad_NOT_FOUND_MESSAGE
        "Enable C++ (e.g. project(<name> CXX)) before find_package(reflection_ad).")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/reflection_adTargets.cmake")

if(NOT TARGET reflection_ad::flags)
    include("${CMAKE_CURRENT_LIST_DIR}/reflection_ad_flags.cmake")
    add_library(reflection_ad::flags INTERFACE IMPORTED)
    reflection_ad_set_flags(reflection_ad::flags)
    set_property(TARGET reflection_ad::cx_std APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES reflection_ad::flags)
endif()
