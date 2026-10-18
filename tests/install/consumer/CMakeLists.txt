# A downstream project that uses an installed reflection_ad, the way a user
# would: find_package and one target, with no flags of its own.
# run_consumer_test.cmake installs the package and builds this against it.
cmake_minimum_required(VERSION 3.25)
project(reflection_ad_consumer CXX)

find_package(reflection_ad CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE reflection_ad::ad)
