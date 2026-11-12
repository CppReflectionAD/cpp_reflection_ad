#pragma once

#include <test_simple_include.hpp>

void stringCompare(std::string const& val1, std::string const& val2) {
  size_t min_size = std::min(val1.size(), val2.size());
  for (size_t i = 0; i < min_size; i += 1) {
    if (val1[i] != val2[i]) {
      std::cout << std::format("Difference at: {} ('{}' <-> '{}')", i, val1[i], val2[i]) << std::endl;
    }
  }
  EXPECT_EQUAL_ARRAY(val1, val2);
}