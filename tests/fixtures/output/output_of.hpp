#ifndef OUTPUT_OUTPUT_OF_HPP
#define OUTPUT_OUTPUT_OF_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// The lines `f` prints to std::cout.
template <class F> std::vector<std::string> output_of(F f) {
  std::ostringstream captured;
  std::streambuf *previous = std::cout.rdbuf(captured.rdbuf());
  f();
  std::cout.rdbuf(previous);

  std::vector<std::string> lines;
  std::istringstream in(captured.str());
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);
  return lines;
}

#endif
