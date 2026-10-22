#ifndef OUTPUT_OUTPUT_OF_HPP
#define OUTPUT_OUTPUT_OF_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// The lines `f` prints to std::cout.
template <class F> std::vector<std::string> output_of(F f) {
  std::ostringstream captured;
  {
    // Gives std::cout its buffer back, also if f() throws, before `captured`
    // is destroyed.
    struct restore {
      std::streambuf *previous;
      ~restore() { std::cout.rdbuf(previous); }
    } guard{std::cout.rdbuf(captured.rdbuf())};
    f();
  }

  std::vector<std::string> lines;
  std::istringstream in(captured.str());
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);
  return lines;
}

#endif
