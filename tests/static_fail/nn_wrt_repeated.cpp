#include <nn/nn.hpp>

inline double product(double a, double b) { return a * b; }

int main() {
  // this should fail to compile: parameter 0 is listed twice
  [[maybe_unused]] auto r = nn::ad::grad<^^product, 0, 0>(2.0, 3.0);
  return 0;
}
