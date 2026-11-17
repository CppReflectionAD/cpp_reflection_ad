#include <nn/nn.hpp>

inline double product(double a, double b) { return a * b; }

int main() {
  // this should fail to compile: product has no parameter 2
  [[maybe_unused]] auto r = nn::ad::value_and_grad<^^product, 2>(2.0, 3.0);
  return 0;
}
