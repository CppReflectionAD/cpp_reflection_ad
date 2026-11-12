#include <nn/nn.hpp>

// An adjoint has its parameter's type, so differentiating with respect to an
// integer parameter would truncate the gradient. value_and_grad rejects it.
inline double scaled(double x, unsigned char c) { return -c * x; }

int main() {
  // this should fail to compile: parameter 1 is an unsigned char
  [[maybe_unused]] auto r =
      nn::ad::value_and_grad<^^scaled, 1>(2.0, static_cast<unsigned char>(1));
  return 0;
}
