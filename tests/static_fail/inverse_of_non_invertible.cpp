#include "../is_invertible.hpp"

inline double fn_square(double x) { return x * x; }

static_assert(!ad::is_invertible<^^fn_square>());

int main() {
  // this should fail to compile, because fn_square is not invertible
  [[maybe_unused]] auto inv = ad::inverse<^^fn_square>{};
  return 0;
}
