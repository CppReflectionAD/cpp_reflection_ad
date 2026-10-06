#include "../is_invertible.hpp"

inline double fn_square(double x) { return x * x; }
inline double fn_square_plus(double x, double y) { return x * x + y; }

static_assert(!ad::is_invertible<^^fn_square>());
static_assert(!ad::is_invertible_wrt<^^fn_square_plus, 0>());

int main() {
  // Two independent failures with different messages, so the snapshot only
  // matches if every error is extracted, not just the first one.
  [[maybe_unused]] auto inv = ad::inverse<^^fn_square>{};
  [[maybe_unused]] auto inv_wrt = ad::inverse_wrt<^^fn_square_plus, 0>{};
  return 0;
}
