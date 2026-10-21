# I Hate Macros (`ihm`)

A C++ testing framework without macros, built on C++26 reflection (P2996) and
on the expression reflection proposed in P4395. Its spirit:

- as close as possible to GoogleTest (`ihm::expect_equal` for `EXPECT_EQ`, ...)
- header-only
- macro-free
- as simple as possible

For now its main purpose is to find gaps in the reflection implementations:
each feature it needs and a compiler fork lacks shows up as a known gap under
`tests/known_gap/`. Later it might become a more robust tool.

```cpp
#include <ihm/ihm.hpp>

int main() {
  ihm::expect_equal(1 + 1, 2);
  return ihm::result();
}
```

`ihm.hpp` needs only C++26 and works with both forks.
`clang_only/ihm_expressions.hpp` also prints the checked expressions'
text, by reflecting the test's body with P4395 (clang fork only).
