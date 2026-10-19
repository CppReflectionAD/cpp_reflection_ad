// TEST-PASSES-ON: clang
//
// Known GCC gap: comparing two reflections of statements can crash GCC.
//
// Walking a function body with statements_of() and comparing each statement
// with the one it came from (`s != stmt`) is an internal compiler error in
// cp_tree_equal when the body has a declaration followed by a `for` loop.
// Either one alone compiles. Clang accepts the file.
//
// This blocks include/testing/clang_only/reflect_test.hpp, which walks a test
// body that way to find its checks, so that header and its test are
// clang-only for now.
//
// tests/CMakeLists.txt builds this file as an ordinary test on clang (see
// TEST-PASSES-ON above). On GCC it expects the crash recorded in
// statement_reflection_equality.fail.txt. When GCC fixes the gap, move this
// file to tests/common/, drop the TEST-PASSES-ON line and delete the
// .fail.txt; then try moving reflect_test to the common directories.
#include <meta>

namespace m = std::meta;

void decl_then_for() {
  [[maybe_unused]] double a = 1.0;
  for (int i = 0; i < 3; ++i) {
  }
}

// Compares each statement under `stmt` with `stmt`.
consteval void compare_with_parents(m::info stmt) {
  for (m::info s : m::statements_of(stmt))
    if (s != stmt && m::is_statement(s))
      compare_with_parents(s);
}

consteval bool walk_body() {
  compare_with_parents(m::body_of(^^decl_then_for));
  return true;
}

static_assert(walk_body());

int main() {}
