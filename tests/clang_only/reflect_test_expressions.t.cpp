// Checks the reflect_test expression-text layer
// (include/testing/clang_only/reflect_test_expressions.hpp): that a failure
// names its operands by their source text. Checked with test_simple's macros.

#include <testing/clang_only/reflect_test_expressions.hpp>

// side_by_side_same() fails more checks than the default limit.
#define TEST_SIMPLE_MAX_ERRORS 100
#include <test_simple_include.hpp>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

double twice(double x) { return 2 * x; }

double scaled(double x, double factor = 2.0) { return factor * x; }

void passing_checks() {
  double a = 1.0;
  double b = 2.0;
  reflect_test::expect_not_equal(a, b);
  if (a < b) {
    reflect_test::expect_not_equal(a + b, b);
  }
  for (int i = 0; i < 3; ++i)
    reflect_test::expect_not_equal(i, 3);
}

void failing_checks() {
  double a = 1.0;
  double b = 1.0;
  reflect_test::expect_not_equal(a, b);
  if (a == b) {
    reflect_test::expect_not_equal(twice(a), a + b);
  }
  for (int i = 0; i < 1; ++i)
    reflect_test::expect_not_equal(-i, 0);
}

void passing_equal_checks() {
  double a = 1.0;
  double b = 1.0;
  reflect_test::expect_equal(a, b);
  if (a == b) {
    reflect_test::expect_equal(twice(a), a + b);
  }
  for (int i = 0; i < 3; ++i)
    reflect_test::expect_equal(i - i, 0);
}

// Mixes both checks in one body.
void failing_equal_checks() {
  double a = 1.0;
  double b = 2.0;
  reflect_test::expect_equal(a, b);
  reflect_test::expect_not_equal(b, twice(a));
  if (a < b) {
    reflect_test::expect_equal(twice(a), a + b);
  }
}

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

// Whether `line` ends with `suffix`.
bool ends_with(const std::string &line, const std::string &suffix) {
  return line.size() >= suffix.size() &&
         line.compare(line.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The failures in `lines`, each from its "<file>:<line> Failure" line to the
// next one.
std::vector<std::vector<std::string>>
failures_in(const std::vector<std::string> &lines) {
  std::vector<std::vector<std::string>> failures;
  for (const std::string &line : lines) {
    if (ends_with(line, " Failure") || failures.empty())
      failures.emplace_back();
    failures.back().push_back(line);
  }
  return failures;
}

int compare_side_by_side();

int main() {
  namespace detail = reflect_test::detail;
  reflect_test::max_errors = 100;

  // Passing checks print nothing and count no error.
  const auto passing =
      output_of([] { reflect_test::run_with_text<^^passing_checks>(); });
  EXPECT_EQUAL(passing.size(), 0u);
  EXPECT_EQUAL(reflect_test::result(), 0);

  // Each failing check prints where it is, its operands' text and the values
  // of those that aren't literals.
  const auto failing = failures_in(
      output_of([] { reflect_test::run_with_text<^^failing_checks>(); }));
  EXPECT_EQUAL(reflect_test::result(), 3);
  EXPECT_EQUAL(failing.size(), 3u);
  if (failing.size() == 3) {
    EXPECT_TRUE(
        ends_with(failing[0][0], "reflect_test_expressions.t.cpp:34 Failure"));
    EXPECT_TRUE(failing[0] ==
                std::vector<std::string>({failing[0][0], "Expected a != b",
                                          "  a: 1.0", "  b: 1.0"}));
    EXPECT_TRUE(
        ends_with(failing[1][0], "reflect_test_expressions.t.cpp:36 Failure"));
    EXPECT_TRUE(failing[1] == std::vector<std::string>(
                                  {failing[1][0], "Expected twice(a) != a + b",
                                   "  twice(a): 2.0", "  a + b: 2.0"}));
    EXPECT_TRUE(
        ends_with(failing[2][0], "reflect_test_expressions.t.cpp:39 Failure"));
    EXPECT_TRUE(failing[2] ==
                std::vector<std::string>(
                    {failing[2][0], "Expected -i != 0", "  -i: 0"}));
  }

  // The same for expect_equal, mixed with expect_not_equal in one body.
  const auto passing_equal =
      output_of([] { reflect_test::run_with_text<^^passing_equal_checks>(); });
  EXPECT_EQUAL(passing_equal.size(), 0u);
  EXPECT_EQUAL(reflect_test::result(), 3);

  const auto failing_equal = failures_in(
      output_of([] { reflect_test::run_with_text<^^failing_equal_checks>(); }));
  EXPECT_EQUAL(reflect_test::result(), 6);
  EXPECT_EQUAL(failing_equal.size(), 3u);
  if (failing_equal.size() == 3) {
    EXPECT_TRUE(ends_with(failing_equal[0][0],
                          "reflect_test_expressions.t.cpp:57 Failure"));
    EXPECT_EQUAL(failing_equal[0][1], "Expected a == b");
    EXPECT_TRUE(ends_with(failing_equal[1][0],
                          "reflect_test_expressions.t.cpp:58 Failure"));
    EXPECT_EQUAL(failing_equal[1][1], "Expected b != twice(a)");
    EXPECT_TRUE(ends_with(failing_equal[2][0],
                          "reflect_test_expressions.t.cpp:60 Failure"));
    EXPECT_EQUAL(failing_equal[2][1], "Expected twice(a) == a + b");
  }

  // Outside run_with_text() a check prints `lhs` and `rhs`, as without this
  // layer.
  const auto outside = output_of([] { reflect_test::expect_not_equal(1, 1); });
  EXPECT_EQUAL(reflect_test::result(), 7);
  EXPECT_EQUAL(outside.size(), 4u);
  if (outside.size() == 4)
    EXPECT_EQUAL(outside[1], "Expected lhs != rhs");

  for (const auto &failure : failing)
    for (const std::string &line : failure)
      std::cout << "  | " << line << std::endl;

  detail::error_count = 0;

  // The macros and the functions print the same first two lines, except
  // where the text can't be recovered from the reflection.
  EXPECT_EQUAL(compare_side_by_side(), 0);
  TEST_END;
}

// Macros and functions side by side: each EXPECT_* is followed, on the next
// line, by the same check through reflect_test. All checks fail.

void side_by_side_same() {
  double a = 1.0;
  double b = 2.0;
  double c = 3.0;
  int i = 1;
  unsigned n = 2u;
  std::string s = "abc";
  std::vector<double> v = {1.0, 2.0};
  EXPECT_EQUAL(a, b);
  reflect_test::expect_equal(a, b);
  EXPECT_EQUAL(a + b, c * b);
  reflect_test::expect_equal(a + b, c * b);
  EXPECT_EQUAL((a + b) * c, a - (b - c));
  reflect_test::expect_equal((a + b) * c, a - (b - c));
  EXPECT_EQUAL(twice(a), -b);
  reflect_test::expect_equal(twice(a), -b);
  EXPECT_EQUAL(a, 2.5);
  reflect_test::expect_equal(a, 2.5);
  EXPECT_EQUAL(b, 1e-13);
  reflect_test::expect_equal(b, 1e-13);
  EXPECT_NOT_EQUAL(c, 3.0);
  reflect_test::expect_not_equal(c, 3.0);
  EXPECT_EQUAL(a < b, false);
  reflect_test::expect_equal(a < b, false);
  EXPECT_EQUAL(a < b ? a : b, c);
  reflect_test::expect_equal(a < b ? a : b, c);
  EXPECT_EQUAL(i, 2);
  reflect_test::expect_equal(i, 2);
  EXPECT_EQUAL(n, 3u);
  reflect_test::expect_equal(n, 3u);
  EXPECT_EQUAL(i * (i + 1), 1);
  reflect_test::expect_equal(i * (i + 1), 1);
  EXPECT_EQUAL(i, 'b');
  reflect_test::expect_equal(i, 'b');
  EXPECT_EQUAL(i, '\n');
  reflect_test::expect_equal(i, '\n');
  EXPECT_EQUAL(s, "abd");
  reflect_test::expect_equal(s, "abd");
  EXPECT_EQUAL(s == "abc", false);
  reflect_test::expect_equal(s == "abc", false);
  EXPECT_EQUAL("abd" != s, false);
  reflect_test::expect_equal("abd" != s, false);
  EXPECT_EQUAL(v[0], 2.0);
  reflect_test::expect_equal(v[0], 2.0);
  EXPECT_EQUAL(scaled(v[1]), 1.0);
  reflect_test::expect_equal(scaled(v[1]), 1.0);
}

// Where the text differs: see the header comment of
// reflect_test_expressions.hpp.
void side_by_side_different() {
  int i = 1;
  std::string s = "abc";
  std::vector<double> v = {1.0, 2.0};
  EXPECT_EQUAL(std::abs(i - 3), 1);
  reflect_test::expect_equal(std::abs(i - 3), 1);
  EXPECT_EQUAL(v.size(), 3u);
  reflect_test::expect_equal(v.size(), 3u);
  EXPECT_EQUAL(static_cast<double>(i), 2.0);
  reflect_test::expect_equal(static_cast<double>(i), 2.0);
  EXPECT_EQUAL(0x10, i);
  reflect_test::expect_equal(0x10, i);
  EXPECT_EQUAL(1.e3, v[0]);
  reflect_test::expect_equal(1.e3, v[0]);
  EXPECT_NOT_EQUAL(s, std::string("abc"));
  reflect_test::expect_not_equal(s, std::string("abc"));
}

// The file name and line of a "<file>:<line> Failure" line.
std::string location_of(const std::string &failure) {
  const std::size_t end = failure.rfind(' ');
  const std::size_t start = failure.rfind('/', end);
  return failure.substr(start == std::string::npos ? 0 : start + 1,
                        end - (start == std::string::npos ? 0 : start + 1));
}

// "<file>:<line + 1>" for "<file>:<line>".
std::string next_line(const std::string &location) {
  const std::size_t colon = location.rfind(':');
  return location.substr(0, colon + 1) +
         std::to_string(std::stoi(location.substr(colon + 1)) + 1);
}

// Runs `Fn` and prints the failures of each macro check and of the function
// check after it side by side, then the values the function prints. Returns
// the number of pairs whose first two lines differ, other than the function's
// line being the next one.
template <std::meta::info Fn> int side_by_side(const char *title) {
  const int macro_errors = detail_test::_result;
  detail_test::_result = 0;
  const auto failures =
      failures_in(output_of([] { reflect_test::run_with_text<Fn>(); }));
  detail_test::_result = macro_errors;
  reflect_test::detail::error_count = 0;

  const auto column = [](std::string text) {
    text.resize(std::max<std::size_t>(text.size(), 44), ' ');
    return text;
  };
  std::cout << title << ":\n  " << column("EXPECT_*") << " | "
            << "reflect_test::expect_*" << std::endl;
  int different = 0;
  for (std::size_t i = 0; i + 1 < failures.size(); i += 2) {
    const auto &macro = failures[i];
    const auto &function = failures[i + 1];
    if (macro.size() != 2 || function.size() < 2) {
      ++different;
      continue;
    }
    const std::string macro_at = location_of(macro[0]);
    const std::string function_at = location_of(function[0]);
    const bool same =
        function_at == next_line(macro_at) && macro[1] == function[1];
    different += same ? 0 : 1;
    std::cout << "  " << column(macro_at) << " | " << function_at << "\n  "
              << column(macro[1]) << " | " << function[1]
              << (same ? "" : "   <- differs") << std::endl;
    for (std::size_t j = 2; j < function.size(); ++j)
      std::cout << "  " << column("") << " | " << function[j] << std::endl;
  }
  if (failures.size() % 2 != 0)
    ++different;
  return different;
}

int compare_side_by_side() {
  const int different = side_by_side<^^side_by_side_same>("same text");
  side_by_side<^^side_by_side_different>("different text");
  return different;
}
