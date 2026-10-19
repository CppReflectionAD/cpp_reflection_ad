// Checks reflect_test (include/testing/reflect_test.hpp): that its checks pass
// and fail when they should, and what a failure prints. The output is checked
// with test_simple's macros.

#include <testing/reflect_test.hpp>

#include <test_simple_include.hpp>

#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct point {
  int x;
  double y;
};

bool operator==(const point &a, const point &b) {
  return a.x == b.x && a.y == b.y;
}

// Not an aggregate (it has a constructor), not printable and not a range.
class widget {
public:
  explicit widget(int id) : id_(id) {}
  bool operator==(const widget &) const = default;

private:
  int id_;
};

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

bool ends_with(const std::string &line, const std::string &suffix) {
  return line.size() >= suffix.size() &&
         line.compare(line.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The text a value prints as in a failure.
template <class T> std::string text(const T &v) {
  return reflect_test::detail::value_text(v);
}

// A check in a helper, called from a lambda.
void check_twice(double x, double expected) {
  reflect_test::expect_equal(2 * x, expected);
}

} // namespace

int main() {
  namespace detail = reflect_test::detail;

  // Passing checks print nothing and count no error.
  const auto passing = output_of([] {
    reflect_test::expect_equal(1 + 1, 2);
    reflect_test::expect_not_equal(1.0, 2.0);
    reflect_test::expect_equal(std::string("a"), "a");
    reflect_test::expect_equal(point{1, 2.5}, point{1, 2.5});
    check_twice(1.5, 3.0);
  });
  EXPECT_EQUAL(passing.size(), 0u);
  EXPECT_EQUAL(reflect_test::result(), 0);

  // A failure prints where it is, what was expected and both values.
  const auto failing = output_of([] { reflect_test::expect_equal(1.0, 2); });
  EXPECT_EQUAL(reflect_test::result(), 1);
  EXPECT_EQUAL(failing.size(), 4u);
  if (failing.size() == 4) {
    EXPECT_TRUE(ends_with(failing[0], "reflect_test.t.cpp:81 Failure"));
    EXPECT_EQUAL(failing[1], "Expected lhs == rhs");
    EXPECT_EQUAL(failing[2], "  lhs: 1.0");
    EXPECT_EQUAL(failing[3], "  rhs: 2");
  }

  // The same for expect_not_equal, and for a check in a helper.
  const auto more = output_of([] {
    reflect_test::expect_not_equal('a', 'a');
    check_twice(1.0, 3.0);
  });
  EXPECT_EQUAL(reflect_test::result(), 3);
  EXPECT_EQUAL(more.size(), 8u);
  if (more.size() == 8) {
    EXPECT_TRUE(ends_with(more[0], "reflect_test.t.cpp:93 Failure"));
    EXPECT_EQUAL(more[1], "Expected lhs != rhs");
    EXPECT_EQUAL(more[2], "  lhs: 'a'");
    EXPECT_EQUAL(more[3], "  rhs: 'a'");
    EXPECT_TRUE(ends_with(more[4], "reflect_test.t.cpp:61 Failure"));
    EXPECT_EQUAL(more[5], "Expected lhs == rhs");
    EXPECT_EQUAL(more[6], "  lhs: 2.0");
    EXPECT_EQUAL(more[7], "  rhs: 3.0");
  }
  detail::error_count = 0;

  // How values print.
  EXPECT_EQUAL(text(true), "true");
  EXPECT_EQUAL(text('x'), "'x'");
  EXPECT_EQUAL(text(-3), "-3");
  EXPECT_EQUAL(text(42u), "42");
  EXPECT_EQUAL(text(2.0), "2.0");
  EXPECT_EQUAL(text(0.1), "0.1");
  EXPECT_EQUAL(text(1e-13), "1e-13");
  EXPECT_EQUAL(text(2.5f), "2.5");
  EXPECT_EQUAL(text(std::numeric_limits<double>::infinity()), "inf");
  EXPECT_EQUAL(text(std::string("a\"b")), "\"a\\\"b\"");
  EXPECT_EQUAL(text("abc"), "\"abc\"");
  const char *null_string = nullptr;
  EXPECT_EQUAL(text(null_string), "nullptr");
  EXPECT_EQUAL(text(nullptr), "nullptr");
  EXPECT_EQUAL(text(std::vector<double>{1.0, 2.5}), "{1.0, 2.5}");
  EXPECT_EQUAL(text(std::vector<int>(20, 7)),
               "{7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, ...}");
  EXPECT_EQUAL(text(std::vector<std::string>{"a", "b"}), "{\"a\", \"b\"}");
  EXPECT_EQUAL(text(point{1, 2.5}), "{.x = 1, .y = 2.5}");
  EXPECT_EQUAL(text(std::vector<point>{{1, 0.5}}), "{{.x = 1, .y = 0.5}}");
  const std::string widget_text = text(widget(1));
  EXPECT_TRUE(widget_text.starts_with("<") && widget_text.ends_with(">") &&
              widget_text.find("widget") != std::string::npos);

  // A failure with values that aren't printable with `<<`.
  const auto aggregates = output_of(
      [] { reflect_test::expect_equal(point{1, 2.5}, point{1, 3.0}); });
  EXPECT_EQUAL(aggregates.size(), 4u);
  if (aggregates.size() == 4) {
    EXPECT_EQUAL(aggregates[2], "  lhs: {.x = 1, .y = 2.5}");
    EXPECT_EQUAL(aggregates[3], "  rhs: {.x = 1, .y = 3.0}");
  }
  detail::error_count = 0;

  for (const std::string &line : failing)
    std::cout << "  | " << line << std::endl;
  TEST_END;
}
