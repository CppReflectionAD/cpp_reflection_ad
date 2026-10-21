// Checks reflect_test (include/testing/reflect_test.hpp): that its checks pass
// and fail when they should, and what a failure prints. The output is checked
// with test_simple's macros.

#include <testing/reflect_test.hpp>

#include <test_simple_include.hpp>

#include <cstdint>
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

// An aggregate with a base.
struct base {
  int x;
};
struct derived : base {
  int y;
};

bool operator==(const derived &a, const derived &b) {
  return a.x == b.x && a.y == b.y;
}

// An aggregate with an anonymous union.
struct tagged {
  int kind;
  union {
    int i;
    float f;
  };
};

bool operator==(const tagged &a, const tagged &b) {
  return a.kind == b.kind && a.i == b.i;
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
    reflect_test::expect_equal(derived{{1}, 2}, derived{{1}, 2});
    reflect_test::expect_equal(tagged{1, {2}}, tagged{1, {2}});
    check_twice(1.5, 3.0);
  });
  EXPECT_EQUAL(passing.size(), 0u);
  EXPECT_EQUAL(reflect_test::result(), 0);

  // A failure prints where it is, what was expected and both values.
  const auto failing = output_of([] { reflect_test::expect_equal(1.0, 2); });
  EXPECT_EQUAL(reflect_test::result(), 1);
  EXPECT_EQUAL(failing.size(), 4u);
  if (failing.size() == 4) {
    EXPECT_TRUE(ends_with(failing[0], "reflect_test.t.cpp:109 Failure"));
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
    EXPECT_TRUE(ends_with(more[0], "reflect_test.t.cpp:121 Failure"));
    EXPECT_EQUAL(more[1], "Expected lhs != rhs");
    EXPECT_EQUAL(more[2], "  lhs: 'a'");
    EXPECT_EQUAL(more[3], "  rhs: 'a'");
    EXPECT_TRUE(ends_with(more[4], "reflect_test.t.cpp:87 Failure"));
    EXPECT_EQUAL(more[5], "Expected lhs == rhs");
    EXPECT_EQUAL(more[6], "  lhs: 2.0");
    EXPECT_EQUAL(more[7], "  rhs: 3.0");
  }
  detail::error_count = 0;

  // How values print.
  EXPECT_EQUAL(text(true), "true");
  EXPECT_EQUAL(text('x'), "'x'");
  EXPECT_EQUAL(text('\n'), "'\\n'");
  EXPECT_EQUAL(text('\''), "'\\''");
  EXPECT_EQUAL(text('"'), "'\"'");
  EXPECT_EQUAL(text('\r'), "'\\r'");
  EXPECT_EQUAL(text('\x01'), "'\\x01'");
  EXPECT_EQUAL(text('\x7f'), "'\\x7f'");
  EXPECT_EQUAL(text(std::string("a\x1b[0m")), "\"a\\x1b[0m\"");
  EXPECT_EQUAL(text(std::string("\xc3\xa9")), "\"\xc3\xa9\""); // UTF-8 é
  EXPECT_EQUAL(text(std::uint8_t{0}), "0");
  EXPECT_EQUAL(text(std::uint8_t{10}), "10");
  EXPECT_EQUAL(text(std::int8_t{-65}), "-65");
  EXPECT_EQUAL(text(-3), "-3");
  EXPECT_EQUAL(text(42u), "42");
  EXPECT_EQUAL(text(2.0), "2.0");
  EXPECT_EQUAL(text(0.1), "0.1");
  EXPECT_EQUAL(text(1e-13), "1e-13");
  EXPECT_EQUAL(text(2.5f), "2.5");
  EXPECT_EQUAL(text(std::numeric_limits<double>::infinity()), "inf");
  EXPECT_EQUAL(text(std::string("a\"b")), "\"a\\\"b\"");
  EXPECT_EQUAL(text(std::string("a'\tb\n")), "\"a'\\tb\\n\"");
  EXPECT_EQUAL(text(std::string("a\0b", 3)), "\"a\\0b\"");
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
  EXPECT_EQUAL(text(derived{{1}, 2}), "{{.x = 1}, .y = 2}");
  EXPECT_EQUAL(text(tagged{1, {2}}), "{.kind = 1, <anonymous union>}");
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

  // expect_near_abs passes within the tolerance, in the less precise type.
  const auto near = output_of([] {
    reflect_test::expect_near_abs(1.0, 1.0 + 1e-12, 1e-10);
    reflect_test::expect_near_abs(1.0, 1.05, -0.1);
    reflect_test::expect_near_abs(0.1f, 0.1, 1e-9);
    reflect_test::expect_near_abs(2, 2.5, 1);
  });
  EXPECT_EQUAL(near.size(), 0u);
  EXPECT_EQUAL(reflect_test::result(), 0);

  // A failure prints the difference, the tolerance and both values.
  const auto far = output_of([] {
    reflect_test::expect_near_abs(1.0, 1.5, 0.1);
    reflect_test::expect_near_abs(1.0, 1.1, 0.1);
    reflect_test::expect_near_abs(std::numeric_limits<double>::quiet_NaN(), 1.0,
                                  0.1);
  });
  EXPECT_EQUAL(reflect_test::result(), 3);
  EXPECT_EQUAL(far.size(), 15u);
  if (far.size() == 15) {
    EXPECT_TRUE(ends_with(far[0], " Failure"));
    EXPECT_EQUAL(far[1], "Expected |lhs - rhs| < 0.1");
    EXPECT_EQUAL(far[2], "  |lhs - rhs|: 0.5");
    EXPECT_EQUAL(far[3], "  lhs: 1.0");
    EXPECT_EQUAL(far[4], "  rhs: 1.5");
    EXPECT_EQUAL(far[7], "  |lhs - rhs|: 0.10000000000000009");
    EXPECT_EQUAL(far[12], "  |lhs - rhs|: nan");
  }
  detail::error_count = 0;

  for (const std::string &line : failing)
    std::cout << "  | " << line << std::endl;
  TEST_END;
}
