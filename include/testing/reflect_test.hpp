#ifndef REFLECTION_AD_TESTING_REFLECT_TEST_HPP
#define REFLECTION_AD_TESTING_REFLECT_TEST_HPP

// Test checks without macros. A failed check prints where it is and the
// values it compared:
//
//   void my_test() {
//     double a = 1.0;
//     reflect_test::expect_equal(a + a, 3.0);
//   }
//   int main() {
//     my_test();
//     return reflect_test::result();
//   }
//
// prints
//
//   my_test.cpp:3 Failure
//   Expected lhs == rhs
//     lhs: 2.0
//     rhs: 3.0
//
// The checks are plain function templates, so they work anywhere: in helpers,
// lambdas, templates and loops, several to a line. A value prints with `<<`
// if it can. Strings are quoted, floating-point values print in the shortest
// form that reads back exactly, ranges print their elements (`{1, 2}`),
// aggregates their members (`{.x = 1, .y = 2.5}`, by reflecting the type),
// and anything else prints its type name (`<widget>`).
//
// testing/clang_only/reflect_test_expressions.hpp adds the operands' source
// text, as EXPECT_EQUAL prints it, in place of `lhs` and `rhs`.

#include <charconv>
#include <cstddef>
#include <iostream>
#include <limits>
#include <meta>
#include <ranges>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace reflect_test {

namespace detail {

inline int error_count = 0;

inline void increase_error_count() {
  constexpr int limit = 10;
  if (++error_count == limit) {
    std::cout << "maximum number of errors exceeded" << std::endl;
    throw std::runtime_error("maximum number of errors exceeded");
  }
}

// What a failed check calls its operands. A null name stands for the
// operand's value, as for a literal.
struct operand_names {
  const char *lhs = "lhs";
  const char *rhs = "rhs";
};

// The names of the operands of the check at a source location, if set.
// testing/clang_only/reflect_test_expressions.hpp sets it.
inline operand_names (*operand_names_at)(const std::source_location &) =
    nullptr;

// Values as text.

constexpr std::string quoted(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
    }
  }
  return out + "\"";
}

// The shortest text that reads back as `v`, with a `.0` if it would read as
// an integer.
template <class T> void write_floating(std::ostream &os, T v) {
  char buffer[64];
  const auto [end, error] = std::to_chars(buffer, buffer + sizeof buffer, v);
  if (error != std::errc{}) {
    os << v;
    return;
  }
  const std::string_view text(buffer, static_cast<std::size_t>(end - buffer));
  os << text;
  if (text.find_first_of(".en") == std::string_view::npos) // e, inf, nan
    os << ".0";
}

template <class T>
inline constexpr std::string_view type_name =
    std::define_static_string(std::meta::display_string_of(^^T));

template <class T> void write_value(std::ostream &os, const T &v);

template <class R> void write_range(std::ostream &os, const R &range) {
  constexpr std::size_t limit = 16;
  os << '{';
  std::size_t count = 0;
  for (const auto &element : range) {
    if (count != 0)
      os << ", ";
    if (count++ == limit) {
      os << "...";
      break;
    }
    write_value(os, element);
  }
  os << '}';
}

template <class T> void write_members(std::ostream &os, const T &v) {
  os << '{';
  std::string_view separator;
  template for (constexpr std::meta::info member :
                std::define_static_array(std::meta::nonstatic_data_members_of(
                    ^^T, std::meta::access_context::unchecked()))) {
    os << separator << '.' << std::meta::identifier_of(member) << " = ";
    write_value(os, v.[:member:]);
    separator = ", ";
  }
  os << '}';
}

template <class T> void write_value(std::ostream &os, const T &v) {
  if constexpr (std::is_same_v<T, bool>) {
    os << (v ? "true" : "false");
  } else if constexpr (std::is_same_v<T, char>) {
    os << '\'' << v << '\'';
  } else if constexpr (std::is_floating_point_v<T>) {
    write_floating(os, v);
  } else if constexpr (std::is_null_pointer_v<T>) {
    os << "nullptr";
  } else if constexpr (std::is_convertible_v<const T &, std::string_view>) {
    if constexpr (std::is_pointer_v<T>) {
      if (v == nullptr) {
        os << "nullptr";
        return;
      }
    }
    os << quoted(v);
  } else if constexpr (requires { os << v; }) {
    os << v;
  } else if constexpr (std::ranges::input_range<const T>) {
    write_range(os, v);
  } else if constexpr (std::is_class_v<T> && std::is_aggregate_v<T>) {
    write_members(os, v);
  } else {
    os << '<' << type_name<T> << '>';
  }
}

template <class T> std::string value_text(const T &v) {
  std::ostringstream os;
  os.precision(std::numeric_limits<double>::max_digits10);
  write_value(os, v);
  return os.str();
}

// Prints, like EXPECT_EQUAL and EXPECT_NOT_EQUAL, where the check is and
// what it expected, then the value of each operand that isn't a literal.
template <class L, class R>
void report_failure(const std::source_location &loc, std::string_view op,
                    const L &lhs, const R &rhs) {
  const std::string lhs_value = value_text(lhs);
  const std::string rhs_value = value_text(rhs);
  const operand_names names =
      operand_names_at != nullptr ? operand_names_at(loc) : operand_names{};
  const std::string lhs_name = names.lhs != nullptr ? names.lhs : lhs_value;
  const std::string rhs_name = names.rhs != nullptr ? names.rhs : rhs_value;

  std::cout << loc.file_name() << ":" << loc.line() << " Failure" << std::endl;
  std::cout << "Expected " << lhs_name << " " << op << " " << rhs_name
            << std::endl;
  if (lhs_name != lhs_value)
    std::cout << "  " << lhs_name << ": " << lhs_value << std::endl;
  if (rhs_name != rhs_value)
    std::cout << "  " << rhs_name << ": " << rhs_value << std::endl;
  increase_error_count();
}
} // namespace detail

// Number of failed checks so far.
inline int result() { return detail::error_count; }

// Replaces EXPECT_EQUAL(VAL1, VAL2).
template <class L, class R>
void expect_equal(const L &lhs, const R &rhs,
                  std::source_location loc = std::source_location::current()) {
  if (lhs != rhs)
    detail::report_failure(loc, "==", lhs, rhs);
}

// Replaces EXPECT_NOT_EQUAL(VAL1, VAL2).
template <class L, class R>
void expect_not_equal(
    const L &lhs, const R &rhs,
    std::source_location loc = std::source_location::current()) {
  if (lhs == rhs)
    detail::report_failure(loc, "!=", lhs, rhs);
}

} // namespace reflect_test

#endif // REFLECTION_AD_TESTING_REFLECT_TEST_HPP
