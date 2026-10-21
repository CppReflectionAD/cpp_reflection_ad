#ifndef REFLECTION_AD_IHM_IHM_HPP
#define REFLECTION_AD_IHM_IHM_HPP

// I Hate Macros (ihm): a C++ test framework without macros, built on C++26
// reflection (see include/ihm/README.md). Its checks are plain functions
// that replace test_simple's macros: ihm::expect_equal for EXPECT_EQUAL, and
// so on. A failed check prints where it is and the values it compared:
//
//   void my_test() {
//     double a = 1.0;
//     ihm::expect_equal(a + a, 3.0);
//   }
//   int main() {
//     my_test();
//     return ihm::result();
//   }
//
// prints
//
//   my_test.cpp:3 Failure
//   Expected lhs == rhs
//     lhs: 2.0
//     rhs: 3.0
//
// Being function templates, the checks work anywhere: in helpers, lambdas,
// templates and loops, several to a line. They find where they are
// with std::source_location, and print values by reflection: with `<<` if a
// value has it; strings and char arrays quoted; floating-point values in the
// shortest form that reads back exactly (a long double wider than a double
// with all the digits it may need); ranges and other arrays as their elements
// (`{1, 2}`); aggregates as their members (`{.x = 1, .y = 2.5}`); and anything
// else as its type name (`<widget>`).
//
// ihm/clang_only/ihm_expressions.hpp adds the operands' source
// text, as EXPECT_EQUAL prints it, in place of `lhs` and `rhs`.

#include <charconv>
#include <cmath>
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
#include <type_traits>

namespace ihm {

// The number of failed checks at which a test stops (it throws). A test that
// needs more sets it before its checks run, e.g. at the start of main().
inline int max_errors = 10;

namespace detail {

inline int error_count = 0;

inline void increase_error_count() {
  if (++error_count == max_errors) {
    std::cout << "maximum number of errors exceeded" << std::endl;
    throw std::runtime_error("maximum number of errors exceeded");
  }
}

// C++ operator precedence, higher binds tighter.
enum precedence : int {
  comma_prec = 1,
  assignment_prec, // also the conditional operator
  or_prec,
  and_prec,
  bitor_prec,
  xor_prec,
  bitand_prec,
  equality_prec,
  relational_prec,
  spaceship_prec,
  shift_prec,
  additive_prec,
  multiplicative_prec,
  unary_prec,
  postfix_prec,
  primary_prec
};

// What a failed check calls its operands, and the precedence of each name's
// outermost operator, so that a message can parenthesise it. A null name
// stands for the operand's value, as for a literal.
struct operand_names {
  const char *lhs = "lhs";
  const char *rhs = "rhs";
  int lhs_prec = primary_prec;
  int rhs_prec = primary_prec;
};

// The names of the operands of the check at a source location, if set.
// ihm/clang_only/ihm_expressions.hpp sets it.
inline operand_names (*operand_names_at)(const std::source_location &) =
    nullptr;

// Values as text.

// `s` between `quote`s, escaped as in a C++ literal, so that the text is on
// one line and blanks and control characters show. Other control characters
// print as `\xNN`.
constexpr std::string quoted(std::string_view s, char quote = '"') {
  constexpr std::string_view hex = "0123456789abcdef";
  std::string out(1, quote);
  for (const char c : s) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == quote || c == '\\')
      out += '\\';
    if (c == '\n')
      out += "\\n";
    else if (c == '\t')
      out += "\\t";
    else if (c == '\r')
      out += "\\r";
    else if (c == '\0')
      out += "\\0";
    else if (byte < 0x20 || byte == 0x7f) {
      out += "\\x";
      out += hex[byte / 16];
      out += hex[byte % 16];
    } else
      out += c;
  }
  return out + quote;
}

// The shortest text that reads back as `v`, with a `.0` if it would read as
// an integer. libc++'s to_chars converts a long double to double first, so a
// long double more precise than a double prints with all the digits it may
// need instead (not the shortest), lest two that differ beyond a double's
// precision print alike.
template <class T> void write_floating(std::ostream &os, T v) {
  std::string text;
  if constexpr (std::numeric_limits<T>::digits >
                std::numeric_limits<double>::digits) {
    std::ostringstream wide;
    wide.precision(std::numeric_limits<T>::max_digits10);
    wide << v;
    text = wide.str();
  } else {
    char buffer[64]; // more than the longest shortest form
    text.assign(buffer, std::to_chars(buffer, buffer + sizeof buffer, v).ptr);
  }
  os << text;
  if (text.find_first_of(".en") == std::string::npos) // e, inf, nan
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

// An aggregate as its initializer: its bases first, as `{...}`, then its
// members, as `.name = value`. An anonymous union has no name and its active
// member isn't known, so it prints as `<anonymous union>`.
template <class T> void write_members(std::ostream &os, const T &v) {
  constexpr auto unchecked = std::meta::access_context::unchecked();
  os << '{';
  std::string_view separator;
  template for (constexpr std::meta::info base :
                std::define_static_array(std::meta::bases_of(^^T, unchecked))) {
    os << separator;
    write_value(os,
                static_cast<const typename[:std::meta::type_of(base):] &>(v));
    separator = ", ";
  }
  template for (constexpr std::meta::info member : std::define_static_array(
                    std::meta::nonstatic_data_members_of(^^T, unchecked))) {
    os << separator;
    if constexpr (std::meta::has_identifier(member)) {
      os << '.' << std::meta::identifier_of(member) << " = ";
      write_value(os, v.[:member:]);
    } else {
      os << "<anonymous union>";
    }
    separator = ", ";
  }
  os << '}';
}

template <class T> void write_value(std::ostream &os, const T &v) {
  if constexpr (std::is_same_v<T, bool>) {
    os << (v ? "true" : "false");
  } else if constexpr (std::is_same_v<T, char>) {
    os << quoted(std::string_view(&v, 1), '\'');
  } else if constexpr (std::is_same_v<T, signed char> ||
                       std::is_same_v<T, unsigned char>) {
    os << +v; // a number (std::int8_t, std::uint8_t), not a character
  } else if constexpr (std::is_floating_point_v<T>) {
    write_floating(os, v);
  } else if constexpr (std::is_null_pointer_v<T>) {
    os << "nullptr";
  } else if constexpr (std::is_array_v<T>) {
    // Before the `<<` branch, where an array would decay to a pointer.
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_extent_t<T>>,
                                 char>) {
      // Up to the first NUL, if any: a string literal prints as written, and
      // a char buffer without one isn't read past its end.
      const std::string_view s(v, std::extent_v<T>);
      os << quoted(s.substr(0, s.find('\0')));
    } else {
      write_range(os, v);
    }
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

// The names a failed check prints for its operands, their precedence, and
// their values' text.
struct operand_texts {
  std::string lhs_name;
  std::string lhs_value;
  std::string rhs_name;
  std::string rhs_value;
  int lhs_prec;
  int rhs_prec;

  // `lhs op rhs`, with the parentheses that a left-associative operator of
  // precedence `prec` needs around the operands.
  std::string joined(std::string_view op, int prec) const {
    const auto at = [](const std::string &name, int name_prec, int min_prec) {
      return name_prec < min_prec ? "(" + name + ")" : name;
    };
    return at(lhs_name, lhs_prec, prec) + " " + std::string(op) + " " +
           at(rhs_name, rhs_prec, prec + 1);
  }
};

template <class L, class R>
operand_texts texts_of(const std::source_location &loc, const L &lhs,
                       const R &rhs) {
  const operand_names names =
      operand_names_at != nullptr ? operand_names_at(loc) : operand_names{};
  operand_texts t{
      "", value_text(lhs), "", value_text(rhs), names.lhs_prec, names.rhs_prec};
  t.lhs_name = names.lhs != nullptr ? names.lhs : t.lhs_value;
  t.rhs_name = names.rhs != nullptr ? names.rhs : t.rhs_value;
  return t;
}

// Prints where a failed check is and what it expected.
inline void print_failure(const std::source_location &loc,
                          std::string_view expected) {
  std::cout << loc.file_name() << ":" << loc.line() << " Failure" << std::endl;
  std::cout << "Expected " << expected << std::endl;
}

// Prints a value under its name, unless the name is the value (a literal).
inline void print_value(std::string_view name, std::string_view value) {
  if (name != value)
    std::cout << "  " << name << ": " << value << std::endl;
}

// Prints, like EXPECT_EQUAL and EXPECT_NOT_EQUAL, where the check is and
// what it expected, then the value of each operand that isn't a literal.
template <class L, class R>
void report_failure(const std::source_location &loc, std::string_view op,
                    const L &lhs, const R &rhs) {
  const operand_texts t = texts_of(loc, lhs, rhs);
  print_failure(loc, t.joined(op, equality_prec));
  print_value(t.lhs_name, t.lhs_value);
  print_value(t.rhs_name, t.rhs_value);
  increase_error_count();
}

// The floating-point type expect_near_abs compares in: the less precise of
// its operands' types (double for an integer), so that a float isn't held to
// a double's precision.
template <class T>
using floating_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;

template <class L, class R>
using near_t = std::conditional_t<(std::numeric_limits<floating_t<L>>::digits <=
                                   std::numeric_limits<floating_t<R>>::digits),
                                  floating_t<L>, floating_t<R>>;
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

// Replaces EXPECT_NEAR_ABS(VAL1, VAL2, TOL): passes if |lhs - rhs| <
// |tolerance|, in the less precise of the operands' floating-point types.
// A failure prints
//
//   Expected |lhs - rhs| < 1e-08
//     |lhs - rhs|: 0.5
//     lhs: 2.0
//     rhs: 2.5
template <class L, class R, class T>
  requires std::is_arithmetic_v<L> && std::is_arithmetic_v<R> &&
           std::is_arithmetic_v<T>
void expect_near_abs(
    L lhs, R rhs, T tolerance,
    std::source_location loc = std::source_location::current()) {
  using F = detail::near_t<L, R>;
  const F difference = std::abs(static_cast<F>(lhs) - static_cast<F>(rhs));
  const F limit = std::abs(static_cast<F>(tolerance));
  if (difference < limit)
    return;
  const detail::operand_texts t = detail::texts_of(loc, lhs, rhs);
  const std::string distance = "|" + t.joined("-", detail::additive_prec) + "|";
  detail::print_failure(loc, distance + " < " + detail::value_text(limit));
  detail::print_value(distance, detail::value_text(difference));
  detail::print_value(t.lhs_name, t.lhs_value);
  detail::print_value(t.rhs_name, t.rhs_value);
  detail::increase_error_count();
}

} // namespace ihm

#endif // REFLECTION_AD_IHM_IHM_HPP
