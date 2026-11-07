#ifndef REFLECTION_AD_IHM_IHM_EXPRESSIONS_HPP
#define REFLECTION_AD_IHM_IHM_EXPRESSIONS_HPP

// An optional layer over ihm/ihm.hpp: a failed check names its
// operands by their source text, which test_simple_include.hpp gets from the
// preprocessor (`#VAL`), instead of `lhs` and `rhs`. The text is read from
// the test function's body with statement and expression reflection.
//
//   void my_test() {
//     double a = 1.0;
//     double b = 2.0;
//     ihm::expect_equal(a + a, b);
//   }
//   int main() { return ihm::run_with_text<^^my_test>(); }
//
// prints, like EXPECT_EQUAL, then the operands' values,
//
//   my_test.cpp:4 Failure
//   Expected a + a == b
//     a + a: 2.0
//     b: 1.0
//
// Only checks written directly in the body of the function given to
// run_with_text() (including nested blocks, `if`s and loops) have their text;
// the others print `lhs` and `rhs`. Each check is found by its line and
// column, so several checks can share a line. Declare one variable per
// declaration in that body (see AGENTS.md).
//
// The text is rebuilt from the expression, not copied from the source, so it
// can differ from what the macros print: parentheses are only those the
// operators' precedence needs, a literal operand prints as its value (`3u` as
// `3`, `1e16` as `1e+16`) and literals inside an operand are respelled (`0x10`
// prints as `16`, `1.e3` as `1000.0`), names lose their qualification
// (`std::abs(x)` prints as `abs(x)`), casts are dropped, and member accesses
// and member function calls
// print as `<expr>` (the reflection gives no member names; known gap
// tests/known_gap/clang/member_access_declaration.cpp). A string literal
// is only supported as a whole operand of a check, or passed to a parameter of
// type `const char *`; anywhere else it doesn't compile.
//
// Clang only: it uses the clang fork's expression-reflection API
// (is_expression_statement, callee_of, ...), which the GCC fork's <meta>
// doesn't have, and the stand-ins in gap.hpp for the parts the clang fork
// lacks too.

#include <ihm/clang_only/gap.hpp>
#include <ihm/ihm.hpp>

#include <cstddef>
#include <limits>
#include <meta>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ihm {

namespace detail {
namespace m = std::meta;

// One check call found in a test body, with its operands' source text and
// the precedence of their outermost operators. A null text marks a string
// literal operand, whose text is its value.
struct check_site {
  const char *file;
  unsigned line;
  unsigned column;
  const char *lhs;
  const char *rhs;
  int lhs_prec;
  int rhs_prec;
};

// The checks of the test body run_with_text() is running.
inline std::span<const check_site> active_sites;

// An expression without its conversions and single-argument copies (see
// ad::detail::stripCasts), and the innermost of those, if any.
struct peeled {
  m::info expr;
  m::info innermost;
};

consteval peeled peel(m::info e) {
  m::info innermost{};
  while (m::is_expression(e) &&
         (m::is_cast(e) ||
          (m::is_construct(e) && m::operands_of(e).size() == 1))) {
    innermost = e;
    e = m::operands_of(e).front();
  }
  return {e, innermost};
}

consteval m::info strip_casts(m::info e) { return peel(e).expr; }

// A literal has kind 0 and no operands. Kind 0 is also the fork's kind for
// expressions it doesn't classify, such as a rewritten `a != b`.
consteval bool is_plain_literal(m::info e) {
  return m::is_literal(e) && m::operands_of(e).empty();
}

// Whether `a` starts after `b` in the source.
consteval bool starts_after(m::info a, m::info b) {
  const std::source_location la = m::source_location_of(a);
  const std::source_location lb = m::source_location_of(b);
  return la.line() != lb.line() ? la.line() > lb.line()
                                : la.column() > lb.column();
}

consteval bool same_start(m::info a, m::info b) {
  return !starts_after(a, b) && !starts_after(b, a);
}

// Literal spelling (string and character literals are quoted with quoted()).

template <class T> consteval std::string integer_text(T value) {
  unsigned long long magnitude = static_cast<unsigned long long>(value);
  bool negative = false;
  if constexpr (std::is_signed_v<T>) {
    negative = value < 0;
    if (negative)
      magnitude = 0ULL - magnitude;
  }
  std::string s;
  do {
    s.insert(s.begin(), static_cast<char>('0' + magnitude % 10));
    magnitude /= 10;
  } while (magnitude != 0);
  return negative ? "-" + s : s;
}

// A non-negative integer of any size, as base-10^9 digits, least significant
// first.
using big_decimal = std::vector<unsigned>;
inline constexpr unsigned big_base = 1'000'000'000;

// n *= factor, for a factor up to 2^31.
consteval void multiply(big_decimal &n, unsigned factor) {
  unsigned long long carry = 0;
  for (unsigned &limb : n) {
    carry += static_cast<unsigned long long>(limb) * factor;
    limb = static_cast<unsigned>(carry % big_base);
    carry /= big_base;
  }
  for (; carry != 0; carry /= big_base)
    n.push_back(static_cast<unsigned>(carry % big_base));
}

// n *= factor^count, for a factor of 2 or 5, a few powers at a time.
consteval void multiply_power(big_decimal &n, unsigned factor, int count) {
  while (count > 0) {
    unsigned f = 1;
    for (; count > 0 && f <= (1u << 31) / factor; --count)
      f *= factor;
    multiply(n, f);
  }
}

// n += delta, for a delta in -2..2 that leaves n non-negative.
consteval void add(big_decimal &n, int delta) {
  for (std::size_t i = 0; delta != 0; ++i) {
    if (i == n.size())
      n.push_back(0);
    const long long sum = static_cast<long long>(n[i]) + delta;
    delta = sum < 0 ? -1 : sum >= big_base ? 1 : 0;
    n[i] =
        static_cast<unsigned>(sum - delta * static_cast<long long>(big_base));
  }
}

// The decimal digits of `n`, with leading zeros up to a multiple of 9.
consteval std::string decimal_digits(const big_decimal &n) {
  std::string s(9 * n.size(), '0');
  std::size_t end = s.size();
  for (unsigned limb : n)
    for (int i = 0; i < 9; ++i, limb /= 10)
      s[--end] = static_cast<char>('0' + limb % 10);
  return s;
}

// `digits` times 10^`exponent`, as a C++ floating literal: fixed notation
// with at least one decimal (`2.0`, `0.125`), or scientific for very large
// or small values (`1e-13`).
consteval std::string decimal_text(std::string d, int exponent) {
  d.erase(0, d.find_first_not_of('0'));
  while (d.size() > 1 && d.back() == '0') {
    d.pop_back();
    ++exponent;
  }
  const int size = static_cast<int>(d.size());
  const int leading = exponent + size - 1;
  if (leading < -4 || leading > 15) {
    std::string s = d.substr(0, 1);
    if (size > 1)
      s += "." + d.substr(1);
    return s + "e" + integer_text(leading);
  }
  if (exponent >= 0)
    return d + std::string(static_cast<std::size_t>(exponent), '0') + ".0";
  const int point = size + exponent;
  if (point > 0)
    return d.substr(0, static_cast<std::size_t>(point)) + "." +
           d.substr(static_cast<std::size_t>(point));
  return "0." + std::string(static_cast<std::size_t>(-point), '0') + d;
}

// The shortest decimal that reads back as `value`. Found exactly, with
// integers: the value and the bounds of the decimals that round to it are
// written out in full, and the shortest decimal between the bounds wins (the
// nearer to the value if there are two).
template <class T> consteval std::string floating_text(T value) {
  constexpr int digits = std::numeric_limits<T>::digits;
  constexpr int min_exponent = std::numeric_limits<T>::min_exponent - digits;
  if (value == 0)
    return "0.0";
  const bool negative = value < 0;
  T x = negative ? -value : value;
  if (!(x <= std::numeric_limits<T>::max())) // infinity or NaN
    return "<literal>";
  // value = ±x * 2^exponent, x an integer below 2^digits.
  T top = 1;
  for (int i = 0; i < digits; ++i)
    top *= 2;
  int exponent = 0;
  for (; x >= top; x /= 2)
    ++exponent;
  for (; x < top / 2; x *= 2)
    --exponent;
  for (; exponent < min_exponent; x /= 2) // a subnormal
    ++exponent;
  // Beyond a double's exponents (a wider long double) the digits would take
  // too many steps to write out at compile time.
  if (exponent > 1100 || exponent < -1100)
    return "<literal>";
  // At a power of two the next value down is nearer than the next value up.
  const bool narrower_below = x == top / 2 && exponent > min_exponent;
  big_decimal mantissa{0};
  for (T bit = top / 2; bit >= 1; bit /= 2) {
    multiply(mantissa, 2);
    if (x >= bit) {
      x -= bit;
      add(mantissa, 1);
    }
  }
  // A decimal exactly halfway to the next value rounds to the even mantissa.
  const bool even = mantissa[0] % 2 == 0;
  multiply(mantissa, 4);
  // In units of 2^(exponent - 2), the value is 4 * mantissa, and the bounds
  // are halfway to the next values down and up. 2^-n is 5^n / 10^n.
  const int twos = exponent - 2;
  const int decimal_exponent = twos < 0 ? twos : 0;
  const auto digits_of = [&](int offset) {
    big_decimal n = mantissa;
    add(n, offset);
    if (twos >= 0)
      multiply_power(n, 2, twos);
    else
      multiply_power(n, 5, -twos);
    return decimal_digits(n);
  };
  std::string low = digits_of(narrower_below ? -1 : -2);
  std::string exact = digits_of(0);
  const std::string high = digits_of(2);
  low.insert(0, high.size() - low.size(), '0');
  exact.insert(0, high.size() - exact.size(), '0');
  const auto reads_back = [&](const std::string &d) {
    return (low < d || (even && low == d)) && (d < high || (even && d == high));
  };
  for (std::size_t end = exact.find_first_not_of('0') + 1;; ++end) {
    const std::string down =
        exact.substr(0, end) + std::string(high.size() - end, '0');
    std::string up = down;
    std::size_t i = end;
    for (; i > 0 && up[i - 1] == '9'; --i)
      up[i - 1] = '0';
    if (i > 0)
      ++up[i - 1];
    const bool up_reads_back = i > 0 && reads_back(up);
    const bool down_reads_back = reads_back(down);
    const bool up_nearer = end < exact.size() && exact[end] >= '5';
    const std::string sign = negative ? "-" : "";
    if (up_reads_back && (up_nearer || !down_reads_back))
      return sign + decimal_text(up, decimal_exponent);
    if (down_reads_back) // at the latest when `down` is all of `exact`
      return sign + decimal_text(down, decimal_exponent);
  }
}

// A value reflection as a C++ literal.
consteval std::string value_text(m::info v) {
  const m::info t = m::dealias(m::remove_cv(m::type_of(v)));
  if (t == ^^bool)
    return m::extract<bool>(v) ? "true" : "false";
  if (t == ^^char) {
    const char c = m::extract<char>(v);
    return quoted(std::string_view(&c, 1), '\'');
  }
  if (t == ^^int)
    return integer_text(m::extract<int>(v));
  if (t == ^^unsigned)
    return integer_text(m::extract<unsigned>(v)) + "u";
  if (t == ^^long)
    return integer_text(m::extract<long>(v)) + "l";
  if (t == ^^unsigned long)
    return integer_text(m::extract<unsigned long>(v)) + "ul";
  if (t == ^^long long)
    return integer_text(m::extract<long long>(v)) + "ll";
  if (t == ^^unsigned long long)
    return integer_text(m::extract<unsigned long long>(v)) + "ull";
  if (t == ^^float)
    return floating_text(m::extract<float>(v)) + "f";
  if (t == ^^double)
    return floating_text(m::extract<double>(v));
  if (t == ^^long double)
    return floating_text(m::extract<long double>(v)) + "L";
  if (t == ^^decltype(nullptr))
    return "nullptr";
  return std::string(m::display_string_of(v));
}

consteval bool is_char_pointer(m::info t) {
  return m::is_pointer_type(t) &&
         m::dealias(m::remove_cv(m::remove_pointer(t))) == ^^char;
}

// The literal `p.expr`. A string literal can't be evaluated by itself, but
// its conversion to `const char *` can. A null pointer literal converted to
// `const char *` (`p == nullptr`, `p == 0`) isn't a string: it prints as
// written.
consteval std::string literal_text(peeled p) {
  if (m::is_cast(p.innermost)) {
    const m::info v = m::constant_of(p.innermost);
    if (is_char_pointer(m::type_of(v))) {
      const char *s = m::extract<const char *>(v);
      if (s != nullptr)
        return quoted(s);
    }
  }
  return value_text(m::constant_of(p.expr));
}

// Expression text, with the parentheses that operator precedence needs (see
// `precedence` in ihm/ihm.hpp).

consteval int binary_precedence(m::operators op) {
  switch (op) {
  case m::op_star:
  case m::op_slash:
  case m::op_percent:
    return multiplicative_prec;
  case m::op_plus:
  case m::op_minus:
    return additive_prec;
  case m::op_less_less:
  case m::op_greater_greater:
    return shift_prec;
  case m::op_spaceship:
    return spaceship_prec;
  case m::op_less:
  case m::op_greater:
  case m::op_less_equals:
  case m::op_greater_equals:
    return relational_prec;
  case m::op_equals_equals:
  case m::op_exclamation_equals:
    return equality_prec;
  case m::op_ampersand:
    return bitand_prec;
  case m::op_caret:
    return xor_prec;
  case m::op_pipe:
    return bitor_prec;
  case m::op_ampersand_ampersand:
    return and_prec;
  case m::op_pipe_pipe:
    return or_prec;
  case m::op_comma:
    return comma_prec;
  default: // =, +=, ...
    return assignment_prec;
  }
}

struct expr_text {
  std::string text;
  int prec;
};

consteval expr_text print(m::info e);

// The text of `e` as an operand that needs precedence `min_prec`.
consteval std::string print_at(m::info e, int min_prec) {
  const expr_text t = print(e);
  return t.prec < min_prec ? "(" + t.text + ")" : t.text;
}

consteval expr_text binary_text(m::operators op, m::info lhs, m::info rhs) {
  const int p = binary_precedence(op);
  const bool right_assoc = p == assignment_prec;
  return {print_at(lhs, right_assoc ? p + 1 : p) + " " +
              std::string(m::symbol_of(op)) + " " +
              print_at(rhs, right_assoc ? p : p + 1),
          p};
}

consteval expr_text unary_text(m::operators op, m::info operand, bool postfix) {
  const std::string symbol(m::symbol_of(op));
  if (postfix)
    return {print_at(operand, postfix_prec) + symbol, postfix_prec};
  std::string o = print_at(operand, unary_prec);
  // -(-a), not --a
  if ((symbol == "-" || symbol == "+") && o.front() == symbol.back())
    o = "(" + o + ")";
  return {symbol + o, unary_prec};
}

consteval std::string arguments_text(const std::vector<m::info> &args,
                                     std::size_t first) {
  std::string s = "(";
  bool any = false;
  for (std::size_t i = first; i < args.size(); ++i) {
    if (m::source_location_of(args[i]).line() == 0)
      continue; // a default argument
    s += (any ? ", " : "") + print_at(args[i], assignment_prec);
    any = true;
  }
  return s + ")";
}

consteval std::string callee_name(m::info callee) {
  if (m::has_template_arguments(callee))
    callee = m::template_of(callee);
  if (m::has_identifier(callee))
    return std::string(m::identifier_of(callee));
  return "<function>";
}

// A call of an overloaded operator, written as the operator.
consteval expr_text operator_call_text(m::operators op,
                                       const std::vector<m::info> &ops) {
  const std::size_t args = ops.size() - 1; // ops[0] is the callee
  if (op == m::op_square_brackets && args == 2)
    return {print_at(ops[1], postfix_prec) + "[" + print(ops[2]).text + "]",
            postfix_prec};
  if (op == m::op_parentheses)
    return {print_at(ops[1], postfix_prec) + arguments_text(ops, 2),
            postfix_prec};
  if ((op == m::op_plus_plus || op == m::op_minus_minus) && args == 2)
    return unary_text(op, ops[1], true); // x++ passes a dummy int
  if (args == 2)
    return binary_text(op, ops[1], ops[2]);
  if (args == 1)
    return unary_text(op, ops[1], false);
  return {"<expr>", primary_prec};
}

consteval expr_text call_text(m::info e) {
  const std::vector<m::info> ops = m::operands_of(e); // [callee, args...]
  const m::info callee = m::callee_of(e);
  if (callee == m::info{})
    return {"<expr>", primary_prec}; // a member function call
  if (m::is_operator_function(callee))
    return operator_call_text(m::operator_of(callee), ops);
  return {callee_name(callee) + arguments_text(ops, 1), postfix_prec};
}

// The two operands of a comparison `a == b` or `a <=> b`, built in or
// overloaded.
consteval std::vector<m::info> comparison_operands(m::info e) {
  std::vector<m::info> ops = m::operands_of(e);
  if (m::is_function_call(e))
    ops.erase(ops.begin());
  return ops;
}

// C++20 rewrites `a != b` as `!(a == b)` and `a < b` as `(a <=> b) < 0`,
// possibly with the operands swapped. Recover what was written.
consteval expr_text rewritten_text(m::info e) {
  const m::info form = strip_casts(m::operands_of(e).front());
  m::operators op{};
  m::info comparison{};
  if (m::is_unary_operator(form) &&
      m::expression_operator_of(form) == m::op_exclamation) {
    op = m::op_exclamation_equals;
    comparison = strip_casts(m::operands_of(form).front());
  } else if (m::is_binary_operator(form)) {
    op = m::expression_operator_of(form);
    const std::vector<m::info> sides = m::operands_of(form);
    comparison = strip_casts(
        is_plain_literal(strip_casts(sides[0])) ? sides[1] : sides[0]);
  } else if (m::is_function_call(form)) {
    const m::info callee = m::callee_of(form);
    if (!m::is_operator_function(callee))
      return {"<expr>", primary_prec};
    const m::operators called_op = m::operator_of(callee);
    if (called_op == m::op_equals_equals) {
      op = called_op; // `a == b` rewritten as `b == a`
      comparison = form;
    } else if (called_op == m::op_less || called_op == m::op_less_equals ||
               called_op == m::op_greater ||
               called_op == m::op_greater_equals) {
      op = called_op;
      const std::vector<m::info> sides = comparison_operands(form);
      if (sides.size() != 2)
        return {"<expr>", primary_prec};
      comparison = strip_casts(
          is_plain_literal(strip_casts(sides[0])) ? sides[1] : sides[0]);
    } else {
      return {"<expr>", primary_prec};
    }
  }
  if (comparison == m::info{})
    return {"<expr>", primary_prec};
  const std::vector<m::info> ops = comparison_operands(comparison);
  if (ops.size() != 2)
    return {"<expr>", primary_prec};
  return starts_after(ops[0], ops[1]) ? binary_text(op, ops[1], ops[0])
                                      : binary_text(op, ops[0], ops[1]);
}

consteval expr_text print(m::info e) {
  const peeled p = peel(e);
  e = p.expr;
  if (m::is_variable_reference(e))
    return {std::string(m::identifier_of(m::declaration_of(e))), primary_prec};
  if (is_plain_literal(e))
    return {literal_text(p), primary_prec};
  if (m::is_literal(e))
    return rewritten_text(e);
  if (m::is_unary_operator(e)) {
    const m::info operand = m::operands_of(e).front();
    return unary_text(m::expression_operator_of(e), operand,
                      same_start(e, operand));
  }
  if (m::is_binary_operator(e) || m::expression_operator_of(e) == m::op_comma) {
    const std::vector<m::info> ops = m::operands_of(e);
    return binary_text(m::expression_operator_of(e), ops[0], ops[1]);
  }
  if (m::is_function_call(e))
    return call_text(e);
  if (m::is_conditional_operator(e))
    return {print_at(m::condition_of(e), or_prec) + " ? " +
                print_at(m::true_expression_of(e), comma_prec + 1) + " : " +
                print_at(m::false_expression_of(e), assignment_prec),
            assignment_prec};
  if (gap::expression_operator_of(e) == m::op_square_brackets) {
    const std::vector<m::info> ops = m::operands_of(e);
    return {print_at(ops[0], postfix_prec) + "[" + print(ops[1]).text + "]",
            postfix_prec};
  }
  return {"<expr>", primary_prec};
}

// Source-like text of an expression, in place of the macros' `#VAL`.
consteval std::string text_of(m::info e) { return print(e).text; }

// Whether `call` calls one of the checks.
consteval bool is_check(m::info call) {
  const m::info callee = m::callee_of(call);
  if (!m::is_function(callee) || !m::has_template_arguments(callee))
    return false;
  const m::info check = m::template_of(callee);
  return check == ^^expect_equal || check == ^^expect_not_equal ||
         check == ^^expect_near_abs;
}

consteval bool is_whole_literal(m::info operand) {
  return is_plain_literal(strip_casts(operand));
}

// The text of a check's operand, or null for a literal. A literal operand is
// named by its value's text, printed at run time (`3u` as `3`, `1e300` as
// `1e+300`), so it gets no value line of its own; a string literal also can't
// be evaluated here, as it binds to the check's reference parameter without
// converting to a pointer.
consteval const char *operand_text(m::info operand) {
  if (is_whole_literal(operand))
    return nullptr;
  return std::define_static_string(text_of(operand));
}

// The precedence of the outermost operator in operand_text().
consteval int operand_precedence(m::info operand) {
  return is_whole_literal(operand) ? primary_prec : print(operand).prec;
}

// The text of a check call's operands, and where the call is: the same file,
// line and column as the std::source_location it passes at run time.
consteval check_site site_of(m::info call) {
  const std::source_location loc = m::source_location_of(call);
  const auto ops = m::operands_of(call); // [callee, lhs, rhs, ..., loc]
  return {std::define_static_string(std::string_view(loc.file_name())),
          static_cast<unsigned>(loc.line()),
          static_cast<unsigned>(loc.column()),
          operand_text(ops[1]),
          operand_text(ops[2]),
          operand_precedence(ops[1]),
          operand_precedence(ops[2])};
}

consteval void collect_sites(m::info stmt, std::vector<check_site> &out,
                             int depth) {
  if (depth > 64)
    throw "ihm: statements nested too deeply";
  for (m::info s : m::statements_of(stmt)) {
    if (!m::is_expression_statement(s)) {
      collect_sites(s, out, depth + 1);
      continue;
    }
    const m::info e = strip_casts(gap::expression_of(s));
    if (m::is_function_call(e) && is_check(e))
      out.push_back(site_of(e));
  }
}

consteval std::vector<check_site> collect_sites(m::info fn) {
  std::vector<check_site> out;
  collect_sites(m::body_of(fn), out, 0);
  return out;
}

// The source text of the operands of the check at `loc`, if it is one of
// active_sites.
inline operand_names names_from_sites(const std::source_location &loc) {
  for (const check_site &s : active_sites)
    if (s.line == loc.line() && s.column == loc.column() &&
        std::string_view(s.file) == loc.file_name())
      return {s.lhs, s.rhs, s.lhs_prec, s.rhs_prec};
  return {};
}
} // namespace detail

// Runs the test function `Fn` (a `void()` function), its failed checks named
// by their operands' source text, and returns result().
template <std::meta::info Fn> int run_with_text() {
  static constexpr auto sites =
      std::define_static_array(detail::collect_sites(Fn));
  // Puts back the previous sites and names, also if Fn throws.
  struct restore {
    std::span<const detail::check_site> previous_sites = detail::active_sites;
    decltype(detail::operand_names_at) previous_names =
        detail::operand_names_at;
    ~restore() {
      detail::active_sites = previous_sites;
      detail::operand_names_at = previous_names;
    }
  } guard;
  detail::active_sites = sites;
  detail::operand_names_at = &detail::names_from_sites;
  [:Fn:]();
  return result();
}

} // namespace ihm

#endif // REFLECTION_AD_IHM_IHM_EXPRESSIONS_HPP
