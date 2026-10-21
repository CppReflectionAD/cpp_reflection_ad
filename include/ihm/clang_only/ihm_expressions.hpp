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
// the others print `lhs` and `rhs`. Each check is found by its line, so put
// at most one check per line. Declare one variable per declaration in that
// body (see AGENTS.md).
//
// The text is rebuilt from the expression, not copied from the source, so it
// can differ from what the macros print: parentheses are only those the
// operators' precedence needs, literals are respelled (`0x10` prints as `16`,
// `1.e3` as `1000.0`), names lose their qualification (`std::abs(x)` prints as
// `abs(x)`), casts are dropped, and member accesses and member function calls
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

template <class F> consteval F power_of_ten(int n) {
  F p = 1;
  for (int i = 0; i < n; ++i)
    p *= 10;
  return p;
}

// `digits` times 10^`exponent`, as a C++ floating literal: fixed notation
// with at least one decimal (`2.0`, `0.125`), or scientific for very large
// or small values (`1e-13`).
consteval std::string decimal_text(unsigned long long digits, int exponent) {
  std::string d = integer_text(digits);
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

// The shortest decimal that reads back as `value`, searched in double, or in
// long double for a long double, which can be more precise. Up to 19
// significant digits.
template <class T> consteval std::string floating_text(T value) {
  using F = std::common_type_t<T, double>;
  if (value == 0)
    return "0.0";
  const bool negative = value < 0;
  const T magnitude = negative ? -value : value;
  const F x = magnitude;
  int leading = 0;
  for (F y = x; y >= 10; y /= 10)
    ++leading;
  for (F y = x; y < 1; y *= 10)
    --leading;
  for (int precision = 1; precision <= std::numeric_limits<F>::max_digits10;
       ++precision) {
    const int exponent = leading - precision + 1;
    if (exponent > 300 || exponent < -300)
      break;
    const F scale = power_of_ten<F>(exponent < 0 ? -exponent : exponent);
    const F scaled = exponent < 0 ? x * scale : x / scale;
    if (!(scaled < 1e19))
      break;
    const unsigned long long digits =
        static_cast<unsigned long long>(scaled + F(0.5));
    const F back = exponent < 0 ? static_cast<F>(digits) / scale
                                : static_cast<F>(digits) * scale;
    if (static_cast<T>(back) == magnitude)
      return (negative ? "-" : "") + decimal_text(digits, exponent);
  }
  return "<literal>";
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
// its conversion to `const char *` can.
consteval std::string literal_text(peeled p) {
  if (m::is_cast(p.innermost)) {
    const m::info v = m::constant_of(p.innermost);
    if (is_char_pointer(m::type_of(v)))
      return quoted(m::extract<const char *>(v));
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
    op = m::op_equals_equals; // `a == b` rewritten as `b == a`
    comparison = form;
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

consteval bool is_string_literal(m::info operand, m::info type) {
  return m::is_array_type(type) && is_plain_literal(strip_casts(operand));
}

// The text of a check's operand of type `type`, or null for a string literal.
// The literal binds to the check's reference parameter without converting to
// a pointer, so literal_text can't evaluate it; report_failure prints its
// value instead.
consteval const char *operand_text(m::info operand, m::info type) {
  if (is_string_literal(operand, type))
    return nullptr;
  return std::define_static_string(text_of(operand));
}

// The precedence of the outermost operator in operand_text().
consteval int operand_precedence(m::info operand, m::info type) {
  return is_string_literal(operand, type) ? primary_prec : print(operand).prec;
}

// The text of a check call's operands, and where the call is: the same file
// and line as the std::source_location it passes at run time.
consteval check_site site_of(m::info call) {
  const std::source_location loc = m::source_location_of(call);
  const auto ops = m::operands_of(call); // [callee, lhs, rhs, ..., loc]
  const auto types = m::template_arguments_of(m::callee_of(call)); // [L, R...]
  return {std::define_static_string(std::string_view(loc.file_name())),
          static_cast<unsigned>(loc.line()),
          operand_text(ops[1], types[0]),
          operand_text(ops[2], types[1]),
          operand_precedence(ops[1], types[0]),
          operand_precedence(ops[2], types[1])};
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
    if (s.line == loc.line() && std::string_view(s.file) == loc.file_name())
      return {s.lhs, s.rhs, s.lhs_prec, s.rhs_prec};
  return {};
}
} // namespace detail

// Runs the test function `Fn` (a `void()` function), its failed checks named
// by their operands' source text, and returns result().
template <std::meta::info Fn> int run_with_text() {
  static constexpr auto sites =
      std::define_static_array(detail::collect_sites(Fn));
  const std::span<const detail::check_site> previous_sites =
      detail::active_sites;
  const auto previous_names = detail::operand_names_at;
  detail::active_sites = sites;
  detail::operand_names_at = &detail::names_from_sites;
  [:Fn:]();
  detail::active_sites = previous_sites;
  detail::operand_names_at = previous_names;
  return result();
}

} // namespace ihm

#endif // REFLECTION_AD_IHM_IHM_EXPRESSIONS_HPP
