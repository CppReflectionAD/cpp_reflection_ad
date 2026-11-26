// assert.cpp - the self-explaining assertion of P4395 section 2.2, three ways.
//
// MY_ASSERT(in_range(x, lo, hi)) is a macro: it wraps its argument in a
// lambda, [&] { return in_range(x, lo, hi); }, and reflects on the body of
// that lambda's call operator. The variables' values are read back from the
// lambda's captures.
//
// my_assert<^^in_range>(x, lo, hi) uses no macro: it is handed the
// reflection of the function explicitly, and gets the values as arguments.
// Without a macro the reflection has to be passed like this: a function
// called as my_assert(in_range(x, lo, hi)) would only receive a bool.
//
// my_assert2<in_range>(x, lo, hi) is the same without the ^^: it takes the
// function itself as a template argument and gets the reflection back with
// reflect_function.
//
// In all three, the expression's tree and its text are built at compile time,
// and calls are followed into the callee's body. The two forms without a macro
// get the caller's line from a defaulted std::source_location argument, but
// can only show the callee's parameter names, not the caller's variable names.

#include <algorithm>
#include <cstdio>
#include <meta>
#include <source_location>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pa {

namespace m = std::meta;
using m::info;

// --- the tree, flattened at compile time ------------------------------------

struct node {
  enum kind_t { variable, parameter, binary, call } kind;
  m::operators op{};
  int child[4] = {-1, -1, -1, -1}; // binary: operands; call: arguments
  int children = 0;
  int slot = -1; // variable: capture number; parameter: position
  int body = -1; // call: root of the callee's body
  int depth = 0; // calls deep, for indentation
  bool boolean = false;
  const char *text = nullptr;
  const char *name = nullptr;                   // call: callee
  const char *const *parameter_names = nullptr; // call
};

consteval info peel(info e) {
  while (m::is_cast(e))
    e = m::operands_of(e).front();
  return e;
}

consteval info returned_expression(info fn) {
  for (info s : m::statements_of(m::body_of(fn)))
    if (m::is_return_statement(s))
      return peel(m::return_value_of(s));
  throw "no return statement";
}

struct builder {
  std::vector<node> nodes;
  std::vector<std::string> variables; // in order of first use = capture order
  std::string text;                   // text of the last node built

  // Append the nodes for `e`, children first. `fn` is the function whose
  // body `e` is in, or null for the asserted expression.
  consteval int flatten(info e, info fn, int depth) {
    using enum m::operators;
    e = peel(e);
    node n{};
    n.depth = depth;
    if (m::is_binary_operator(e)) {
      n.kind = node::binary;
      n.op = m::expression_operator_of(e);
      std::vector<info> ops = m::operands_of(e);
      n.child[0] = flatten(ops[0], fn, depth);
      std::string lhs = text;
      n.child[1] = flatten(ops[1], fn, depth);
      n.children = 2;
      text = lhs + " " + std::string(m::symbol_of(n.op)) + " " + text;
      n.boolean = n.op == op_less || n.op == op_less_equals ||
                  n.op == op_greater || n.op == op_greater_equals ||
                  n.op == op_equals_equals || n.op == op_exclamation_equals ||
                  n.op == op_ampersand_ampersand || n.op == op_pipe_pipe;
    } else if (m::is_function_call(e)) {
      n.kind = node::call;
      info callee = m::callee_of(e);
      std::string call = std::string(m::identifier_of(callee)) + "(";
      std::vector<info> ops = m::operands_of(e); // [callee, args...]
      for (std::size_t i = 1; i < ops.size(); ++i) {
        n.child[n.children++] = flatten(ops[i], fn, depth);
        call += (i > 1 ? ", " : "") + text;
      }
      n.name = std::define_static_string(m::identifier_of(callee));
      n.boolean = m::return_type_of(callee) == ^^bool;
      std::vector<const char *> names;
      for (info p : m::parameters_of(callee))
        names.push_back(std::define_static_string(m::identifier_of(p)));
      n.parameter_names = std::define_static_array(names).data();
      n.body = flatten(returned_expression(callee), callee, depth + 1);
      text = call + ")";
    } else if (m::is_variable_reference(e)) {
      text = std::string(m::identifier_of(m::declaration_of(e)));
      if (fn == info{}) { // a variable of the asserting scope
        n.kind = node::variable;
        n.slot = int(variables.size());
        for (std::size_t i = 0; i < variables.size(); ++i)
          if (variables[i] == text)
            n.slot = int(i);
        if (n.slot == int(variables.size()))
          variables.push_back(text);
      } else { // a parameter of the function being followed
        n.kind = node::parameter;
        std::vector<info> params = m::parameters_of(fn);
        for (std::size_t i = 0; i < params.size(); ++i)
          if (m::identifier_of(params[i]) == text)
            n.slot = int(i);
      }
    } else {
      throw "my_assert: unsupported expression";
    }
    n.text = std::define_static_string(text);
    nodes.push_back(n);
    return int(nodes.size()) - 1;
  }
};

// The lambda's call operator.
consteval info call_operator(info closure) {
  for (info f : m::members_of(closure, m::access_context::unchecked()))
    if (m::is_function(f) && m::is_operator_function(f))
      return f;
  throw "not a lambda";
}

consteval builder build(info closure) {
  builder b;
  b.flatten(returned_expression(call_operator(closure)), {}, 0);
  return b;
}

consteval std::vector<const char *> variable_names_of(info closure) {
  std::vector<const char *> names;
  for (const std::string &v : build(closure).variables)
    names.push_back(std::define_static_string(v));
  return names;
}

template <class L>
constexpr std::span<const node> tree =
    std::define_static_array(build(^^L).nodes);
template <class L>
constexpr std::span<const char *const> variable_names =
    std::define_static_array(variable_names_of(^^L));
template <class L>
constexpr std::span<const info> captures = std::define_static_array(
    m::nonstatic_data_members_of(^^L, m::access_context::unchecked()));

template <class L, std::size_t... I>
void read_captures(const L &l, double *out, std::index_sequence<I...>) {
  ((out[I] = double(l.[:captures<L>[I]:])), ...);
}

// --- the run-time interpreter -----------------------------------------------

struct trace {
  double value[64];
  bool evaluated[64] = {};
};

inline double eval(std::span<const node> t, int i, const double *vars,
                   const double *params, trace &tr) {
  using enum m::operators;
  const node &n = t[std::size_t(i)];
  auto sub = [&](int c) { return eval(t, n.child[c], vars, params, tr); };
  double v = 0;
  if (n.kind == node::variable)
    v = vars[n.slot];
  else if (n.kind == node::parameter)
    v = params[n.slot];
  else if (n.kind == node::call) {
    double args[4] = {};
    for (int c = 0; c < n.children; ++c)
      args[c] = sub(c);
    v = eval(t, n.body, vars, args, tr);
  } else if (n.op == op_ampersand_ampersand)
    v = sub(0) && sub(1);
  else if (n.op == op_pipe_pipe)
    v = sub(0) || sub(1);
  else {
    const double x = sub(0);
    const double y = sub(1);
    v = n.op == op_plus             ? x + y
        : n.op == op_minus          ? x - y
        : n.op == op_star           ? x * y
        : n.op == op_slash          ? x / y
        : n.op == op_less           ? x < y
        : n.op == op_less_equals    ? x <= y
        : n.op == op_greater        ? x > y
        : n.op == op_greater_equals ? x >= y
        : n.op == op_equals_equals  ? x == y
                                    : x != y;
  }
  tr.value[i] = v;
  tr.evaluated[i] = true;
  return v;
}

// --- the report -------------------------------------------------------------

inline void print_value(const node &n, double v) {
  if (n.boolean)
    std::printf("%s", v != 0 ? "true" : "false");
  else
    std::printf("%g", v);
}

// Every sub-expression of node i, children first; a call shows its body.
inline void report(std::span<const node> t, int i, const trace &tr, bool root,
                   int width) {
  const node &n = t[std::size_t(i)];
  if (n.kind == node::variable || n.kind == node::parameter)
    return;
  for (int c = 0; c < n.children; ++c)
    report(t, n.child[c], tr, false, width);
  if (n.kind == node::call && tr.evaluated[i]) {
    std::printf("    %*sinside %s(", 2 * n.depth, "", n.name);
    for (int c = 0; c < n.children; ++c) {
      std::printf("%s%s = ", c ? ", " : "", n.parameter_names[c]);
      print_value(t[std::size_t(n.child[c])], tr.value[n.child[c]]);
    }
    std::printf("):\n");
    report(t, n.body, tr, false, width);
  }
  if (root)
    return;
  std::printf("    %*s%-*s  ", 2 * n.depth, "", width - 2 * n.depth, n.text);
  if (tr.evaluated[i])
    print_value(n, tr.value[i]);
  else
    std::printf("(not evaluated)");
  std::printf("\n");
}

template <class L>
bool check(const L &expr,
           std::source_location where = std::source_location::current()) {
  if (expr())
    return true;
  constexpr std::span<const node> t = tree<L>;
  constexpr std::size_t nvars = variable_names<L>.size();
  static_assert(t.size() <= 64);
  static_assert(captures<L>.size() == nvars, "only local variables");
  double vars[nvars + 1] = {};
  read_captures(expr, vars, std::make_index_sequence<nvars>{});
  trace tr;
  const int root = int(t.size()) - 1;
  eval(t, root, vars, nullptr, tr);

  std::printf("MY_ASSERT(%s) failed at line %u\n", t.back().text,
              unsigned(where.line()));
  std::printf("    where");
  for (std::size_t i = 0; i < nvars; ++i)
    std::printf("%s %s = %g", i ? "," : "", variable_names<L>[i], vars[i]);
  std::printf("\n");
  int width = 0; // line the values up after the longest sub-expression
  for (const node &n : t.first(t.size() - 1))
    width = std::max(width, 2 * n.depth + int(std::string_view(n.text).size()));
  report(t, root, tr, true, width);
  return false;
}

// --- the function forms: my_assert<^^fn>(args...), my_assert2<fn>(args...) --

consteval std::vector<node> function_tree(info fn) {
  builder b;
  b.flatten(returned_expression(fn), fn, 1);
  return b.nodes;
}

consteval std::vector<const char *> parameter_names_of(info fn) {
  std::vector<const char *> names;
  for (info p : m::parameters_of(fn))
    names.push_back(std::define_static_string(m::identifier_of(p)));
  return names;
}

// Shared by both function forms; `form` is how the heading names the call,
// e.g. "my_assert<^^" for my_assert<^^in_range>(5, 10, 1).
template <info Fn, class... Args>
bool check_call(const char *form, std::source_location where,
                const Args &...args) {
  static_assert(sizeof...(Args) == m::parameters_of(Fn).size());
  if ([:Fn:](args...))
    return true;
  constexpr std::span<const node> t =
      std::define_static_array(function_tree(Fn));
  constexpr std::span<const char *const> names =
      std::define_static_array(parameter_names_of(Fn));
  constexpr const char *name = std::define_static_string(m::identifier_of(Fn));
  static_assert(t.size() <= 64);
  const double params[] = {double(args)..., 0};
  trace tr;
  const int root = int(t.size()) - 1;
  eval(t, root, nullptr, params, tr);

  std::printf("%s%s>(", form, name);
  for (std::size_t i = 0; i < sizeof...(Args); ++i)
    std::printf("%s%g", i ? ", " : "", params[i]);
  std::printf(") failed at line %u\n    inside %s(", unsigned(where.line()),
              name);
  for (std::size_t i = 0; i < sizeof...(Args); ++i)
    std::printf("%s%s = %g", i ? ", " : "", names[i], params[i]);
  std::printf("):\n");
  int width = 0;
  for (const node &n : t)
    width = std::max(width, 2 * n.depth + int(std::string_view(n.text).size()));
  report(t, root, tr, false, width);
  return false;
}

// The call object behind both function forms. Its parameters are the
// function's own parameter types (P...), not a deduced pack, so a defaulted
// std::source_location can follow them and records the caller's line.
template <info Fn, class Sig> struct asserter;
template <info Fn, class R, class... P> struct asserter<Fn, R(P...)> {
  const char *form;
  bool operator()(P... args, std::source_location where =
                                 std::source_location::current()) const {
    return check_call<Fn>(form, where, args...);
  }
};

template <info Fn>
constexpr asserter<Fn, typename[:m::type_of(Fn):]> my_assert{"my_assert<^^"};

// my_assert2<in_range>(args...): the function itself as the template
// argument; reflect_function turns it back into the reflection.
template <auto &F>
constexpr asserter<m::reflect_function(F),
                   typename[:m::type_of(m::reflect_function(F)):]>
    my_assert2{"my_assert2<"};

} // namespace pa

#define MY_ASSERT(...) ::pa::check([&] { return __VA_ARGS__; })

using pa::my_assert;
using pa::my_assert2;

// --- the example from the paper ---------------------------------------------

bool in_range(double x, double lo, double hi) {
  return lo <= hi && x >= lo && x <= hi;
}

int main() {
  double x = 5, lo = 10, hi = 1;
  MY_ASSERT(in_range(x, lo, hi));
  std::printf("\n");
  my_assert<^^in_range>(x, lo, hi);
  std::printf("\n");
  my_assert2<in_range>(x, lo, hi);
}
