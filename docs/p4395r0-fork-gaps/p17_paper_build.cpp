// §5.1: the paper's build() extract, verbatim apart from minimal Node/Op/
// Builder stubs, with `namespace m = std::meta` as in the paper.
#include <cstddef>
#include <meta>
#include <vector>

enum class Op { Input, Other };
struct Node {
  Op op;
};
struct Builder {
  std::vector<std::meta::info> names;
  std::vector<std::size_t> slots;
  std::vector<Node> nodes;
  consteval std::size_t emit(Node n) {
    nodes.push_back(n);
    return nodes.size() - 1;
  }
  consteval std::size_t lower(std::meta::info e) {
    for (std::meta::info o : std::meta::operands_of(e))
      if (std::meta::is_expression(o))
        lower(o);
    return emit({Op::Other});
  }
};

namespace m = std::meta;
consteval std::vector<Node> build(m::info fn) {
  Builder b;
  for (m::info p : m::parameters_of(fn)) {
    b.names.push_back(p);
    b.slots.push_back(b.emit({Op::Input}));
  }
  for (m::info s : m::statements_of(m::body_of(fn))) {
    if (m::is_declaration_statement(s)) { // double v = expr, w = expr2;
      auto decls = m::declarators_of(s);
      auto inits = m::initializers_of(s);
      for (std::size_t i = 0; i < decls.size(); ++i) {
        b.names.push_back(decls[i]);
        b.slots.push_back(b.lower(inits[i]));
      }
    } else // return expr;
      b.lower(m::return_operand_of(s));
  }
  return b.nodes;
}

double f(double x, double y) {
  double v = x * y, w = v + x;
  return v * w;
}
static_assert(build(^^f).size() > 2,
              "P17 paper build() runs on a multi-declarator body");
