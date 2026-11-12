// A block is a compound statement ([stmt.block]) whether or not it declares
// anything, so statements_of gives it in the block it is in, and its own
// statements only inside it, on both compilers.
//
// GCC used to leave out a block that declares nothing, or give its statements
// in the enclosing block (#245). Fixed in CppReflectionAD/gcc#28, which has the
// same checks in g++.dg/reflect/nested_block1.C.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

consteval info stmt(info s, std::size_t i) { return m::statements_of(s)[i]; }

consteval std::size_t count(info s) { return m::statements_of(s).size(); }

// Statement I of the first statement of FN's body.
consteval info inner(info fn, std::size_t i) {
  return stmt(stmt(m::body_of(fn), 0), i);
}

consteval bool block_p(info s, std::size_t n) {
  return m::is_compound_statement(s) && !m::is_expression_statement(s) &&
         !m::is_declaration_statement(s) && !m::is_return_statement(s) &&
         count(s) == n;
}

// A block with N statements, then a return.
consteval bool block_then_return(info fn, std::size_t n) {
  info b = m::body_of(fn);
  return count(b) == 2 && block_p(stmt(b, 0), n) &&
         m::is_return_statement(stmt(b, 1));
}

inline double f_decl(double x) {
  {
    [[maybe_unused]] int a = 0;
  }
  return x;
}
inline double f_empty(double x) {
  {
  }
  return x;
}
inline double f_expr(double x) {
  {
    (void)(x + 1);
  }
  return x;
}
inline double f_null(double x) {
  {
    ;
  }
  return x;
}
inline double f_nested(double x) {
  {
    {
      (void)(x + 1);
    }
  }
  return x;
}
template <typename T> T t_empty(T x) {
  {
  }
  return x;
}

static_assert(block_then_return(^^f_decl, 1));
static_assert(m::is_declaration_statement(inner(^^f_decl, 0)));

static_assert(block_then_return(^^f_empty, 0));

static_assert(block_then_return(^^f_expr, 1));
static_assert(m::is_expression_statement(inner(^^f_expr, 0)));

// null_statement_reflection checks what kind of statement the `;` is.
static_assert(block_then_return(^^f_null, 1));

static_assert(block_then_return(^^f_nested, 1));
static_assert(block_p(inner(^^f_nested, 0), 1));

static_assert(block_then_return(^^t_empty<double>, 0));

// Two blocks that are alike are two statements.
static_assert(stmt(m::body_of(^^f_empty), 0) !=
              stmt(m::body_of(^^t_empty<double>), 0));

int main() { return 0; }
