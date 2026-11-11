// An operand that names a reference is an id-expression on both compilers, as
// one that names a variable of another type is: the reference already
// denotes the object it refers to.
//
// GCC used to wrap it in a conversion, the dereference it adds to every use
// of a reference, so it failed is_id_expression (#249). Fixed in
// CppReflectionAD/gcc#31, which has these checks, and more, in
// g++.dg/reflect/reference_operand1.C.
//
// Only operands used as lvalues are checked: where a reference is read, as in
// `return r;` or `r + x`, clang shows the lvalue-to-rvalue conversion, as it
// does for any variable, and GCC shows none.
#include <ad/meta.h>

#include <cstddef>
#include <string_view>

namespace m = ad::meta;
using std::meta::info;

inline void inc_value([[maybe_unused]] int i) { ++i; }
inline void inc_ref(int &i) { ++i; }
inline void inc_local(int a) {
  int &r = a;
  ++r;
}
inline void assign_ref(int &r, int x) { r = x; }
inline void add_assign_ref(int &r, int x) { r += x; }
inline int &return_ref(int &r) { return r; }

// Operand K of the expression in statement I of FN's body.
consteval info operand(info fn, std::size_t i, std::size_t k) {
  return m::operands_of(
      m::expression_of(m::statements_of(m::body_of(fn))[i]))[k];
}

consteval bool names(info e, std::string_view name) {
  return m::is_id_expression(e) && !m::is_conversion(e) &&
         m::operands_of(e).empty() &&
         m::identifier_of(m::declaration_of(e)) == name;
}

static_assert(names(operand(^^inc_value, 0, 0), "i"), "++ on an int parameter");
static_assert(names(operand(^^inc_ref, 0, 0), "i"), "++ on an int & parameter");
static_assert(names(operand(^^inc_local, 1, 0), "r"), "++ on a local int &");
static_assert(names(operand(^^assign_ref, 0, 0), "r"),
              "assignment to an int & parameter");
static_assert(names(operand(^^add_assign_ref, 0, 0), "r"),
              "compound assignment to an int & parameter");
static_assert(
    names(m::return_operand_of(m::statements_of(m::body_of(^^return_ref))[0]),
          "r"),
    "an int & returned as an int &");

int main() {}
