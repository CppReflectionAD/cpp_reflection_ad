// A copy of a class value is a construct, on both compilers (#248).
//
// ad_tensor_operator_by_value.cpp covers a class that isn't trivially
// copyable, which GCC passes by invisible reference. This test covers the
// other two forms: a trivially copyable class passed by value (`pass`) and a
// returned copy of a by-value parameter (`copy`). GCC used to give both no
// operands. Fixed in CppReflectionAD/gcc#25.
#include <ad/meta.h>

namespace m = ad::meta;

struct S {
  double v;
};
inline S copy(S a) { return a; }
inline S pass(S a) { return copy(a); }

consteval m::info returned(m::info fn) {
  return m::return_operand_of(m::statements_of(m::body_of(fn))[0]);
}

static_assert(m::is_construct(returned(^^copy)),
              "a returned copy of a class value is a construct");
static_assert(m::is_construct(m::operands_of(returned(^^pass))[0]),
              "a class value passed by value is a construct");

int main() {}
