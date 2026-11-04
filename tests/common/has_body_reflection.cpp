// has_body (P4395R0 §3.1) says whether body_of applies to a reflection: true
// for a function defined with a body, and false for anything else. A library
// that reflects arbitrary callees asks it before calling body_of, which is an
// error on clang (no std::meta::exception) and throws on GCC.
//
// clang gained has_body in CppReflectionAD/clang-p4395#8 (#204); GCC has it
// since CppReflectionAD/gcc#11.
#include <ad/meta.h>

namespace m = ad::meta;

double defined(double x) { return x; }
double declared_only(double x);
double deleted(double) = delete;
struct S {
  double v;
  bool operator==(const S &) const = default;
};
template <typename T> T tmpl(T x) { return x; }
template <typename T> T declared_tmpl(T x);
int var;

static_assert(m::has_body(^^defined));
static_assert(m::is_compound_statement(m::body_of(^^defined)));
static_assert(!m::has_body(^^declared_only));
static_assert(!m::has_body(^^deleted));
static_assert(!m::has_body(^^S::operator==));

// A specialization is instantiated first; one of a template that isn't
// defined has no body.
static_assert(m::has_body(^^tmpl<double>));
static_assert(m::is_compound_statement(m::body_of(^^tmpl<float>)));
static_assert(!m::has_body(^^declared_tmpl<double>));

// Not a function: a template, a variable, a type, a statement, null.
static_assert(!m::has_body(^^tmpl));
static_assert(!m::has_body(^^var));
static_assert(!m::has_body(^^S));
static_assert(!m::has_body(m::body_of(^^defined)));
static_assert(!m::has_body(m::info{}));

int main() {}
