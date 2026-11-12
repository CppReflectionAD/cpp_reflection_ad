// Two reflections of a parameter that compare equal are the same template
// argument, on both compilers, though the definition adds a top-level const
// that the declaration doesn't have.
//
// GCC compared them as the same parameter but hashed them with its type, so
// they named two specializations of X and comparing the two types crashed GCC
// ("canonical types differ for identical types", #271). Fixed in
// CppReflectionAD/gcc#34, which has these checks, and more, in
// g++.dg/reflect/template_arg_hash1.C.
#include <meta>
#include <type_traits>

using std::meta::info;

template <info I> struct X {};

void f(int, long);
constexpr info early0 = std::meta::parameters_of(^^f)[0];
constexpr info early1 = std::meta::parameters_of(^^f)[1];
void f(const int, const long) {}
constexpr info late0 = std::meta::parameters_of(^^f)[0];

static_assert(early0 == late0);
static_assert(std::is_same_v<X<early0>, X<late0>>,
              "one parameter before and after the definition");
static_assert(!std::is_same_v<X<early0>, X<early1>>, "two parameters");

int main() {
  f(0, 0);
  return 0;
}
