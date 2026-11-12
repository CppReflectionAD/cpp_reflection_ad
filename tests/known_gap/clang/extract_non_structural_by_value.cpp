// Known Clang gap: extract<T> by value accepts a non-structural T.
//
// For a non-reference T, extract<T>(r) of a variable is
// extract-value<T>(constant_of(r)), and constant_of is reflect_constant, which
// mandates a structural type ([meta.reflection.extract],
// [meta.reflection.result]). std::span has private members, so it isn't
// structural and the extract below is ill-formed. GCC rejects it; Clang
// accepts it and returns the span.
//
// The gap is accepting invalid code, so this file can't stop compiling when
// it's fixed. Instead the static_assert fails only because Clang accepted the
// extract. When Clang rejects the extract, the error changes, the .fail.txt
// comparison fails, and this test can be deleted.
#include <meta>
#include <span>
#include <vector>

consteval std::span<const int> make() {
  return std::define_static_array(std::vector<int>{1, 2});
}
constexpr std::span<const int> s = make();

// Valid on both compilers: extract by reference needs no structural type.
static_assert(std::meta::extract<const std::span<const int> &>(^^s).size() ==
              2);

constexpr std::size_t n = std::meta::extract<std::span<const int>>(^^s).size();
static_assert(n != 2, "Clang accepted extract<std::span<const int>> by value");

int main() {}
