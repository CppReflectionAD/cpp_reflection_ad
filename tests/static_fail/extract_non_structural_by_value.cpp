// extract<T> by value must reject a non-structural T, on every compiler.
//
// For a non-reference T, extract<T>(r) of a variable is
// extract-value<T>(constant_of(r)), and constant_of is reflect_constant, which
// mandates a structural type ([meta.reflection.extract],
// [meta.reflection.result]). std::span has private members, so it isn't
// structural and the extract below is ill-formed.
//
// GCC always rejected it. clang-p2996 accepted it and returned the span until
// clang-p4395#11 (#275). Upstream compilers implementing P2996 may trip over
// the same point, so this file keeps the case here, not only in the forks'
// testsuites.
//
// Extracting the same variable by reference is valid, and is checked here too,
// so that the by-value extract is the only thing in this file that can fail.
// On clang the snapshot checks only that `n` isn't a constant expression: the
// note that names the non-structural type isn't compared.
#include <meta>
#include <span>
#include <vector>

consteval std::span<const int> make() {
  return std::define_static_array(std::vector<int>{1, 2});
}
constexpr std::span<const int> s = make();

static_assert(std::meta::extract<const std::span<const int> &>(^^s).size() ==
              2);

constexpr std::size_t n = std::meta::extract<std::span<const int>>(^^s).size();

int main() {}
