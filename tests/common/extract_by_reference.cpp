// extract<T> by reference needs no structural type: extract<const T &>(r) of
// a variable of a non-structural type such as std::span gives a reference to
// the variable ([meta.reflection.extract]). By value, the same extract is
// ill-formed; static_fail/extract_non_structural_by_value checks that.
#include <meta>
#include <span>
#include <vector>

consteval std::span<const int> make() {
  return std::define_static_array(std::vector<int>{1, 2});
}
constexpr std::span<const int> s = make();

static_assert(std::meta::extract<const std::span<const int> &>(^^s).size() ==
              2);
static_assert(&std::meta::extract<const std::span<const int> &>(^^s) == &s);

int main() {}
