// TEST-PASSES-ON: clang
//
// Known GCC gap (P2996 implementation): a range-for written directly over
// members_of(namespace) in a consteval function sees the namespace as it was
// where the function was defined. members_of should return every member
// declared before some point in the evaluation context
// ([meta.reflection.member.queries]), and the static_assert below follows
// lib::reg::A. Binding the result to a local first gives the right
// answer. Clang gets both right.
//
// Found in the nn engine (#121), whose rule registry is a namespace that rules
// are added to after the engine is defined.
#include <meta>

namespace lib {
namespace reg {}
namespace detail {
consteval std::size_t count_range_for() {
  std::size_t n = 0;
  for (std::meta::info r :
       std::meta::members_of(^^reg, std::meta::access_context::unchecked()))
    n += std::meta::is_type(r);
  return n;
}

consteval std::size_t count_via_local() {
  const auto members =
      std::meta::members_of(^^reg, std::meta::access_context::unchecked());
  std::size_t n = 0;
  for (std::meta::info r : members)
    n += std::meta::is_type(r);
  return n;
}
} // namespace detail
} // namespace lib

namespace lib::reg {
struct A {};
} // namespace lib::reg

static_assert(lib::detail::count_via_local() == 1);
static_assert(lib::detail::count_range_for() == 1,
              "range-for over members_of misses lib::reg::A");

int main() {}
