// Probes use the paper's spelling, std::meta.
#include <meta>
namespace m = std::meta;
#include <cstddef>
#include <vector>

template <std::size_t I> consteval m::info stmt(m::info fn) {
  return m::statements_of(m::body_of(fn))[I];
}
consteval std::size_t nstmts(m::info fn) {
  return m::statements_of(m::body_of(fn)).size();
}
