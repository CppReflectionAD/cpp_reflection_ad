// Probes use the paper's spelling (std::meta) unless USE_AD selects the
// repo's ad::meta compatibility shim.
#ifdef USE_AD
#include <ad/meta.h>
namespace m = ad::meta;
#else
#include <meta>
namespace m = std::meta;
#endif
#include <cstddef>
#include <vector>

template <std::size_t I> consteval m::info stmt(m::info fn) {
  return m::statements_of(m::body_of(fn))[I];
}
consteval std::size_t nstmts(m::info fn) {
  return m::statements_of(m::body_of(fn)).size();
}
