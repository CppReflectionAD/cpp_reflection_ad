// §3.1: has_body(fn) is true for a defined function, false for a declared-only
// one.
#include "probe.h"
double defined(double x) { return x; }
double declared_only(double x);
static_assert(m::has_body(^^defined), "P01 has_body(defined)");
static_assert(!m::has_body(^^declared_only), "P01 !has_body(declared_only)");
