// §3.1: without has_body, can a failing body_of be caught and used as a test?
#include "probe.h"
double defined(double x) { return x; }
double declared_only(double x);
consteval bool has_body_via_catch(m::info fn) {
  try {
    (void)m::body_of(fn);
    return true;
  } catch (const std::meta::exception &) {
    return false;
  }
}
static_assert(has_body_via_catch(^^defined), "P20 defined");
static_assert(!has_body_via_catch(^^declared_only), "P20 declared_only caught");
