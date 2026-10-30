// §3.1/§4.1: without has_body, what does body_of do on a function with no
// definition? (Informational: the paper makes has_body the precondition.)
#include "probe.h"
double declared_only(double x);
constexpr m::info b = m::body_of(^^declared_only);
static_assert(b == m::info{},
              "P02 body_of(declared_only) is a null reflection");
