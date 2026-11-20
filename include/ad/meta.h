// meta.h — the namespace that code reflecting function bodies uses.
//
// ad::meta is std::meta. Both forks now have the same statement/expression
// API, including Option 2 loop reflection and if-statement branch accessors;
// ad::meta is where any compiler-specific workaround goes.

#ifndef REFLECT_DEMO_META_H
#define REFLECT_DEMO_META_H

#include <meta>

namespace ad::meta {
using namespace std::meta;
} // namespace ad::meta

#endif
