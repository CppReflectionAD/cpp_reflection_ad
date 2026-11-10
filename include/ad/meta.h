// meta.h — the namespace that code reflecting function bodies uses.
//
// ad::meta is std::meta. Both forks now have the same statement/expression
// API (siqiyyyy/clang-p2996#11, CppReflectionAD/gcc#11), so nothing here
// differs by compiler; ad::meta is where a workaround for one fork goes if it
// needs one again.

#ifndef REFLECT_DEMO_META_H
#define REFLECT_DEMO_META_H

#include <meta>

namespace ad::meta {
using namespace std::meta;
} // namespace ad::meta

#endif
