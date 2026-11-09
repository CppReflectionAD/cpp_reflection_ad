// TEST-PASSES-ON: clang
//
// Known GCC gap (#262): with -fsanitize=nonnull-attribute or
// -fsanitize=returns-nonnull-attribute, both part of -fsanitize=undefined, a
// pointer from define_static_string can't be compared against null in a
// constant expression. So `.find()` on its result fails in a static_assert too,
// because char_traits::find checks the pointer it finds against null; this file
// checks only the comparison, so its snapshot doesn't depend on libstdc++.
//
// Either check turns off -fdelete-null-pointer-checks, and without it GCC no
// longer folds `&<template parameter object> != 0` to true. Reflection isn't
// involved: system GCC 13.3 rejects the same comparison on a pointer into any
// template parameter object.
//
// tests/CMakeLists.txt builds this file with the sanitizer flag and expects the
// errors in sanitize_nonnull_static_string.fail.txt. When GCC fixes the gap,
// the test fails because the file compiles: move it to tests/common/ (without
// the flag or the TEST-PASSES-ON line) and delete the .fail.txt. Then drop the
// nonnull-attribute,returns-nonnull-attribute entry that REFLECTION_AD_SANITIZE
// adds for GCC in cmake/reflect_flags.cmake.
#include <meta>

constexpr const char *p = std::define_static_string("abc");

static_assert(p != nullptr, "a static string is never null");

int main() {}
