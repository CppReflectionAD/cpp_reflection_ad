//
// A pointer from define_static_string is non-null in a constant expression,
// also with -fsanitize=nonnull-attribute or
// -fsanitize=returns-nonnull-attribute (both part of -fsanitize=undefined),
// which tests/CMakeLists.txt adds to this file. Either check turns off
// -fdelete-null-pointer-checks, and without it GCC didn't fold `&<template
// parameter object> != 0` to true, so a static_assert on the comparison, or on
// `.find()`, which checks char_traits::find's result against null, failed
// (#262). CppReflectionAD/gcc#36 fixes that.
#include <meta>

constexpr const char *p = std::define_static_string("abc");

// Compared through a parameter: GCC's -Waddress flags `p != nullptr` itself
// once it folds.
constexpr bool non_null(const char *q) { return q != nullptr; }
static_assert(non_null(p), "a static string is never null");

int main() {}
// Also .find(), which checks char_traits::find's result against null.
#include <string_view>
static_assert(std::string_view(std::define_static_string("hello world"))
                  .find("world") == 6);
