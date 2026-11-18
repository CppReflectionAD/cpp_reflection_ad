// TEST-PASSES-ON: clang
// Known GCC gap: template_of(^^f<3>) == ^^f holds, yet as a template argument
// it doesn't name the same specialization: substitute() doesn't find reg<^^f>.
// (Spelling reg<template_of(...)> and comparing it with reg<^^f> makes GCC
// ICE: "canonical types differ for identical types".) So a registry keyed on a
// function template can't be looked up from a call to one of its
// specializations.
//
// When GCC fixes it, the test fails because the file compiles: move it to
// tests/common/ and delete its .fail.txt.
#include <meta>

template <int N> int f() { return N; }

template <std::meta::info K> struct reg;
template <> struct reg<^^f> {};

static_assert(std::meta::template_of(^^f<3>) == ^^f);
static_assert(std::meta::is_complete_type(std::meta::substitute(
    ^^reg, {
               std::meta::reflect_constant(std::meta::template_of(^^f<3>))})));

int main() {}
