// Known clang gap, inherited from upstream clang's CUDA support: a constexpr
// variable is kept off the device when its initializer creates a temporary
// with a non-trivial destructor, even one destroyed during constant
// evaluation.
//
// Device code can use a namespace-scope constexpr variable only if clang makes
// it __constant__, which it does only if the initializer passes a syntactic
// check. `rejected` fails it because of the temporary Dtor{}. `accepted` holds
// the same value, with the temporary moved inside a consteval function, and
// passes. Reading a member directly needs no device copy; calling a member
// function does.
//
// Reflection hits this whenever it keeps a query's result in a variable.
// std::define_static_array is how reflection code turns a query's
// std::vector into a constant, and the vector is the temporary, so
// `params_rejected` below is rejected too. `params_accepted` makes the same
// call inside a consteval function. The same stops a kernel using
// ad::nodes_of, which is std::define_static_array(build_nodes<Fn>()).
//
// tests/CMakeLists.txt compiles this file as CUDA device code to LLVM bitcode,
// which needs neither a CUDA SDK nor clang's NVPTX backend, and expects the
// errors in cuda_constexpr_var_on_device.fail.txt, one for each rejected
// variable. When clang fixes the gap, the file compiles and the test fails at
// the link step instead (the object is device bitcode): then delete the test
// and its .fail.txt.
#include <cstddef>
#include <meta>
#include <span>

struct Dtor {
  constexpr ~Dtor() {}
};

struct Box {
  int v;
  constexpr int get() const { return v; }
};

consteval Box box(const Dtor &) { return {3}; }
consteval Box boxed() { return box(Dtor{}); }

inline constexpr Box rejected = box(Dtor{});
inline constexpr Box accepted = boxed();

__attribute__((global)) void kernel(int *out) {
  constexpr int a = rejected.get();
  constexpr int b = accepted.get();
  out[0] = a + b;
}

inline float f(float x, float y) { return x * y; }

consteval std::span<const std::meta::info> params_of_f() {
  return std::define_static_array(std::meta::parameters_of(^^f));
}

inline constexpr auto params_rejected =
    std::define_static_array(std::meta::parameters_of(^^f));
inline constexpr auto params_accepted = params_of_f();

__attribute__((global)) void reflected(unsigned *out) {
  constexpr std::size_t a = params_rejected.size();
  constexpr std::size_t b = params_accepted.size();
  out[0] = static_cast<unsigned>(a + b);
}
