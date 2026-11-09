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
// This is what stops a kernel using ad::nodes_of, which is
// std::define_static_array(build_nodes<Fn>()): the temporary is the
// std::vector that build_nodes returns.
//
// tests/CMakeLists.txt compiles this file as CUDA device code to LLVM bitcode,
// which needs neither a CUDA SDK nor clang's NVPTX backend, and expects the
// error in cuda_constexpr_var_on_device.fail.txt (none for `accepted`). When
// clang fixes the gap, the file compiles and the test fails at the link step
// instead (the object is device bitcode): then delete the test and its
// .fail.txt.

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
