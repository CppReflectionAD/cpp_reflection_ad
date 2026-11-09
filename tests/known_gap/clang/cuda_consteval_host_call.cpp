// Known clang gap, inherited from upstream clang's CUDA support: host/device
// checks apply to code that only runs during constant evaluation.
//
// runtime_only_host() is constexpr, so clang treats it as __host__ __device__,
// and its runtime path calls a host-only function. A kernel reaches it only
// through the consteval three(), which runs at compile time, so the host call
// can never happen on the device. Clang rejects the kernel anyway.
//
// Reflection hits this in its most basic use. Queries such as parameters_of
// return a std::vector, whose constexpr members reach host-only libc++
// functions in the same way, so `reflected` below is rejected too, with an
// error inside libc++ for each one. Queries that return no vector, such as
// identifier_of, are accepted. The same stops a kernel calling
// ad::gradient_reverse, whose static constexpr node lists are built from a
// std::vector.
//
// tests/CMakeLists.txt compiles this file as CUDA device code to LLVM bitcode,
// which needs neither a CUDA SDK nor clang's NVPTX backend, and expects the
// errors in cuda_consteval_host_call.fail.txt: the first is from `kernel`, the
// rest from `reflected`. When clang fixes the gap, the file compiles and the
// test fails at the link step instead (the object is device bitcode): then
// delete the test and its .fail.txt.
#include <cstddef>
#include <meta>

void host_only();

constexpr void runtime_only_host() {
  if !consteval {
    host_only();
  }
}

consteval int three() {
  runtime_only_host();
  return 3;
}

__attribute__((global)) void kernel(int *out) { out[0] = three(); }

inline float f(float x, float y) { return x * y; }

__attribute__((global)) void reflected(unsigned *out) {
  constexpr std::size_t arity = std::meta::parameters_of(^^f).size();
  out[0] = static_cast<unsigned>(arity);
}
