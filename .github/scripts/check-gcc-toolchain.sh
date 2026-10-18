#!/usr/bin/env bash
# Checks that a GCC toolchain prefix has what the tests and T1 need, using
# nothing outside the prefix: reflection with the fork's libstdc++, and ASan
# and UBSan. CI runs it before saving the toolchain cache.
#
# Usage: check-gcc-toolchain.sh <prefix>
set -euo pipefail

P="$(realpath "$1")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

flags=(-std=c++2c -freflection)
link=(-Wl,-rpath,"$P/lib64")
fail() { echo "check-gcc-toolchain: $*" >&2; exit 1; }

cat > reflect.cpp <<'CPP'
#include <meta>
#include <vector>
int f() { return 1; }
consteval bool has_return(std::meta::info fn) {
  for (std::meta::info s : std::meta::statements_of(std::meta::body_of(fn)))
    if (std::meta::is_return_statement(s))
      return true;
  return false;
}
static_assert(has_return(^^f));
int main() { return std::vector<int>{f()}.at(0) - 1; }
CPP
"$P/bin/g++" "${flags[@]}" reflect.cpp -o reflect "${link[@]}"
./reflect || fail "reflect.cpp built against the prefix doesn't run"
echo "ok: reflection and libstdc++"

# A raw array: the fork's ASan doesn't see this read through std::vector's
# storage (true unstripped too).
cat > asan.cpp <<'CPP'
int main(int argc, char**) {
  int* p = new int[2];
  int r = p[argc + 1];
  delete[] p;
  return r;
}
CPP
"$P/bin/g++" "${flags[@]}" -g -fsanitize=address,undefined asan.cpp -o asan "${link[@]}"
if ./asan 2> asan.log; then
  fail "ASan missed a heap-buffer-overflow"
fi
grep -q 'AddressSanitizer: heap-buffer-overflow' asan.log || { cat asan.log; fail "no ASan report"; }
echo "ok: ASan"

cat > ubsan.cpp <<'CPP'
#include <climits>
int main(int argc, char**) {
  int x = INT_MAX;
  x += argc;
  return x > 0;
}
CPP
"$P/bin/g++" "${flags[@]}" -fsanitize=undefined ubsan.cpp -o ubsan "${link[@]}"
./ubsan 2> ubsan.log || true
grep -q 'runtime error: signed integer overflow' ubsan.log || { cat ubsan.log; fail "no UBSan report"; }
echo "ok: UBSan"
