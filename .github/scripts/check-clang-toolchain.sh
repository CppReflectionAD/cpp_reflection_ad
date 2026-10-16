#!/usr/bin/env bash
# Checks that a clang toolchain prefix has what the tests and the tooling items
# need, using nothing outside the prefix: reflection with the fork's libc++,
# ASan and UBSan with symbolized reports (T1), coverage (T3) and clang-tidy
# (L4). CI runs it before saving the toolchain cache.
#
# Usage: check-clang-toolchain.sh <prefix>
set -euo pipefail

P="$(realpath "$1")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

flags=(-std=c++2c -freflection -fparameter-reflection -fexpansion-statements
  -stdlib=libc++ -isystem "$P/include/c++/v1")
link=(-L"$P/lib" -Wl,-rpath,"$P/lib")
fail() { echo "check-clang-toolchain: $*" >&2; exit 1; }

# No binary or library may still point into the build tree. readelf fails on
# the non-ELF files (run-clang-tidy is Python), which pipefail mustn't see.
if { readelf -d "$P"/bin/* "$P"/lib/*.so* 2>/dev/null || true; } |
  grep -E 'R(UN)?PATH' | grep -v -F '[$ORIGIN'; then
  fail "a RUNPATH points outside the prefix (above)"
fi

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
"$P/bin/clang++" "${flags[@]}" reflect.cpp -o reflect "${link[@]}"
./reflect || fail "reflect.cpp built against the prefix doesn't run"
echo "ok: reflection and libc++"

cat > asan.cpp <<'CPP'
#include <vector>
int main(int argc, char**) {
  std::vector<int> v(2);
  return v.data()[argc + 1];
}
CPP
"$P/bin/clang++" "${flags[@]}" -g -fsanitize=address,undefined asan.cpp -o asan "${link[@]}"
if ASAN_SYMBOLIZER_PATH="$P/bin/llvm-symbolizer" ./asan 2> asan.log; then
  fail "ASan missed a heap-buffer-overflow"
fi
grep -q 'AddressSanitizer: heap-buffer-overflow' asan.log || { cat asan.log; fail "no ASan report"; }
grep -q 'asan.cpp:4' asan.log || { cat asan.log; fail "the ASan report isn't symbolized"; }
echo "ok: ASan, symbolized"

cat > ubsan.cpp <<'CPP'
#include <climits>
int main(int argc, char**) {
  int x = INT_MAX;
  x += argc;
  return x > 0;
}
CPP
"$P/bin/clang++" "${flags[@]}" -fsanitize=undefined ubsan.cpp -o ubsan "${link[@]}"
./ubsan 2> ubsan.log || true
grep -q 'runtime error: signed integer overflow' ubsan.log || { cat ubsan.log; fail "no UBSan report"; }
echo "ok: UBSan"

"$P/bin/clang++" "${flags[@]}" -fprofile-instr-generate -fcoverage-mapping \
  reflect.cpp -o cov "${link[@]}"
LLVM_PROFILE_FILE=cov.profraw ./cov
"$P/bin/llvm-profdata" merge -o cov.profdata cov.profraw
"$P/bin/llvm-cov" report ./cov -instr-profile=cov.profdata | grep -q '^TOTAL' ||
  fail "llvm-cov has no report"
echo "ok: coverage"

# A parse error (say, of ^^) is a clang-diagnostic-error, which fails the run.
sed 's/^int main.*/int main() { return f() - 1; }/' reflect.cpp > tidy.cpp
"$P/bin/clang-tidy" --quiet -checks='-*,bugprone-*' --warnings-as-errors='*' \
  tidy.cpp -- "${flags[@]}"
echo "ok: clang-tidy"
