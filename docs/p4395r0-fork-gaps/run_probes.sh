#!/usr/bin/env bash
# Compile each p*.cpp probe (or those starting with $1) on clang and GCC and
# print PASS, or the first errors.
#
# Compilers default to the fetched toolchains in the checkout's build/ (see
# CONTRIBUTING.md); override with CLANGXX and GXX.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
CLANGXX=${CLANGXX:-$root/build/clang-p2996/bin/clang++}
GXX=${GXX:-$root/build/gcc-p2996/artifacts/bin/g++}
inc="-I$here"
clang_flags="-std=c++2c -freflection -fparameter-reflection -fexpansion-statements -stdlib=libc++ -isystem $(dirname "$CLANGXX")/../include/c++/v1 -fsyntax-only"
gcc_flags="-std=c++2c -freflection -fsyntax-only"
# A runaway constant evaluation can exhaust memory; cap it.
ulimit -v 6000000

for f in "$here"/p*.cpp; do
  b=$(basename "$f" .cpp)
  [ -n "${1:-}" ] && [[ $b != "$1"* ]] && continue
  echo "## $b"
  for cfg in clang gcc; do
    case $cfg in
    clang) cmd="$CLANGXX $clang_flags $inc $f" ;;
    gcc) cmd="$GXX $gcc_flags $inc $f" ;;
    esac
    if out=$(timeout 120 $cmd 2>&1); then
      echo "  $cfg: PASS"
    else
      echo "  $cfg: FAIL"
      grep -E "error|static assertion" <<<"$out" | head -4 | sed "s|$here/||g; s|^|      |" | cut -c1-200
    fi
  done
done
