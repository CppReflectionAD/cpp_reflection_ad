#!/usr/bin/env bash
# Compile each p*.cpp probe (or those starting with $1) and print PASS, or the
# first errors. Configurations: clang; GCC through std::meta (as the paper
# spells it) and through ad::meta; and, if GXX_FIXED names a second GCC, that
# one the same two ways.
#
# Compilers default to the fetched toolchains in the checkout's build/ (see
# CONTRIBUTING.md); override with CLANGXX and GXX.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
CLANGXX=${CLANGXX:-$root/build/clang-p2996/bin/clang++}
GXX=${GXX:-$root/build/gcc-p2996/artifacts/bin/g++}
GXX_FIXED=${GXX_FIXED:-}
inc="-I$here -I$root/include"
clang_flags="-std=c++2c -freflection -fparameter-reflection -fexpansion-statements -stdlib=libc++ -isystem $(dirname "$CLANGXX")/../include/c++/v1 -fsyntax-only"
gcc_flags="-std=c++2c -freflection -fsyntax-only"
# A walk over GCC's self-referencing operands can exhaust memory; cap it.
ulimit -v 6000000

configs="clang gcc:std gcc:ad"
[ -n "$GXX_FIXED" ] && configs="$configs gcc-fixed:std gcc-fixed:ad"

for f in "$here"/p*.cpp; do
  b=$(basename "$f" .cpp)
  [ -n "${1:-}" ] && [[ $b != "$1"* ]] && continue
  echo "## $b"
  for cfg in $configs; do
    case $cfg in
    clang) cmd="$CLANGXX $clang_flags $inc $f" ;;
    gcc:std) cmd="$GXX $gcc_flags $inc $f" ;;
    gcc:ad) cmd="$GXX $gcc_flags $inc -DUSE_AD $f" ;;
    gcc-fixed:std) cmd="$GXX_FIXED $gcc_flags $inc $f" ;;
    gcc-fixed:ad) cmd="$GXX_FIXED $gcc_flags $inc -DUSE_AD $f" ;;
    esac
    if out=$(timeout 120 $cmd 2>&1); then
      echo "  $cfg: PASS"
    else
      echo "  $cfg: FAIL"
      grep -E "error|static assertion" <<<"$out" | head -4 | sed "s|$here/||g; s|^|      |" | cut -c1-200
    fi
  done
done
