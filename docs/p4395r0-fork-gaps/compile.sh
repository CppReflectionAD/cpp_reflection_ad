#!/usr/bin/env bash
# Compile one probe on clang and GCC (and GXX_FIXED, if set) and show the
# diagnostics, e.g.
#   compile.sh d01_dumps.cpp -DWHICH=5          # dump the tree for case 5
#   compile.sh r07_gcc_body_operands_ice.cpp -DWHICH=4
# The dumps print through a failing static_assert, so an error is expected.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
CLANGXX=${CLANGXX:-$root/build/clang-p2996/bin/clang++}
GXX=${GXX:-$root/build/gcc-p2996/artifacts/bin/g++}
GXX_FIXED=${GXX_FIXED:-}
f=$here/$1
shift
inc="-I$here -I$root/include"
ulimit -v 6000000

clean() {
  grep -vE "^\s*[0-9]* \||^\s*\|?\s*\^|In file included|expanded from macro|in expansion of macro|^0x|^\s+(/|\?\?)" |
    sed "s|$here/||g" | head -60
}

echo "### clang"
timeout 120 "$CLANGXX" -std=c++2c -freflection -fparameter-reflection -fexpansion-statements \
  -stdlib=libc++ -isystem "$(dirname "$CLANGXX")/../include/c++/v1" -fsyntax-only $inc "$@" "$f" 2>&1 | clean
echo "### gcc"
timeout 120 "$GXX" -std=c++2c -freflection -fsyntax-only $inc "$@" "$f" 2>&1 | clean
if [ -n "$GXX_FIXED" ]; then
  echo "### gcc-fixed"
  timeout 120 "$GXX_FIXED" -std=c++2c -freflection -fsyntax-only $inc "$@" "$f" 2>&1 | clean
fi
