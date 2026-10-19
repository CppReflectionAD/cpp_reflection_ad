#!/usr/bin/env bash
# Fills the CI clang toolchain cache. It installs the build dependencies, then
# runs the recipe that `run_tests.py --build-compilers` uses
# (tools/toolchain/clang.py), and installs a trimmed copy of the result into a
# clean prefix, stripped on Linux. On macOS, Xcode and the runner image already
# have the build dependencies.
#
# The toolchain cache key hashes this file and the recipe (see
# toolchain-cache-clang.yml), so a change to either rebuilds the compiler, and
# nothing else does.
#
# Usage, from the repo root: build-clang-toolchain.sh [source-dir] [prefix]
#   source-dir  clang-p2996 checkout (default: clang-p2996)
#   prefix      install prefix (default: /opt/toolchain-clang)
set -euo pipefail

SRC="${1:-clang-p2996}"
PREFIX="${2:-/opt/toolchain-clang}"

# clang itself is Release here, where a local build is RelWithDebInfo: its
# debug info would only cost build time and disk. The runtimes are
# RelWithDebInfo in both.
recipe_args=(--source-dir "$SRC" --root build/clang-p2996 \
  --runtimes-dir build/libcxx --build-type Release --prefix "$PREFIX")

if [[ "$(uname -s)" == Linux ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake ninja-build clang lld \
    python3 zlib1g-dev git
  # The recipe's strip step uses GNU strip options, which Apple's strip
  # rejects.
  recipe_args+=(--strip)
fi

sudo mkdir -p "$PREFIX"
sudo chown "$(id -u):$(id -g)" "$PREFIX"

python3 -m tools.toolchain.clang "${recipe_args[@]}"

echo "Installed size: $(du -sh "$PREFIX" | cut -f1)"
