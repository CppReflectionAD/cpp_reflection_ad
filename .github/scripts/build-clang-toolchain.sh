#!/usr/bin/env bash
# Fills the CI clang toolchain cache. It installs the build dependencies, then
# runs the recipe that `run_tests.py --build-compilers` uses
# (tools/toolchain/clang.py), and installs a trimmed, stripped copy of the
# result into a clean prefix.
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

# Build dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build clang lld \
  python3 zlib1g-dev git

sudo mkdir -p "$PREFIX"
sudo chown "$(id -u):$(id -g)" "$PREFIX"

# clang itself is Release here, where a local build is RelWithDebInfo: the
# cache is stripped, so its debug info would only cost build time and disk.
# The runtimes are RelWithDebInfo in both.
python3 -m tools.toolchain.clang \
  --source-dir "$SRC" \
  --root build/clang-p2996 \
  --runtimes-dir build/libcxx \
  --build-type Release \
  --prefix "$PREFIX" \
  --strip

echo "Installed size: $(du -sh "$PREFIX" | cut -f1)"
