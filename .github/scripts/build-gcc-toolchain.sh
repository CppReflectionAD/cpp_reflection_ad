#!/usr/bin/env bash
# Fills the CI GCC toolchain cache. It installs the build dependencies, then
# runs the recipe that `run_tests.py --build-compilers` uses
# (tools/toolchain/gcc.py) to build and install the fork into a clean prefix,
# stripped on Linux. On macOS, Xcode already has the build dependencies.
#
# The toolchain cache key hashes this file and the recipe (see
# toolchain-cache-gcc.yml), so a change to either rebuilds the compiler, and
# nothing else does.
#
# Usage, from the repo root: build-gcc-toolchain.sh [source-dir] [prefix]
#   source-dir  GCC checkout (default: the one tools/toolchain/gcc.py picks for
#               this OS, gcc-darwin-reflect on macOS and gcc-mirror elsewhere)
#   prefix      install prefix (default: /opt/toolchain-gcc)
# The build directory is build/<basename of source-dir>.
set -euo pipefail

SRC="${1:-$(python3 -c 'from tools.toolchain.gcc import SOURCE_SUBMODULE; print(SOURCE_SUBMODULE)')}"
PREFIX="${2:-/opt/toolchain-gcc}"

recipe_args=(--source-dir "$SRC" --build-dir "build/$(basename "$SRC")" \
  --prefix "$PREFIX" --strip)

if [[ "$(uname -s)" == Linux ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential flex bison libgmp-dev libmpfr-dev \
    libmpc-dev texinfo zlib1g-dev dejagnu python3 git
fi

sudo mkdir -p "$PREFIX"
sudo chown "$(id -u):$(id -g)" "$PREFIX"

python3 -m tools.toolchain.gcc "${recipe_args[@]}"

echo "Installed size: $(du -sh "$PREFIX" | cut -f1)"
