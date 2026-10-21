#!/usr/bin/env bash
# Fills the CI GCC toolchain cache. It installs the build dependencies, then
# runs the recipe that `run_tests.py --build-compilers` uses
# (tools/toolchain/gcc.py) to build and install the fork into a clean prefix,
# stripped on Linux. On macOS, Xcode already has the build dependencies.
#
# The toolchain cache key hashes this file and the recipe, so a change to
# either rebuilds the compiler. So does a change to the key's other parts (see
# toolchain-cache-gcc.yml): the submodule pin and, on macOS, the runner's SDK
# and linker.
#
# Usage, from the repo root: build-gcc-toolchain.sh [source-dir] [prefix]
#   source-dir  GCC checkout (default: the one tools/toolchain/gcc.py picks for
#               this OS, gcc-darwin-reflect on macOS and gcc-mirror elsewhere)
#   prefix      install prefix (default: /opt/toolchain-gcc)
# The build directory is build/<basename of source-dir>.
set -euo pipefail

SRC="${1:-$(python3 -m tools.toolchain.gcc --print-source-submodule)}"
PREFIX="${2:-/opt/toolchain-gcc}"

if [[ "$(uname -s)" == Linux ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential flex bison libgmp-dev libmpfr-dev \
    libmpc-dev texinfo zlib1g-dev dejagnu python3 git
fi

sudo mkdir -p "$PREFIX"
sudo chown "$(id -u):$(id -g)" "$PREFIX"

python3 -m tools.toolchain.gcc --source-dir "$SRC" \
  --build-dir "build/$(basename "$SRC")" --prefix "$PREFIX" --strip

echo "Installed size: $(du -sh "$PREFIX" | cut -f1)"
