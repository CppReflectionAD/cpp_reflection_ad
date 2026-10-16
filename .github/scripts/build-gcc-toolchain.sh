#!/usr/bin/env bash
# Fills the CI GCC toolchain cache. It installs the build dependencies, then
# runs the recipe that `run_tests.py --build-compilers` uses
# (tools/toolchain/gcc.py) to build and install the fork, stripped, into a
# clean prefix.
#
# The toolchain cache key hashes this file and the recipe (see
# toolchain-cache-gcc.yml), so a change to either rebuilds the compiler, and
# nothing else does.
#
# Usage, from the repo root: build-gcc-toolchain.sh [source-dir] [prefix]
#   source-dir  gcc-mirror checkout (default: gcc-mirror)
#   prefix      install prefix (default: /opt/toolchain-gcc)
# The build directory is build/<basename of source-dir>.
set -euo pipefail

SRC="${1:-gcc-mirror}"
PREFIX="${2:-/opt/toolchain-gcc}"

# Build dependencies
sudo apt-get update
sudo apt-get install -y build-essential flex bison libgmp-dev libmpfr-dev \
  libmpc-dev texinfo zlib1g-dev dejagnu python3 git

sudo mkdir -p "$PREFIX"
sudo chown "$(id -u):$(id -g)" "$PREFIX"

python3 -m tools.toolchain.gcc \
  --source-dir "$SRC" \
  --build-dir "build/$(basename "$SRC")" \
  --prefix "$PREFIX" \
  --strip

echo "Installed size: $(du -sh "$PREFIX" | cut -f1)"
