#!/usr/bin/env bash
# Builds the gcc-mirror fork and installs it, stripped, into a clean prefix for
# the CI toolchain cache.
#
# The toolchain cache key hashes this file (see toolchain-cache-gcc.yml), so
# any change here rebuilds the compiler, and nothing else does.
#
# Usage: build-gcc-toolchain.sh [source-dir] [prefix]
#   source-dir  gcc-mirror checkout (default: gcc-mirror)
#   prefix      install prefix (default: /opt/toolchain-gcc)
# The build directory is build/<basename of source-dir>, relative to the
# current directory.
set -euo pipefail

SRC="$(realpath "${1:-gcc-mirror}")"
PREFIX="${2:-/opt/toolchain-gcc}"
BUILD="build/$(basename "$SRC")"

# Build dependencies
sudo apt-get update
sudo apt-get install -y build-essential flex bison libgmp-dev libmpfr-dev \
  libmpc-dev texinfo zlib1g-dev dejagnu python3 git

# Build and install into a clean prefix
sudo mkdir -p "$PREFIX"
mkdir -p "$BUILD"

(cd "$SRC" && contrib/download_prerequisites)

cd "$BUILD"
"$SRC/configure" \
  --disable-bootstrap \
  --disable-multilib \
  --disable-nls \
  --enable-languages=c,c++ \
  "--prefix=$PREFIX"

make -j"$(nproc)"
sudo make install

# Strip binaries
sudo find "$PREFIX" -type f \( -name "*.so*" -o -perm /111 \) \
  -exec sh -c 'file "$1" | grep -q ELF && strip --strip-unneeded "$1"' _ {} \; 2>/dev/null || true
echo "Installed size: $(du -sh "$PREFIX" | cut -f1)"
