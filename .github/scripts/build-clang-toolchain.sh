#!/usr/bin/env bash
# Builds the clang-p2996 fork and its libc++ runtimes, and installs them into
# a clean prefix for the CI toolchain cache.
#
# The toolchain cache key hashes this file (see toolchain-cache-clang.yml), so
# any change here rebuilds the compiler, and nothing else does.
#
# Usage: build-clang-toolchain.sh [source-dir] [prefix]
#   source-dir  clang-p2996 checkout (default: clang-p2996)
#   prefix      install prefix (default: /opt/toolchain-clang)
set -euo pipefail

SRC="${1:-clang-p2996}"
PREFIX="${2:-/opt/toolchain-clang}"

# Build dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build clang lld \
  python3 zlib1g-dev git

cd "$SRC"

# clang (Release)
cmake -G Ninja -S llvm -B build-llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="$(python3 -c "import platform; print('X86' if 'x86' in platform.machine() else 'AArch64')")" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_PARALLEL_LINK_JOBS=1
ninja -C build-llvm -j"$(nproc)" clang

# libc++ runtimes (Release)
cmake -G Ninja -S runtimes -B build-runtimes \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
  -DCMAKE_C_COMPILER="$(pwd)/build-llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(pwd)/build-llvm/bin/clang++" \
  -DLLVM_BINARY_DIR="$(pwd)/build-llvm" \
  -DLLVM_CONFIG_PATH="$(pwd)/build-llvm/bin/llvm-config" \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/build-runtimes/install"
ninja -C build-runtimes -j"$(nproc)" cxx cxxabi install

# Install into a clean prefix
sudo mkdir -p "$PREFIX"

sudo mkdir -p "$PREFIX/bin"
for bin in build-llvm/bin/clang \
           build-llvm/bin/clang++ \
           build-llvm/bin/llvm-config; do
  [ -f "$bin" ] && sudo cp -L "$bin" "$PREFIX/bin/"
done

sudo mkdir -p "$PREFIX/include"
sudo cp -r build-llvm/include/c++ "$PREFIX/include/"

sudo mkdir -p "$PREFIX/lib"
# Copy installed runtimes from CMAKE_INSTALL_PREFIX.
if [ -d build-runtimes/install/lib ]; then
  sudo cp -a build-runtimes/install/lib/. "$PREFIX/lib/"
fi

# Keep the LLVM lib directory snapshot as a fallback for transitive deps.
if [ -d build-llvm/lib ]; then
  sudo cp -a build-llvm/lib/. "$PREFIX/lib/"
fi

# Runtime libs may live under different subdirs depending on target/config.
mapfile -t RUNTIME_LIBS < <(find "$PREFIX/lib" \( -type f -o -type l \) \
  \( -name 'libc++*.so*' -o -name 'libc++*.a' -o -name 'libc++abi*.so*' -o -name 'libc++abi*.a' -o -name 'libunwind*.so*' -o -name 'libunwind*.a' \) 2>/dev/null || true)

if [ "${#RUNTIME_LIBS[@]}" -eq 0 ]; then
  echo "ERROR: No libc++ runtime artifacts found!"
  echo "Expected to find libc++/libc++abi/libunwind under $PREFIX/lib"
  echo "Contents of $PREFIX/lib:"
  find "$PREFIX/lib" -maxdepth 2 -type f | sort || true
  exit 1
else
  echo "Found ${#RUNTIME_LIBS[@]} runtime library candidates"
fi

# Ensure unversioned linker names exist (e.g. -lc++ -> libc++.so).
for lib in libc++ libc++abi libunwind; do
  if [ ! -e "$PREFIX/lib/${lib}.so" ]; then
    candidate="$(ls -1 "$PREFIX/lib/${lib}.so."* 2>/dev/null | head -n 1 || true)"
    if [ -n "$candidate" ]; then
      sudo ln -sf "$(basename "$candidate")" "$PREFIX/lib/${lib}.so"
    fi
  fi
done

# Clang builtin headers (e.g. stddef.h) live under lib/clang/*/include.
# Without this, libc++'s include_next <stddef.h> can fail.
sudo cp -r build-llvm/lib/clang "$PREFIX/lib/"

sudo cp libcxx/include/meta "$PREFIX/include/c++/v1/meta"

# Make sure plain libc++ (.so or .a) exists directly under lib/.
mapfile -t libcxx_candidates < <(find "$PREFIX/lib" \( -type f -o -type l \) \
  \( -name 'libc++.so*' -o -name 'libc++.a' \))

if [ "${#libcxx_candidates[@]}" -eq 0 ]; then
  echo "ERROR: Missing libc++ runtime under $PREFIX/lib"
  echo "Current $PREFIX/lib contents:"
  find "$PREFIX/lib" -maxdepth 3 -mindepth 1 -type f | sort || true
  exit 1
fi

if [ ! -e "$PREFIX/lib/libc++.so" ] && [ ! -e "$PREFIX/lib/libc++.a" ]; then
  preferred=""
  for c in "${libcxx_candidates[@]}"; do
    case "$c" in
      *.so|*.so.*)
        preferred="$c"
        break
        ;;
    esac
  done
  if [ -z "$preferred" ]; then
    preferred="${libcxx_candidates[0]}"
  fi
  if [ "$(dirname "$preferred")" != "$PREFIX/lib" ]; then
    sudo ln -sf "$preferred" "$PREFIX/lib/$(basename "$preferred")"
  fi
  if [[ "$preferred" == *.so || "$preferred" == *.so.* ]]; then
    sudo ln -sf "$(basename "$preferred")" "$PREFIX/lib/libc++.so"
  else
    sudo ln -sf "$(basename "$preferred")" "$PREFIX/lib/libc++.a"
  fi
fi

echo "Installed size: $(du -sh "$PREFIX" | cut -f1)"
