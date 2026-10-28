#!/usr/bin/env bash
# Prints, on one line, the parts of the macOS runner that a GCC built on it
# depends on: the SDK path it's configured with (--with-sysroot), the SDK's
# build version and the system linker's version. The path alone isn't enough:
# Xcode's SDK path is a symlink that stays the same when an image update
# changes the SDK or the linker inside the same Xcode.
#
# toolchain-cache-gcc.yml puts a hash of this in the macOS cache key, and the
# macOS test legs (toolchain-test.yml) compare it with their own runner's.
set -euo pipefail

sdk_path="$(xcrun --show-sdk-path)"
sdk_build="$(xcrun --show-sdk-build-version)"
ld_version="$(xcrun ld -v 2>&1 | sed -n 's/.*PROJECT://p')"
echo "SDK ${sdk_path} (${sdk_build}), ${ld_version}"
