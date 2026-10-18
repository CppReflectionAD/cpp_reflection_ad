"""The recipe for the GCC reflection toolchain.

`run_tests.py --build-compilers` calls build_gcc() for a local build into
build/gcc-p2996/artifacts (on macOS from the gcc-darwin-reflect submodule).
CI's toolchain cache (.github/scripts/build-gcc-toolchain.sh) runs this file
with its own prefix, stripped:

    python3 -m tools.toolchain.gcc --build-dir build/gcc-mirror \\
        --prefix /opt/toolchain-gcc --strip

The CI cache key hashes this file and common.py, so any change to the recipe
rebuilds the toolchain.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .common import (
    ROOT,
    CompilerSpec,
    ensure_directory,
    ensure_submodule,
    log,
    render_command_failure,
    require_success,
    run_command,
    run_command_streamed,
    strip_binaries,
)


def macos_sdk_path(verbose: bool) -> str:
    """Resolve the active macOS SDK path via xcrun."""
    result = run_command(["xcrun", "--show-sdk-path"], cwd=ROOT, verbose=verbose)
    if result.returncode != 0:
        details = render_command_failure(result)
        raise SystemExit(f"Failed to resolve macOS SDK path\n\n{details}")

    sdk_path = result.stdout.strip()
    if not sdk_path:
        raise SystemExit("xcrun --show-sdk-path returned an empty SDK path")
    return sdk_path


def build_gcc(
    spec: CompilerSpec, args: argparse.Namespace, strip: bool = False
) -> None:
    """Build GCC and install it into spec.binary_dir, stripped if asked (CI)."""
    log(f"[gcc] building from {spec.source_dir} (this can take a long time)")
    ensure_submodule(spec.source_dir, args.verbose)
    ensure_directory(spec.build_dir)

    deps = ["contrib/download_prerequisites"]
    deps_result = run_command(deps, cwd=spec.source_dir, verbose=args.verbose)
    require_success(deps_result, "gcc download_prerequisites")

    log("[gcc] configuring gcc...")
    configure = [
        str(spec.source_dir / "configure"),
        "--disable-bootstrap",
        "--disable-multilib",
        "--disable-nls",
        "--enable-languages=c,c++",
        f"--prefix={spec.binary_dir}",
    ]
    if sys.platform == "darwin":
        configure.append(f"--with-sysroot={macos_sdk_path(args.verbose)}")
    configure_result = run_command_streamed(
        configure, cwd=spec.build_dir, verbose=args.verbose
    )
    require_success(configure_result, "gcc configure")

    log(f"[gcc] building with make (-j{args.jobs})...")
    build = ["make", "-j", str(args.jobs)]
    build_result = run_command_streamed(build, cwd=spec.build_dir, verbose=args.verbose)
    require_success(build_result, "gcc build")

    log("[gcc] installing...")
    install = ["make", "install"]
    install_result = run_command_streamed(
        install, cwd=spec.build_dir, verbose=args.verbose
    )
    require_success(install_result, "gcc install")

    if strip:
        log(f"[gcc] stripping {spec.binary_dir}")
        strip_binaries(spec.binary_dir, args.verbose)

    log(f"[gcc] done -> {spec.executable}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build and install the GCC reflection fork the way `run_tests.py "
            "--build-compilers` does (how CI fills its toolchain cache)."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(ROOT / "gcc-mirror"),
        help="GCC checkout. Default: the gcc-mirror submodule.",
    )
    parser.add_argument(
        "--build-dir",
        default=str(ROOT / "build" / "gcc-p2996"),
        help="GCC build tree. Default: %(default)s.",
    )
    parser.add_argument(
        "--prefix", help="Install prefix. Default: <build-dir>/artifacts."
    )
    parser.add_argument(
        "--strip", action="store_true", help="Strip the installed toolchain (Linux)."
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    prefix = Path(args.prefix).resolve() if args.prefix else build_dir / "artifacts"
    spec = CompilerSpec(
        name="gcc",
        source_dir=Path(args.source_dir).resolve(),
        build_dir=build_dir,
        binary_dir=prefix,
        executable=prefix / "bin" / "g++",
    )
    build_gcc(spec, args, strip=args.strip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
