"""The recipe for the clang-p2996 toolchain: clang, libc++ and compiler-rt.

`run_tests.py --build-compilers` calls build_clang() for a local build, where
the clang root is the LLVM build tree itself (build/clang-p2996). CI's toolchain
cache (.github/scripts/build-clang-toolchain.sh) runs this file, which builds
the same way and then installs a trimmed, stripped copy into a prefix:

    python3 -m tools.toolchain.clang --build-type Release \\
        --prefix /opt/toolchain-clang --strip

The toolchain holds what the tests and the tooling items in #94 need: clang and
its builtin headers, the fork's libc++ (with <meta>), compiler-rt's sanitizer
and profile runtimes, clang-tidy, llvm-cov, llvm-profdata and llvm-symbolizer.

The CI cache key hashes this file and common.py, so any change to the recipe
rebuilds the toolchain.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from .common import (
    ROOT,
    CompilerSpec,
    ensure_directory,
    ensure_submodule,
    log,
    require_success,
    run_command,
    run_command_streamed,
    strip_binaries,
)

DEFAULT_GCC_TOOLCHAIN = os.environ.get(
    "REFLECT_GCC_TOOLCHAIN",
    (
        "/opt/rh/gcc-toolset-13/root/usr"
        if os.path.exists("/opt/rh/gcc-toolset-13/root/usr")
        else "/usr"
    ),
)

# What the toolchain provides beyond the compiler: clang-tidy (L4), llvm-cov and
# llvm-profdata (T3), and llvm-symbolizer, which the sanitizers (T1) use to
# print stack traces.
TOOL_TARGETS = ("clang-tidy", "llvm-cov", "llvm-profdata", "llvm-symbolizer")
# The install components of the LLVM build that make up the CI prefix. On
# Darwin the driver passes ld the libLTO.dylib next to it, which ld warns about
# on every link if it's missing.
INSTALL_COMPONENTS = ("clang", "clang-resource-headers", *TOOL_TARGETS) + (
    ("LTO",) if sys.platform == "darwin" else ()
)
# LLVM's fix for the sanitizers hanging at startup on macOS 26 (llvm-project
# 2e7d07a, #182943), which clang-p2996 predates. Applied on Darwin only.
DARWIN_COMPILER_RT_PATCH = (
    Path(__file__).parent / "patches" / "compiler-rt-dyld-header.patch"
)


def detect_llvm_target() -> str:
    """Auto-detect LLVM target based on current machine architecture.

    Maps platform.machine() to LLVM target names:
    - ARM64/aarch64 -> AArch64
    - x86_64 -> X86
    """
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "AArch64"
    elif machine in ("x86_64", "amd64"):
        return "X86"
    else:
        # Default to all targets if architecture is unknown
        return "all"


def compiler_rt_dir(args: argparse.Namespace) -> Path:
    """compiler-rt's build tree: next to the libc++ runtimes one."""
    runtimes_dir = Path(args.clang_runtimes_dir).resolve()
    return runtimes_dir.with_name(f"{runtimes_dir.name}-compiler-rt")


def build_clang(
    spec: CompilerSpec, args: argparse.Namespace, build_type: str = "RelWithDebInfo"
) -> None:
    """Build the clang-p2996 fork, its tools and runtimes into the clang root.

    Faithful port of the reference recipe: build clang with a host clang + lld
    (keeps link memory low), then build libc++/libc++abi as a standalone
    runtimes project with the freshly built clang, emitting headers/libs into
    the clang root, then compiler-rt the same way, and finally sync the fork's
    <meta> header into place.

    build_type only applies to a fresh tree: an already configured one keeps
    its own. The runtimes are always RelWithDebInfo, so you can step into the
    fork's libc++ locally; CI strips them instead.
    """
    log("[clang] building the clang-p2996 fork + libc++ (this can take a long time)")
    ensure_submodule(spec.source_dir, args.verbose)
    ensure_directory(spec.build_dir)

    # Configure clang (idempotent; skip if already configured).
    if (spec.build_dir / "build.ninja").is_file():
        log(f"[clang] already configured ({spec.build_dir}); skipping cmake")
    else:
        log("[clang] configuring clang (cmake)...")
        configure = [
            "cmake",
            "-S",
            str(spec.source_dir / "llvm"),
            "-B",
            str(spec.build_dir),
            "-G",
            "Ninja",
            "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_C_COMPILER={args.host_cc}",
            f"-DCMAKE_CXX_COMPILER={args.host_cxx}",
            f"-DPython3_EXECUTABLE={args.python_executable}",
            f"-DLLVM_TARGETS_TO_BUILD={detect_llvm_target()}",
            f"-DLLVM_PARALLEL_COMPILE_JOBS={args.jobs}",
            # Linking clang is memory-hungry; serialize link steps.
            "-DLLVM_PARALLEL_LINK_JOBS=1",
        ]
        require_success(
            run_command_streamed(configure, cwd=ROOT, verbose=args.verbose),
            "clang configure",
        )

    # The `clang` target also produces the clang++/clang-cl symlinks; there is
    # no separate `clang++` ninja target. On Darwin, build the standalone LTO
    # target as well so the driver can find libLTO.dylib next to the toolchain.
    clang_targets = ["clang", *TOOL_TARGETS]
    if sys.platform == "darwin":
        clang_targets.append("LTO")
    log(f"[clang] building {' + '.join(clang_targets)} with ninja (-j{args.jobs})...")
    build = ["ninja", "-C", str(spec.build_dir), "-j", str(args.jobs), *clang_targets]
    require_success(
        run_command_streamed(build, cwd=ROOT, verbose=args.verbose), "clang build"
    )

    build_clang_runtimes(spec, args)
    build_compiler_rt(spec, args)
    log(f"[clang] done -> {spec.executable}")


def runtimes_c_flags(args: argparse.Namespace) -> str:
    """C/C++ flags for building runtimes with the fresh clang."""
    if sys.platform != "darwin" and args.gcc_toolchain:
        return f"--gcc-toolchain={args.gcc_toolchain}"
    return ""


def configure_when_changed(
    configure: list[str], build_dir: Path, label: str, args: argparse.Namespace
) -> None:
    """Run a cmake configure unless build_dir was configured with this command.

    Re-running cmake whenever the configure command changes, not only on a fresh
    tree, lets existing trees pick up new options.
    """
    stamp_file = build_dir / ".run_tests_configure.json"
    try:
        configured = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        configured = None
    if (build_dir / "build.ninja").is_file() and configured == configure:
        log(f"[{label}] already configured ({build_dir}); skipping cmake")
        return
    log(f"[{label}] configuring (cmake)...")
    require_success(
        run_command_streamed(configure, cwd=ROOT, verbose=args.verbose),
        f"{label} configure",
    )
    stamp_file.write_text(json.dumps(configure, indent=2) + "\n", encoding="utf-8")


def apply_source_patch(source_dir: Path, patch: Path, args: argparse.Namespace) -> None:
    """Apply patch to the source tree, unless it's already there.

    The patch stays applied, so ninja doesn't rebuild what it touches, and the
    submodule shows as modified. Once the submodule has the change itself, the
    reverse check passes and this does nothing.
    """
    applied = run_command(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=source_dir,
        verbose=args.verbose,
    )
    if applied.returncode == 0:
        return
    log(f"[patch] applying {patch.name} to {source_dir}")
    require_success(
        run_command(["git", "apply", str(patch)], cwd=source_dir, verbose=args.verbose),
        f"apply {patch.name}",
    )


def build_clang_runtimes(spec: CompilerSpec, args: argparse.Namespace) -> None:
    """Build libc++/libc++abi (plus libunwind off Darwin) and install <meta>.

    Uses the freshly built clang and points LLVM_BINARY_DIR at the clang root so
    the generated headers land in <root>/include/c++/v1 and libs in <root>/lib —
    exactly where the clang flag profile's -isystem/-L expect them.

    On Darwin, libc++abi uses the system unwinder in libSystem, as Apple's own
    libc++abi does, so no separate libunwind is built. libSystem re-exports
    /usr/lib/system/libunwind.dylib, and ld resolves that re-export by leaf name
    through -L, so any libunwind.dylib in <root>/lib is picked up instead and ld
    warns that it "couldn't be matched with any parent library" on every link.
    """
    runtimes_dir = Path(args.clang_runtimes_dir).resolve()
    built_clang = spec.build_dir / "bin" / "clang"
    built_clangxx = spec.build_dir / "bin" / "clang++"
    ensure_directory(runtimes_dir)

    c_flags = runtimes_c_flags(args)
    if sys.platform == "darwin":
        runtimes = "libcxx;libcxxabi"
        platform_options = ["-DLIBCXXABI_USE_LLVM_UNWINDER=OFF"]
    else:
        runtimes = "libcxx;libcxxabi;libunwind"
        platform_options = []
    configure = [
        "cmake",
        "-S",
        str(spec.source_dir / "runtimes"),
        "-B",
        str(runtimes_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_C_COMPILER={built_clang}",
        f"-DCMAKE_CXX_COMPILER={built_clangxx}",
        f"-DCMAKE_C_FLAGS={c_flags}",
        f"-DCMAKE_CXX_FLAGS={c_flags}",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DLLVM_ENABLE_RUNTIMES={runtimes}",
        *platform_options,
        f"-DLLVM_BINARY_DIR={spec.build_dir}",
        f"-DLLVM_CONFIG_PATH={spec.build_dir / 'bin' / 'llvm-config'}",
    ]
    configure_when_changed(configure, runtimes_dir, "runtimes", args)

    # The default target, not just cxx and cxxabi: `cmake --install` also
    # copies libunwind and libc++experimental.
    log(
        f"[runtimes] building {runtimes.replace(';', '/')} with ninja (-j{args.jobs})..."
    )
    build = ["ninja", "-C", str(runtimes_dir), "-j", str(args.jobs)]
    require_success(
        run_command_streamed(build, cwd=ROOT, verbose=args.verbose), "runtimes build"
    )

    if sys.platform == "darwin":
        # Trees built before libunwind was dropped on Darwin still have it in
        # <root>/lib. Nothing links it any more, but ld still finds it.
        for stale in sorted((spec.build_dir / "lib").glob("libunwind*")):
            log(f"[runtimes] removing stale {stale}")
            stale.unlink()

    log("[runtimes] syncing <meta> header into libc++ include dir...")
    sync_meta_header(spec.source_dir, spec.build_dir)


def build_compiler_rt(spec: CompilerSpec, args: argparse.Namespace) -> None:
    """Build compiler-rt's sanitizer and profile runtimes into clang's resource dir.

    T1 needs the sanitizers (-fsanitize=address,undefined) and T3 the profile
    runtime (-fprofile-instr-generate). It is a standalone runtimes build of its
    own, rather than a third runtime next to libc++, because building only the
    host target takes settings (CMAKE_*_COMPILER_TARGET) that would change how
    libc++ is built. Unlike the in-tree build, a standalone compiler-rt doesn't
    know clang's resource dir, so the output and install paths are given.
    """
    rt_dir = compiler_rt_dir(args)
    built_clang = spec.build_dir / "bin" / "clang"
    built_clangxx = spec.build_dir / "bin" / "clang++"
    ensure_directory(rt_dir)

    result = run_command(
        [str(built_clang), "-print-resource-dir"], cwd=None, verbose=args.verbose
    )
    require_success(result, "clang -print-resource-dir")
    resource_dir = Path(result.stdout.strip()).resolve()

    c_flags = runtimes_c_flags(args)
    if sys.platform == "darwin":
        apply_source_patch(spec.source_dir, DARWIN_COMPILER_RT_PATCH, args)
        # macOS only, not the other Apple platforms' slices.
        platform_options = [
            "-DCOMPILER_RT_ENABLE_IOS=OFF",
            "-DCOMPILER_RT_ENABLE_MACCATALYST=OFF",
        ]
    else:
        # The host target only; without this it also tries i386.
        result = run_command(
            [str(built_clang), "-dumpmachine"], cwd=None, verbose=args.verbose
        )
        require_success(result, "clang -dumpmachine")
        triple = result.stdout.strip()
        platform_options = [
            "-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON",
            f"-DCMAKE_C_COMPILER_TARGET={triple}",
            f"-DCMAKE_CXX_COMPILER_TARGET={triple}",
        ]
    configure = [
        "cmake",
        "-S",
        str(spec.source_dir / "runtimes"),
        "-B",
        str(rt_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_C_COMPILER={built_clang}",
        f"-DCMAKE_CXX_COMPILER={built_clangxx}",
        f"-DCMAKE_C_FLAGS={c_flags}",
        f"-DCMAKE_CXX_FLAGS={c_flags}",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DLLVM_ENABLE_RUNTIMES=compiler-rt",
        f"-DLLVM_BINARY_DIR={spec.build_dir}",
        f"-DCOMPILER_RT_OUTPUT_DIR={resource_dir}",
        # Typed STRING, or CMake makes the relative path absolute against the
        # current directory rather than the install prefix.
        f"-DCOMPILER_RT_INSTALL_PATH:STRING={resource_dir.relative_to(spec.build_dir)}",
        # ASan, plus the UBSan and LSan runtimes that every sanitizer build
        # includes, and the profile runtime. Nothing else.
        "-DCOMPILER_RT_SANITIZERS_TO_BUILD=asan",
        "-DCOMPILER_RT_BUILD_SANITIZERS=ON",
        "-DCOMPILER_RT_BUILD_PROFILE=ON",
        "-DCOMPILER_RT_BUILD_BUILTINS=OFF",
        "-DCOMPILER_RT_BUILD_CTX_PROFILE=OFF",
        "-DCOMPILER_RT_BUILD_GWP_ASAN=OFF",
        "-DCOMPILER_RT_BUILD_LIBFUZZER=OFF",
        "-DCOMPILER_RT_BUILD_MEMPROF=OFF",
        "-DCOMPILER_RT_BUILD_ORC=OFF",
        "-DCOMPILER_RT_BUILD_XRAY=OFF",
        *platform_options,
    ]
    configure_when_changed(configure, rt_dir, "compiler-rt", args)

    # The default target, not `compiler-rt`, which leaves out some of what
    # `cmake --install` copies (libclang_rt.stats).
    log(f"[compiler-rt] building with ninja (-j{args.jobs})...")
    build = ["ninja", "-C", str(rt_dir), "-j", str(args.jobs)]
    require_success(
        run_command_streamed(build, cwd=ROOT, verbose=args.verbose),
        "compiler-rt build",
    )


def sync_meta_header(source_dir: Path, root: Path) -> None:
    """Copy the fork's <meta> header into the installed libc++ include dir.

    The fork ships libcxx/include/meta but doesn't wire it into the runtimes
    header install, so it's copied in explicitly (also lets header-only edits
    take effect without a full runtimes rebuild).
    """
    meta_src = source_dir / "libcxx" / "include" / "meta"
    meta_dest = root / "include" / "c++" / "v1" / "meta"
    if not meta_src.is_file():
        raise SystemExit(f"<meta> header not found in fork: {meta_src}")
    ensure_directory(meta_dest.parent)
    shutil.copyfile(meta_src, meta_dest)


def install_clang(
    spec: CompilerSpec, args: argparse.Namespace, prefix: Path, strip: bool
) -> None:
    """Install what the toolchain consumers use, and nothing else, into prefix.

    That is the INSTALL_COMPONENTS of the LLVM build, all of the libc++ and
    compiler-rt runtimes builds, and <meta>.
    """
    log(f"[clang] installing into {prefix}")
    for component in INSTALL_COMPONENTS:
        install = ["cmake", "--install", str(spec.build_dir)]
        install += ["--component", component, "--prefix", str(prefix)]
        require_success(
            run_command(install, cwd=None, verbose=args.verbose),
            f"install {component}",
        )
    for build_dir in (Path(args.clang_runtimes_dir).resolve(), compiler_rt_dir(args)):
        install = ["cmake", "--install", str(build_dir), "--prefix", str(prefix)]
        require_success(
            run_command(install, cwd=None, verbose=args.verbose),
            f"install {build_dir.name}",
        )
    sync_meta_header(spec.source_dir, prefix)
    if strip:
        log(f"[clang] stripping {prefix}")
        strip_binaries(prefix, args.verbose)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the clang-p2996 toolchain the way `run_tests.py "
            "--build-compilers` does, and optionally install it into a prefix "
            "(how CI fills its toolchain cache)."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(ROOT / "clang-p2996"),
        help="clang-p2996 checkout. Default: the submodule.",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT / "build" / "clang-p2996"),
        help="LLVM build tree, which doubles as the clang root. Default: %(default)s.",
    )
    parser.add_argument(
        "--runtimes-dir",
        dest="clang_runtimes_dir",
        default=str(ROOT / "build" / "libcxx"),
        help=(
            "libc++ runtimes build tree; compiler-rt's is next to it, with "
            "-compiler-rt appended. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--build-type",
        default="RelWithDebInfo",
        help="CMAKE_BUILD_TYPE for clang itself, on a fresh tree. Default: %(default)s.",
    )
    parser.add_argument(
        "--prefix", help="Install the toolchain here after building it."
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip the installed toolchain (Linux; needs --prefix).",
    )
    parser.add_argument("--host-cc", default="clang")
    parser.add_argument("--host-cxx", default="clang++")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--gcc-toolchain", default=DEFAULT_GCC_TOOLCHAIN)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.strip and not args.prefix:
        parser.error("--strip needs --prefix")

    root = Path(args.root).resolve()
    spec = CompilerSpec(
        name="clang",
        source_dir=Path(args.source_dir).resolve(),
        build_dir=root,
        binary_dir=root,
        executable=root / "bin" / "clang++",
    )
    build_clang(spec, args, build_type=args.build_type)
    if args.prefix:
        install_clang(spec, args, Path(args.prefix).resolve(), args.strip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
