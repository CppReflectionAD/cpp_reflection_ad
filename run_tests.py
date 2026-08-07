#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
BUILD_ROOT = ROOT / "build"
ARTIFACTS_DIR = BUILD_ROOT / "artifacts"
CLANG_ONLY_DIR = "clang_only"
GCC_ONLY_DIR = "gcc_only"

# The clang reflection fork is built from the clang-p2996 submodule into this
# repo's own build/ tree, so the repo is self-contained (no dependency on any
# externally pre-built compiler). The clang "root" is that build tree: it holds
# bin/clang++, include/c++/v1/meta, and lib/libc++.so, and the clang flag
# profile is derived from it. Override with --clang-root / CLANG_P2996_ROOT to
# point at a compiler built elsewhere.
DEFAULT_CLANG_ROOT = os.environ.get(
    "CLANG_P2996_ROOT", str(BUILD_ROOT / "clang-p2996")
)
# Standalone libc++/libc++abi/libunwind (runtimes) build tree. Built with the
# freshly built clang; its headers/libs are emitted into the clang root via
# LLVM_BINARY_DIR (see build_clang_runtimes).
DEFAULT_CLANG_RUNTIMES_DIR = str(BUILD_ROOT / "libcxx")
DEFAULT_GCC_TOOLCHAIN = os.environ.get(
    "REFLECT_GCC_TOOLCHAIN", "/opt/rh/gcc-toolset-13/root/usr"
)

# Per-test compile flags are declared inline via a `// TEST-FLAGS: ...` comment
# in the first few lines of a test (e.g. benchmarks that need -O2).
TEST_FLAGS_DIRECTIVE = "// TEST-FLAGS:"
TEST_FLAGS_SCAN_LINES = 10


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TestResult:
    compiler: str
    test_file: Path
    compile_result: CommandResult
    run_result: CommandResult | None

    @property
    def compile_ok(self) -> bool:
        return self.compile_result.returncode == 0

    @property
    def run_ok(self) -> bool:
        return self.run_result is None or self.run_result.returncode == 0


@dataclass(frozen=True)
class CompilerSpec:
    name: str
    source_dir: Path
    build_dir: Path
    binary_dir: Path
    executable: Path
    # Reflection flag profile for this compiler: everything the compiler needs
    # beyond -std and the source/-o pair (reflection features, stdlib, include
    # and library paths). This is what lets the same test compile under a
    # different compiler by simply selecting a different profile.
    cxxflags: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reflection compilers and compile tests under tests/."
    )
    parser.add_argument(
        "--compiler",
        choices=("clang", "gcc", "both"),
        default="both",
        help="Compiler to use for the test run.",
    )
    parser.add_argument(
        "--build-compilers",
        action="store_true",
        help="Build the selected compiler submodules before running tests.",
    )
    parser.add_argument(
        "--run-executables",
        action="store_true",
        help="Run successfully compiled test binaries and report runtime failures.",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        metavar="PATTERN",
        help="Optional glob patterns relative to tests/, for example '*.cpp' or 'expr/*.cpp'.",
    )
    parser.add_argument(
        "--std",
        default="c++2c",
        help="C++ language mode to pass to the selected compiler(s). Default: c++2c.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel build jobs. Default: host CPU count.",
    )
    parser.add_argument(
        "--clang-root",
        default=DEFAULT_CLANG_ROOT,
        help=(
            "Root of the built clang-p2996 (holds bin/clang++, include/c++/v1/meta, "
            "lib/libc++.so); doubles as the clang build tree. The clang reflection "
            "flag profile is derived from this. "
            f"Default: {DEFAULT_CLANG_ROOT} (env CLANG_P2996_ROOT)."
        ),
    )
    parser.add_argument(
        "--clang-runtimes-dir",
        default=DEFAULT_CLANG_RUNTIMES_DIR,
        help=(
            "Build directory for the standalone libc++/libc++abi/libunwind runtimes. "
            f"Default: {DEFAULT_CLANG_RUNTIMES_DIR}."
        ),
    )
    parser.add_argument(
        "--host-cxx",
        default=os.environ.get("REFLECT_HOST_CXX", "clang++"),
        help="Host C++ compiler used to build clang. Default: clang++ (env REFLECT_HOST_CXX).",
    )
    parser.add_argument(
        "--host-cc",
        default=os.environ.get("REFLECT_HOST_CC", "clang"),
        help="Host C compiler used to build clang. Default: clang (env REFLECT_HOST_CC).",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter passed to the LLVM build (Python3_EXECUTABLE).",
    )
    parser.add_argument(
        "--gcc-toolchain",
        default=DEFAULT_GCC_TOOLCHAIN,
        help=(
            "GCC toolchain clang uses for the C++ runtime/linker "
            f"(--gcc-toolchain=...). Default: {DEFAULT_GCC_TOOLCHAIN}."
        ),
    )
    parser.add_argument(
        "--gcc-build-dir",
        default=str(BUILD_ROOT / "gcc-mirror"),
        help="Build directory for gcc-mirror.",
    )
    parser.add_argument(
        "--clang-executable",
        help="Path to a clang++ executable, overriding <clang-root>/bin/clang++.",
    )
    parser.add_argument(
        "--gcc-executable",
        help="Path to an existing g++ executable to use instead of the default build output.",
    )
    parser.add_argument(
        "--extra-cxxflag",
        action="append",
        default=[],
        help="Additional compiler flag. Repeat to pass multiple flags.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print commands before running them.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    """Print a progress banner (flushed immediately) for long-running steps."""
    print(f"==> {message}", flush=True)


def run_command(command: list[str], cwd: Path | None, verbose: bool) -> CommandResult:
    if verbose:
        location = str(cwd) if cwd is not None else str(ROOT)
        print(f"[{location}] $ {shlex.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_command_streamed(
    command: list[str], cwd: Path | None, verbose: bool
) -> CommandResult:
    """Run a command with its stdout/stderr inherited (live) rather than captured.

    Used for the long build steps (submodule checkout, cmake, ninja) so the user
    sees progress as it happens. Output isn't captured, so on failure we rely on
    what was already printed.
    """
    location = str(cwd) if cwd is not None else str(ROOT)
    if verbose:
        print(f"[{location}] $ {shlex.join(command)}", flush=True)
    else:
        print(f"    $ {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd)
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout="",
        stderr="",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def selected_compilers(args: argparse.Namespace) -> list[str]:
    if args.compiler == "both":
        return ["clang", "gcc"]
    return [args.compiler]


def clang_cxxflags(clang_root: Path, gcc_toolchain: str) -> tuple[str, ...]:
    """Reflection flag profile for the clang-p2996 fork.

    Mirrors DEMO_FLAGS in the top-level Makefile: the reflection features, the
    libc++ stdlib, the -isystem for the installed <meta> header, and the
    library/rpath for libc++ (all rooted at the built compiler tree).
    """
    libcxx_inc = clang_root / "include" / "c++" / "v1"
    libcxx_lib = clang_root / "lib"
    return (
        "-freflection",
        "-fparameter-reflection",
        "-fexpansion-statements",
        "-stdlib=libc++",
        f"--gcc-toolchain={gcc_toolchain}",
        "-isystem",
        str(libcxx_inc),
        f"-L{libcxx_lib}",
        f"-Wl,-rpath,{libcxx_lib}",
    )


def build_specs(args: argparse.Namespace) -> dict[str, CompilerSpec]:
    clang_root = Path(args.clang_root).resolve()
    gcc_build_dir = Path(args.gcc_build_dir).resolve()
    gcc_binary_dir = gcc_build_dir / "artifacts"
    return {
        "clang": CompilerSpec(
            name="clang",
            source_dir=ROOT / "clang-p2996",
            # The clang root doubles as its build tree (runtimes + <meta> land here).
            build_dir=clang_root,
            binary_dir=clang_root,
            executable=(
                Path(args.clang_executable).resolve()
                if args.clang_executable
                else clang_root / "bin" / "clang++"
            ),
            cxxflags=clang_cxxflags(clang_root, args.gcc_toolchain),
        ),
        "gcc": CompilerSpec(
            name="gcc",
            source_dir=ROOT / "gcc-mirror",
            build_dir=gcc_build_dir,
            binary_dir=gcc_binary_dir,
            executable=(
                Path(args.gcc_executable).resolve()
                if args.gcc_executable
                else gcc_binary_dir / "bin" / "g++"
            ),
            cxxflags=("-std=c++26", "-freflection"),
        ),
    }


def is_test_applicable(test_file: Path, compiler_name: str) -> bool:
    relative_path = test_file.relative_to(TESTS_DIR)
    if not relative_path.parts:
        return True

    top_level_dir = relative_path.parts[0]
    if top_level_dir == CLANG_ONLY_DIR:
        return compiler_name == "clang"
    if top_level_dir == GCC_ONLY_DIR:
        return compiler_name == "gcc"
    return True


def discover_tests(patterns: list[str] | None, compiler_name: str) -> list[Path]:
    if not TESTS_DIR.is_dir():
        raise SystemExit(f"Tests directory not found: {TESTS_DIR}")

    if not patterns:
        candidates = sorted(TESTS_DIR.rglob("*.cpp"))
    else:
        seen: set[Path] = set()
        candidates = []
        for pattern in patterns:
            for test_file in sorted(TESTS_DIR.glob(pattern)):
                if (
                    test_file.is_file()
                    and test_file.suffix == ".cpp"
                    and test_file not in seen
                ):
                    seen.add(test_file)
                    candidates.append(test_file)

    tests = [
        test_file
        for test_file in candidates
        if test_file.is_file() and is_test_applicable(test_file, compiler_name)
    ]

    if not tests:
        requested = ", ".join(patterns or ["*.cpp"])
        raise SystemExit(
            f"No test files found for compiler {compiler_name} and pattern(s): {requested}"
        )
    return tests


def parse_test_flags(test_file: Path) -> list[str]:
    """Read an inline `// TEST-FLAGS: ...` directive from the top of a test.

    Lets a single test declare extra compile flags (e.g. `-O2` for benchmarks)
    without special-casing it in the harness. Returns [] if none is present.
    """
    try:
        with test_file.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(TEST_FLAGS_SCAN_LINES):
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith(TEST_FLAGS_DIRECTIVE):
                    return shlex.split(stripped[len(TEST_FLAGS_DIRECTIVE):])
    except OSError:
        pass
    return []


def ensure_submodule(source_dir: Path, args: argparse.Namespace) -> None:
    """Check out a submodule on demand so the repo builds from a bare clone.

    A fresh clone leaves clang-p2996/gcc-mirror as empty submodule dirs; the
    build needs their sources. Idempotent: a no-op once populated.
    """
    if (source_dir / ".git").exists() or any(source_dir.iterdir()):
        return
    log(f"[{source_dir.name}] checking out submodule (large clone, be patient)...")
    result = run_command_streamed(
        ["git", "submodule", "update", "--init", "--progress", str(source_dir.name)],
        cwd=ROOT,
        verbose=args.verbose,
    )
    require_success(result, f"submodule checkout ({source_dir.name})")


def build_clang(spec: CompilerSpec, args: argparse.Namespace) -> None:
    """Build the clang-p2996 fork and its libc++ runtimes into the clang root.

    Faithful port of the reference recipe: build clang with a host clang + lld
    (keeps link memory low), then build libc++/libc++abi as a standalone
    runtimes project with the freshly built clang, emitting headers/libs into
    the clang root, and finally sync the fork's <meta> header into place.
    """
    log("[clang] building the clang-p2996 fork + libc++ (this can take a long time)")
    ensure_submodule(spec.source_dir, args)
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
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DLLVM_USE_LINKER=lld",
            f"-DCMAKE_C_COMPILER={args.host_cc}",
            f"-DCMAKE_CXX_COMPILER={args.host_cxx}",
            f"-DPython3_EXECUTABLE={args.python_executable}",
            "-DLLVM_TARGETS_TO_BUILD=X86",
            f"-DLLVM_PARALLEL_COMPILE_JOBS={args.jobs}",
            # Linking clang is memory-hungry; serialize link steps.
            "-DLLVM_PARALLEL_LINK_JOBS=1",
        ]
        require_success(
            run_command_streamed(configure, cwd=ROOT, verbose=args.verbose),
            "clang configure",
        )

    # The `clang` target also produces the clang++/clang-cl symlinks; there is
    # no separate `clang++` ninja target.
    log(f"[clang] building clang (+ clang++ symlink) with ninja (-j{args.jobs})...")
    build = ["ninja", "-C", str(spec.build_dir), "-j", str(args.jobs), "clang"]
    require_success(run_command_streamed(build, cwd=ROOT, verbose=args.verbose), "clang build")

    build_clang_runtimes(spec, args)
    log(f"[clang] done -> {spec.executable}")


def build_clang_runtimes(spec: CompilerSpec, args: argparse.Namespace) -> None:
    """Build libc++/libc++abi/libunwind and install the <meta> header.

    Uses the freshly built clang and points LLVM_BINARY_DIR at the clang root so
    the generated headers land in <root>/include/c++/v1 and libs in <root>/lib —
    exactly where the clang flag profile's -isystem/-L expect them.
    """
    runtimes_dir = Path(args.clang_runtimes_dir).resolve()
    built_clang = spec.build_dir / "bin" / "clang"
    built_clangxx = spec.build_dir / "bin" / "clang++"
    ensure_directory(runtimes_dir)

    if (runtimes_dir / "build.ninja").is_file():
        log(f"[runtimes] already configured ({runtimes_dir}); skipping cmake")
    else:
        log("[runtimes] configuring libc++/libc++abi/libunwind (cmake)...")
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
            f"-DCMAKE_C_FLAGS=--gcc-toolchain={args.gcc_toolchain}",
            f"-DCMAKE_CXX_FLAGS=--gcc-toolchain={args.gcc_toolchain}",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi;libunwind",
            f"-DLLVM_BINARY_DIR={spec.build_dir}",
            f"-DLLVM_CONFIG_PATH={spec.build_dir / 'bin' / 'llvm-config'}",
        ]
        require_success(
            run_command_streamed(configure, cwd=ROOT, verbose=args.verbose),
            "runtimes configure",
        )

    log(f"[runtimes] building libc++/libc++abi with ninja (-j{args.jobs})...")
    build = ["ninja", "-C", str(runtimes_dir), "-j", str(args.jobs), "cxx", "cxxabi"]
    require_success(
        run_command_streamed(build, cwd=ROOT, verbose=args.verbose), "runtimes build"
    )

    log("[runtimes] syncing <meta> header into libc++ include dir...")
    sync_meta_header(spec)


def sync_meta_header(spec: CompilerSpec) -> None:
    """Copy the fork's <meta> header into the installed libc++ include dir.

    The fork ships libcxx/include/meta but doesn't wire it into the runtimes
    header install, so it's copied in explicitly (also lets header-only edits
    take effect without a full runtimes rebuild).
    """
    meta_src = spec.source_dir / "libcxx" / "include" / "meta"
    meta_dest = spec.build_dir / "include" / "c++" / "v1" / "meta"
    if not meta_src.is_file():
        raise SystemExit(f"<meta> header not found in fork: {meta_src}")
    ensure_directory(meta_dest.parent)
    shutil.copyfile(meta_src, meta_dest)


def build_gcc(spec: CompilerSpec, args: argparse.Namespace) -> None:
    log("[gcc] building the gcc-mirror fork (this can take a long time)")
    ensure_submodule(spec.source_dir, args)
    ensure_directory(spec.build_dir)

    deps = ["contrib/download_prerequisites"]
    deps_result = run_command(deps, cwd=spec.source_dir, verbose=args.verbose)

    log("[gcc] configuring gcc...")
    configure = [
        str(spec.source_dir / "configure"),
        "--disable-bootstrap", 
        "--disable-multilib",
        "--disable-nls",
        "--enable-languages=c,c++",
        f"--prefix={spec.binary_dir}",
    ]
    configure_result = run_command_streamed(
        configure, cwd=spec.build_dir, verbose=args.verbose
    )
    require_success(configure_result, "gcc configure")

    log(f"[gcc] building with make (-j{args.jobs})...")
    build = ["make", "-j", str(args.jobs)]
    build_result = run_command_streamed(build, cwd=spec.build_dir, verbose=args.verbose)
    require_success(build_result, "gcc build")

    log(f"[gcc] installing...")
    install = ["make", "install"]
    install_result = run_command_streamed(install, cwd=spec.build_dir, verbose=args.verbose)
    require_success(install_result, "gcc install")

    log(f"[gcc] done -> {spec.executable}")


def build_selected_compilers(
    specs: dict[str, CompilerSpec], args: argparse.Namespace
) -> None:
    for compiler_name in selected_compilers(args):
        spec = specs[compiler_name]
        if compiler_name == "clang":
            build_clang(spec, args)
        else:
            build_gcc(spec, args)


def require_success(result: CommandResult, label: str) -> None:
    if result.returncode == 0:
        return
    details = render_command_failure(result)
    raise SystemExit(f"{label} failed\n\n{details}")


def validate_compiler_executable(spec: CompilerSpec) -> None:
    if spec.executable.is_file():
        return
    raise SystemExit(
        f"Compiler executable not found for {spec.name}: {spec.executable}\n"
        f"Build it from the submodule with --build-compilers --compiler {spec.name}, "
        "or point --clang-root at an existing build."
    )


def compile_and_maybe_run(
    spec: CompilerSpec,
    test_file: Path,
    args: argparse.Namespace,
) -> TestResult:
    compiler_artifacts_dir = ARTIFACTS_DIR / spec.name
    ensure_directory(compiler_artifacts_dir)

    output_name = test_file.relative_to(TESTS_DIR).with_suffix("")
    output_path = compiler_artifacts_dir / output_name
    ensure_directory(output_path.parent)

    compile_command = [
        str(spec.executable),
        f"-std={args.std}",
        *spec.cxxflags,
        *parse_test_flags(test_file),
        *args.extra_cxxflag,
        str(test_file),
        "-o",
        str(output_path),
    ]
    relative_path = test_file.relative_to(ROOT)
    log(f"[{spec.name}] compiling {relative_path} ...")
    compile_result = run_command(compile_command, cwd=ROOT, verbose=args.verbose)

    run_result: CommandResult | None = None
    if compile_result.returncode == 0 and args.run_executables:
        log(f"[{spec.name}] running {relative_path} ...")
        run_result = run_command([str(output_path)], cwd=ROOT, verbose=args.verbose)

    return TestResult(
        compiler=spec.name,
        test_file=test_file,
        compile_result=compile_result,
        run_result=run_result,
    )


def render_command_failure(result: CommandResult) -> str:
    lines = [
        f"command: {shlex.join(result.command)}",
        f"exit code: {result.returncode}",
    ]
    if result.stdout.strip():
        lines.append("stdout:")
        lines.append(result.stdout.rstrip())
    if result.stderr.strip():
        lines.append("stderr:")
        lines.append(result.stderr.rstrip())
    return "\n".join(lines)


def print_test_result(result: TestResult) -> None:
    relative_path = result.test_file.relative_to(ROOT)
    if not result.compile_ok:
        print(f"[FAIL][{result.compiler}][compile] {relative_path}")
        print(indent_block(render_command_failure(result.compile_result)))
        return

    if result.run_result is not None and not result.run_ok:
        print(f"[FAIL][{result.compiler}][run] {relative_path}")
        print(indent_block(render_command_failure(result.run_result)))
        return

    phase = "compile+run" if result.run_result is not None else "compile"
    print(f"[PASS][{result.compiler}][{phase}] {relative_path}")


def indent_block(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def summarize(results: list[TestResult], ran_executables: bool) -> int:
    overall_failures = 0
    print()
    print("Summary")
    print("-------")
    for compiler_name in sorted({result.compiler for result in results}):
        compiler_results = [
            result for result in results if result.compiler == compiler_name
        ]
        compile_failures = sum(not result.compile_ok for result in compiler_results)
        run_failures = sum(
            result.compile_ok and not result.run_ok for result in compiler_results
        )
        passes = sum(result.compile_ok and result.run_ok for result in compiler_results)
        overall_failures += compile_failures + run_failures
        summary = (
            f"{compiler_name}: {passes} passed, " f"{compile_failures} compile failures"
        )
        if ran_executables:
            summary += f", {run_failures} runtime failures"
        print(summary)
    return overall_failures


def main() -> int:
    args = parse_args()
    specs = build_specs(args)

    if args.build_compilers:
        log(f"building compiler(s): {', '.join(selected_compilers(args))}")
        build_selected_compilers(specs, args)

    results: list[TestResult] = []
    for compiler_name in selected_compilers(args):
        spec = specs[compiler_name]
        tests = discover_tests(args.tests, compiler_name)
        validate_compiler_executable(spec)
        log(f"[{spec.name}] compiling {len(tests)} test(s) with {spec.executable}")
        for test_file in tests:
            result = compile_and_maybe_run(spec, test_file, args)
            results.append(result)
            print_test_result(result)

    failures = summarize(results, ran_executables=args.run_executables)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
