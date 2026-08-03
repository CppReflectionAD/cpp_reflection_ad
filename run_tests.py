#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
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
    executable: Path


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
        "--clang-build-dir",
        default=str(BUILD_ROOT / "clang-p2996"),
        help="Build directory for clang-p2996.",
    )
    parser.add_argument(
        "--gcc-build-dir",
        default=str(BUILD_ROOT / "gcc-mirror"),
        help="Build directory for gcc-mirror.",
    )
    parser.add_argument(
        "--clang-executable",
        help="Path to an existing clang++ executable to use instead of the default build output.",
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


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def selected_compilers(args: argparse.Namespace) -> list[str]:
    if args.compiler == "both":
        return ["clang", "gcc"]
    return [args.compiler]


def build_specs(args: argparse.Namespace) -> dict[str, CompilerSpec]:
    clang_build_dir = Path(args.clang_build_dir).resolve()
    gcc_build_dir = Path(args.gcc_build_dir).resolve()
    return {
        "clang": CompilerSpec(
            name="clang",
            source_dir=ROOT / "clang-p2996",
            build_dir=clang_build_dir,
            executable=(
                Path(args.clang_executable).resolve()
                if args.clang_executable
                else clang_build_dir / "bin" / "clang++"
            ),
        ),
        "gcc": CompilerSpec(
            name="gcc",
            source_dir=ROOT / "gcc-mirror",
            build_dir=gcc_build_dir,
            executable=(
                Path(args.gcc_executable).resolve()
                if args.gcc_executable
                else gcc_build_dir / "g++"
            ),
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


def build_clang(spec: CompilerSpec, args: argparse.Namespace) -> None:
    ensure_directory(spec.build_dir)
    configure = [
        "cmake",
        "-S",
        str(spec.source_dir / "llvm"),
        "-B",
        str(spec.build_dir),
        "-G",
        "Ninja",
        "-DLLVM_ENABLE_PROJECTS=clang",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    configure_result = run_command(configure, cwd=ROOT, verbose=args.verbose)
    require_success(configure_result, "clang configure")

    build = [
        "cmake",
        "--build",
        str(spec.build_dir),
        "--target",
        "clang",
        "clang++",
        "-j",
        str(args.jobs),
    ]
    build_result = run_command(build, cwd=ROOT, verbose=args.verbose)
    require_success(build_result, "clang build")


def build_gcc(spec: CompilerSpec, args: argparse.Namespace) -> None:
    ensure_directory(spec.build_dir)
    configure = [
        str(spec.source_dir / "configure"),
        "--disable-multilib",
        "--enable-languages=c,c++",
    ]
    configure_result = run_command(configure, cwd=spec.build_dir, verbose=args.verbose)
    require_success(configure_result, "gcc configure")

    build = ["make", "-j", str(args.jobs)]
    build_result = run_command(build, cwd=spec.build_dir, verbose=args.verbose)
    require_success(build_result, "gcc build")


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
        "Build the compiler first with --build-compilers or override the executable path."
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
        *args.extra_cxxflag,
        str(test_file),
        "-o",
        str(output_path),
    ]
    compile_result = run_command(compile_command, cwd=ROOT, verbose=args.verbose)

    run_result: CommandResult | None = None
    if compile_result.returncode == 0 and args.run_executables:
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
        build_selected_compilers(specs, args)

    results: list[TestResult] = []
    for compiler_name in selected_compilers(args):
        spec = specs[compiler_name]
        tests = discover_tests(args.tests, compiler_name)
        validate_compiler_executable(spec)
        for test_file in tests:
            result = compile_and_maybe_run(spec, test_file, args)
            results.append(result)
            print_test_result(result)

    failures = summarize(results, ran_executables=args.run_executables)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
