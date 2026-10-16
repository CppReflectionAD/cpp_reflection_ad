#!/usr/bin/env python3
"""Build the reflection compilers, then build and test this repo through CMake.

The test half is a thin driver over CMakePresets.json: for each selected
compiler and build type it configures the matching preset (clang-release,
gcc-debug, ...), builds, and runs ctest, so a local run and CI run the same
commands. The flags, the test list and the expected failures all live in
CMake. --build-compilers builds the compiler forks from their submodules
first, with the recipes in tools/toolchain/ that CI's toolchain cache uses too.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tools.toolchain import clang as clang_recipe
from tools.toolchain import gcc as gcc_recipe
from tools.toolchain.clang import DEFAULT_GCC_TOOLCHAIN
from tools.toolchain.common import (
    CommandResult,
    CompilerSpec,
    ensure_directory,
    log,
    render_command_failure,
    require_success,
    run_command,
    run_command_streamed,
)

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
BENCHMARKS_DIR = ROOT / "benchmarks"
BUILD_ROOT = ROOT / "build"
# The presets' own binaryDir is build/cmake/<preset>; using the same place
# means this script, `cmake --preset` and VS Code share build trees.
CMAKE_BUILD_ROOT = BUILD_ROOT / "cmake"
BUILD_TYPES = ("debug", "relwithdebinfo", "release")
# compile_check (test_simple/test_simple_cmake.cmake) names these tests
# reflection_ad.<path under tests/, with / as .>; every other test is named
# after its file, up to the first '.'. A known_gap/ file with a
# `// TEST-PASSES-ON:` line is also an ordinary test on the other compiler.
COMPILE_CHECK_DIRS = ("static_fail", "known_gap")

# The clang reflection fork is built from the clang-p2996 submodule (override
# with --clang-source-dir / REFLECT_CLANG_SOURCE_DIR) into this repo's own
# build/ tree, so the repo is self-contained (no dependency on any
# externally pre-built compiler). The clang "root" is that build tree: it holds
# bin/clang++, include/c++/v1/meta, and lib/libc++.so, and CMake derives the
# clang flag profile from the compiler's location in it. Override with
# --clang-root / CLANG_P2996_ROOT to point at a compiler built elsewhere.
DEFAULT_CLANG_SOURCE_DIR = os.environ.get(
    "REFLECT_CLANG_SOURCE_DIR", str(ROOT / "clang-p2996")
)
DEFAULT_CLANG_ROOT = os.environ.get("CLANG_P2996_ROOT", str(BUILD_ROOT / "clang-p2996"))
# Standalone libc++/libc++abi (plus libunwind off Darwin) runtimes build tree.
# Built with the freshly built clang; its headers/libs are emitted into the
# clang root via LLVM_BINARY_DIR (see tools/toolchain/clang.py), and
# compiler-rt is built next to it, in <runtimes dir>-compiler-rt.
DEFAULT_CLANG_RUNTIMES_DIR = str(BUILD_ROOT / "libcxx")
DEFAULT_GCC_BUILD_DIR = os.environ.get(
    "REFLECT_GCC_BUILD_DIR", str(BUILD_ROOT / "gcc-p2996")
)

if sys.platform == "darwin":
    DEFAULT_GCC_SOURCE_DIR = os.environ.get(
        "REFLECT_GCC_SOURCE_DIR", str(ROOT / "gcc-darwin-reflect")
    )
    DEFAULT_GCC_PATCHES_DIR = os.environ.get(
        "REFLECT_GCC_PATCHES_DIR", "/tmp/gcc-reflect-patches"
    )
    DEFAULT_GCC_SYNC_FROM = os.environ.get(
        "REFLECT_GCC_SYNC_FROM", str(ROOT / "gcc-mirror")
    )
else:
    DEFAULT_GCC_SOURCE_DIR = os.environ.get(
        "REFLECT_GCC_SOURCE_DIR", str(ROOT / "gcc-mirror")
    )
    DEFAULT_GCC_PATCHES_DIR = os.environ.get("REFLECT_GCC_PATCHES_DIR", "")
    DEFAULT_GCC_SYNC_FROM = os.environ.get("REFLECT_GCC_SYNC_FROM", "")


@dataclass(frozen=True)
class Selection:
    """What to build and run in one configured build tree."""

    tests: list[str] | None  # ctest names; None runs every test
    benchmarks: list[str]  # benchmark targets to build (and maybe run)


@dataclass
class RunResult:
    compiler: str
    build_type: str
    build_dir: Path
    status: str = "passed"  # or the step that failed, or "skipped"
    detail: str = ""
    # Requested tests or benchmarks this compiler doesn't have.
    missing: list[str] = field(default_factory=list)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse this script's options; everything after `--` goes to CMake."""
    if "--" in argv:
        split = argv.index("--")
        argv, cmake_args = argv[:split], argv[split + 1 :]
    else:
        cmake_args = []

    parser = argparse.ArgumentParser(
        description=(
            "Build the reflection compilers, then configure, build and test "
            "this repo with the CMake presets (one per compiler and build "
            "type). Arguments after `--` are passed to the CMake configure "
            "step, e.g. `-- -DCMAKE_VERBOSE_MAKEFILE=ON`."
        )
    )
    parser.add_argument(
        "--compiler",
        choices=("clang", "gcc", "both"),
        default="both",
        help="Compiler to use for the test run.",
    )
    parser.add_argument(
        "--build-type",
        choices=(*BUILD_TYPES, "all"),
        help=(
            "CMake build type, i.e. which preset (<compiler>-<build type>) to "
            "use; 'all' runs each in turn. Default: release, or debug with "
            "--debugger."
        ),
    )
    parser.add_argument(
        "--build-compilers",
        action="store_true",
        help="Build the selected compiler submodules before running tests.",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        metavar="TEST",
        help=(
            "Tests to build and run, by ctest name ('ad_scalar_demo', "
            "'reflection_ad.static_fail.inverse_of_non_invertible') or by glob "
            "relative to tests/ ('common/ad_scalar_demo.cpp', 'clang_only/*'). "
            "Default: all of them."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        metavar="BENCHMARK",
        help=(
            "Build only these benchmarks, by target name ('ad_bench') or by "
            "glob relative to benchmarks/ ('*.cpp', 'clang_only/*.cpp'); no "
            "names means all of them. On its own it skips the tests. Add "
            "--run-executables to run them too."
        ),
    )
    parser.add_argument(
        "--run-executables",
        action="store_true",
        help=(
            "Run the benchmarks selected with --benchmarks, in the build "
            "directory (ITERATIONS=... tunes some). Tests always run, through "
            "ctest."
        ),
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Configure and build, but don't run ctest or any benchmark.",
    )
    parser.add_argument(
        "--debugger",
        choices=("gdb", "lldb"),
        help=(
            "Build the one test given with --tests and start it under this "
            "debugger, instead of running ctest."
        ),
    )
    parser.add_argument(
        "-B",
        "--build-dir",
        metavar="DIR",
        help=(
            "Put each build tree in DIR/<preset> instead of build/cmake/<preset> "
            "(the presets' own location, shared with `cmake --preset` and VS Code)."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete each selected build tree before configuring it.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel build and test jobs. Default: host CPU count.",
    )
    parser.add_argument(
        "--clang-source-dir",
        default=DEFAULT_CLANG_SOURCE_DIR,
        help=(
            "Path to the clang-p2996 source tree (an llvm-project checkout) used "
            "for clang builds. "
            f"Default: {DEFAULT_CLANG_SOURCE_DIR} (env REFLECT_CLANG_SOURCE_DIR)."
        ),
    )
    parser.add_argument(
        "--clang-root",
        default=DEFAULT_CLANG_ROOT,
        help=(
            "Root of the built clang-p2996 (holds bin/clang++, include/c++/v1/meta, "
            "lib/libc++.so); doubles as the clang build tree. CMake derives the "
            "clang flag profile from the compiler's place in it. "
            f"Default: {DEFAULT_CLANG_ROOT} (env CLANG_P2996_ROOT)."
        ),
    )
    parser.add_argument(
        "--clang-runtimes-dir",
        default=DEFAULT_CLANG_RUNTIMES_DIR,
        help=(
            "Build directory for the standalone libc++/libc++abi runtimes "
            "(plus libunwind except on macOS); compiler-rt's is next to it, "
            "with -compiler-rt appended. "
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
            "GCC toolchain the clang runtimes build uses for the C++ "
            f"runtime/linker (--gcc-toolchain=...). Default: {DEFAULT_GCC_TOOLCHAIN}."
        ),
    )
    parser.add_argument(
        "--gcc-source-dir",
        default=DEFAULT_GCC_SOURCE_DIR,
        help=(
            "Path to the GCC source tree used for gcc builds. "
            f"Default: {DEFAULT_GCC_SOURCE_DIR}."
        ),
    )
    parser.add_argument(
        "--gcc-build-dir",
        default=DEFAULT_GCC_BUILD_DIR,
        help=f"Build directory for gcc. Default: {DEFAULT_GCC_BUILD_DIR}.",
    )
    parser.add_argument(
        "--gcc-patches-dir",
        default=DEFAULT_GCC_PATCHES_DIR,
        help=(
            "Optional directory for exported GCC reflection patches. "
            "Used by macOS darwin-reflect workflows. "
            f"Default: {DEFAULT_GCC_PATCHES_DIR or '(empty)'}"
        ),
    )
    parser.add_argument(
        "--gcc-sync-from",
        default=DEFAULT_GCC_SYNC_FROM,
        help=(
            "On macOS, sync reflection commits from this source GCC repo into "
            "--gcc-source-dir before build. "
            f"Default: {DEFAULT_GCC_SYNC_FROM or '(disabled)'}"
        ),
    )
    parser.add_argument(
        "--gcc-sync-base-ref",
        default="master",
        help=(
            "Base ref in --gcc-sync-from used to compute the patch stack "
            "(merge-base against --gcc-sync-ref). Default: master."
        ),
    )
    parser.add_argument(
        "--gcc-sync-ref",
        default="HEAD",
        help="Ref in --gcc-sync-from that contains reflection commits. Default: HEAD.",
    )
    parser.add_argument(
        "--no-gcc-sync",
        action="store_true",
        help="Disable macOS GCC patch sync from --gcc-sync-from before build.",
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
        help=(
            "Additional compiler flag for this run (CMAKE_CXX_FLAGS). Repeat "
            "to pass multiple flags."
        ),
    )
    # Options from when this script ran the compiler itself. They're still
    # parsed so old command lines get a clear message rather than argparse's.
    parser.add_argument("--opt-level", help=argparse.SUPPRESS)
    parser.add_argument("--std", help=argparse.SUPPRESS)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the full build and test commands and output.",
    )
    args = parser.parse_args(argv)

    if args.opt_level is not None:
        parser.error(
            "--opt-level is gone: the build type sets the optimization level "
            "(--build-type release is -O3, debug is -O0)."
        )
    if args.std is not None:
        parser.error("--std is gone: CMake sets the language standard (c++2c).")
    if args.build_type is None:
        args.build_type = "debug" if args.debugger else "release"
    if args.debugger and (args.compiler == "both" or args.build_type == "all"):
        parser.error("--debugger needs one --compiler and one --build-type.")
    if args.debugger and (not args.tests or len(args.tests) != 1):
        parser.error("--debugger needs exactly one test, given with --tests.")
    return args, cmake_args


def selected_compilers(args: argparse.Namespace) -> list[str]:
    if args.compiler == "both":
        return ["clang", "gcc"]
    return [args.compiler]


def selected_build_types(args: argparse.Namespace) -> list[str]:
    if args.build_type == "all":
        return list(BUILD_TYPES)
    return [args.build_type]


def build_specs(args: argparse.Namespace) -> dict[str, CompilerSpec]:
    clang_root = Path(args.clang_root).resolve()
    clang_source_dir = Path(args.clang_source_dir).resolve()
    gcc_source_dir = Path(args.gcc_source_dir).resolve()
    gcc_build_dir = Path(args.gcc_build_dir).resolve()
    gcc_binary_dir = gcc_build_dir / "artifacts"
    gcc_executable = (
        Path(args.gcc_executable).absolute()
        if args.gcc_executable
        else gcc_binary_dir / "bin" / "g++"
    )
    return {
        "clang": CompilerSpec(
            name="clang",
            source_dir=clang_source_dir,
            # The clang root doubles as its build tree (runtimes + <meta> land here).
            build_dir=clang_root,
            binary_dir=clang_root,
            # absolute(), not resolve(): clang++ is a symlink to clang-<N>, and
            # the name it's run by picks the driver mode (C++ or C).
            executable=(
                Path(args.clang_executable).absolute()
                if args.clang_executable
                else clang_root / "bin" / "clang++"
            ),
        ),
        "gcc": CompilerSpec(
            name="gcc",
            source_dir=gcc_source_dir,
            build_dir=gcc_build_dir,
            binary_dir=gcc_binary_dir,
            executable=gcc_executable,
        ),
    }


def maybe_sync_gcc_sources_for_macos(
    spec: CompilerSpec, args: argparse.Namespace
) -> None:
    """On macOS, apply new reflection commits from gcc-mirror to darwin GCC.

    This keeps the Linux/default source of truth (gcc-mirror) and the macOS
    darwin build tree aligned by replaying only new commits since the previous
    sync state. Non-macOS runs are unaffected.
    """
    if sys.platform != "darwin" or args.no_gcc_sync:
        return

    if not args.gcc_sync_from:
        return

    sync_from = Path(args.gcc_sync_from).resolve()
    sync_to = spec.source_dir.resolve()

    if sync_from == sync_to:
        return

    if not (sync_from / ".git").exists():
        raise SystemExit(f"gcc sync source is not a git repo: {sync_from}")
    if not (sync_to / ".git").exists():
        raise SystemExit(
            "gcc sync target is not a git repo: "
            f"{sync_to}\n"
            "Clone the darwin fork first, or point --gcc-source-dir at a git checkout."
        )

    # Guard against retry loops: if a previous `git am` stopped on conflicts,
    # attempting a new `git am` will fail immediately with rebase-apply present.
    am_state_path = require_git_output(
        sync_to, ["git", "rev-parse", "--git-path", "rebase-apply"], args.verbose
    )
    am_state_dir = Path(am_state_path)
    if not am_state_dir.is_absolute():
        am_state_dir = sync_to / am_state_dir
    if am_state_dir.exists():
        raise SystemExit(
            "gcc sync cannot start: an earlier 'git am' session is still in progress in "
            f"{sync_to}.\n"
            "Finish it with 'git am --continue' after resolving conflicts, or run "
            "'git am --abort' to discard it.\n"
            "You can inspect the current patch with 'git am --show-current-patch=diff'.\n"
            "To bypass sync for this build, pass --no-gcc-sync."
        )

    target_status = run_command(
        ["git", "status", "--porcelain"], cwd=sync_to, verbose=args.verbose
    )
    require_success(target_status, f"git status failed in {sync_to}")
    if target_status.stdout.strip():
        raise SystemExit(
            "gcc sync target has local changes; refusing to apply patch stack on a dirty "
            f"tree: {sync_to}.\n"
            "Commit/stash/clean the target tree, then retry, or pass --no-gcc-sync."
        )

    ensure_directory(
        Path(args.gcc_patches_dir)
        if args.gcc_patches_dir
        else BUILD_ROOT / "gcc-sync-patches"
    )
    patches_dir = (
        Path(args.gcc_patches_dir).resolve()
        if args.gcc_patches_dir
        else (BUILD_ROOT / "gcc-sync-patches").resolve()
    )
    state_file = spec.build_dir / ".gcc_sync_state.json"

    source_head = require_git_output(
        sync_from, ["git", "rev-parse", args.gcc_sync_ref], args.verbose
    )
    base_commit = require_git_output(
        sync_from,
        ["git", "merge-base", args.gcc_sync_ref, args.gcc_sync_base_ref],
        args.verbose,
    )

    start_commit = base_commit
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            previous = state.get("last_synced_source_commit", "")
            if previous:
                ancestor_check = run_command(
                    ["git", "merge-base", "--is-ancestor", previous, source_head],
                    cwd=sync_from,
                    verbose=args.verbose,
                )
                if ancestor_check.returncode == 0:
                    start_commit = previous
        except (OSError, json.JSONDecodeError):
            pass

    pending = require_git_output(
        sync_from,
        ["git", "rev-list", "--count", f"{start_commit}..{source_head}"],
        args.verbose,
    )
    if pending == "0":
        log("[gcc-sync] no new gcc-mirror commits to apply for macOS")
        return

    log(f"[gcc-sync] exporting {pending} commit(s) from {sync_from}...")
    shutil.rmtree(patches_dir, ignore_errors=True)
    ensure_directory(patches_dir)
    export_result = run_command_streamed(
        [
            "git",
            "format-patch",
            "-o",
            str(patches_dir),
            f"{start_commit}..{source_head}",
        ],
        cwd=sync_from,
        verbose=args.verbose,
    )
    require_success(export_result, "gcc sync: format-patch")

    patch_files = sorted(patches_dir.glob("*.patch"))
    if not patch_files:
        raise SystemExit(
            "gcc sync expected patches but none were exported. "
            f"range: {start_commit}..{source_head}"
        )

    log(f"[gcc-sync] applying {len(patch_files)} patch(es) to {sync_to}...")
    apply_result = run_command_streamed(
        ["git", "am", "--3way", *[str(path) for path in patch_files]],
        cwd=sync_to,
        verbose=args.verbose,
    )
    if apply_result.returncode != 0:
        auto_resolved = try_auto_resolve_git_am_conflicts(sync_to, args.verbose)
        if auto_resolved:
            log("[gcc-sync] auto-resolved patch conflicts and completed git am")
        else:
            abort_result = run_command(
                ["git", "am", "--abort"], cwd=sync_to, verbose=args.verbose
            )
            if abort_result.returncode != 0:
                abort_details = render_command_failure(abort_result)
                raise SystemExit(
                    "gcc sync failed while applying patches, and automatic rollback also "
                    f"failed in {sync_to}.\n\n{abort_details}"
                )
            raise SystemExit(
                "gcc sync failed while applying patches, and the in-progress 'git am' "
                "session was rolled back automatically to keep the tree clean.\n"
                f"Target: {sync_to}\n"
                f"Exported patch stack: {patches_dir}\n"
                "Fix the source/target divergence, then retry. Use --no-gcc-sync to skip "
                "this step for a build."
            )

    ensure_directory(state_file.parent)
    state_file.write_text(
        json.dumps(
            {
                "last_synced_source_commit": source_head,
                "sync_from": str(sync_from),
                "sync_to": str(sync_to),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def require_git_output(repo: Path, command: list[str], verbose: bool) -> str:
    """Run a git command and return stripped stdout or raise with details."""
    result = run_command(command, cwd=repo, verbose=verbose)
    require_success(result, f"git command failed in {repo}")
    value = result.stdout.strip()
    if not value:
        raise SystemExit(
            f"git command returned empty output in {repo}: {shlex.join(command)}"
        )
    return value


def try_auto_resolve_git_am_conflicts(repo: Path, verbose: bool) -> bool:
    """Attempt to finish an in-progress git am by auto-resolving merge conflicts.

    Strategy: for each conflict stop, take the incoming patch side ("theirs")
    for unmerged paths, stage, and continue. If the sequence completes, return
    True; otherwise return False so caller can roll back.
    """
    max_rounds = 200
    for _ in range(max_rounds):
        am_state = run_command(
            ["git", "rev-parse", "--git-path", "rebase-apply"],
            cwd=repo,
            verbose=verbose,
        )
        if am_state.returncode != 0:
            return False
        am_state_path = am_state.stdout.strip()
        if not am_state_path:
            return False
        state_dir = Path(am_state_path)
        if not state_dir.is_absolute():
            state_dir = repo / state_dir
        if not state_dir.exists():
            return True

        unmerged = run_command(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=repo,
            verbose=verbose,
        )
        if unmerged.returncode != 0:
            return False
        files = [line.strip() for line in unmerged.stdout.splitlines() if line.strip()]
        if not files:
            return False

        checkout_theirs = run_command(
            ["git", "checkout", "--theirs", "--", *files],
            cwd=repo,
            verbose=verbose,
        )
        if checkout_theirs.returncode != 0:
            return False

        add_result = run_command(
            ["git", "add", "--", *files], cwd=repo, verbose=verbose
        )
        if add_result.returncode != 0:
            return False

        cont = run_command(["git", "am", "--continue"], cwd=repo, verbose=verbose)
        if cont.returncode == 0:
            continue

    return False


def build_selected_compilers(
    specs: dict[str, CompilerSpec], args: argparse.Namespace
) -> None:
    for compiler_name in selected_compilers(args):
        spec = specs[compiler_name]
        if compiler_name == "clang":
            clang_recipe.build_clang(spec, args)
        else:
            maybe_sync_gcc_sources_for_macos(spec, args)
            gcc_recipe.build_gcc(spec, args)


def validate_compiler_executable(spec: CompilerSpec) -> None:
    if spec.executable.is_file():
        return
    raise SystemExit(
        f"Compiler executable not found for {spec.name}: {spec.executable}\n"
        f"Build it from the submodule with --build-compilers --compiler {spec.name}, "
        "or point --clang-root at an existing build."
    )


def looks_like_path(pattern: str) -> bool:
    return (
        "/" in pattern or pattern.endswith(".cpp") or any(c in pattern for c in "*?[")
    )


def test_names_for(test_file: Path) -> tuple[str, ...]:
    """The ctest names CMake may give a test file (see tests/CMakeLists.txt).

    Each compiler has at most one of them: a known_gap/ file is a
    compile_check test on its own compiler and, with a TEST-PASSES-ON line,
    an ordinary test on the other.
    """
    relative = test_file.relative_to(TESTS_DIR)
    plain = test_file.name.split(".")[0]
    if relative.parts[0] in COMPILE_CHECK_DIRS:
        checked = "reflection_ad." + relative.with_suffix("").as_posix().replace(
            "/", "."
        )
        return (checked, plain) if relative.parts[0] == "known_gap" else (checked,)
    return (plain,)


def requested_names(
    patterns: list[str], base_dir: Path, names_for
) -> list[tuple[str, ...]]:
    """Turn --tests/--benchmarks values into ctest or target names.

    A value with a '/', a '.cpp' suffix or a glob character is a glob relative
    to base_dir (a leading 'tests/' or 'benchmarks/' is dropped); anything else
    is already a name. Each entry holds the names one request may go by.
    """
    names: list[tuple[str, ...]] = []
    for pattern in patterns:
        if not looks_like_path(pattern):
            names.append((pattern,))
            continue
        pattern = pattern.removeprefix(f"{base_dir.name}/")
        matches = sorted(
            path for path in base_dir.glob(pattern) if path.suffix == ".cpp"
        )
        if not matches:
            raise SystemExit(f"No .cpp file under {base_dir.name}/ matches {pattern!r}")
        names.extend(names_for(path) for path in matches)
    return list(dict.fromkeys(names))


def select(wanted: list[tuple[str, ...]], available) -> tuple[list[str], list[str]]:
    """The available names among `wanted`, and the requests with none."""
    found: list[str] = []
    missing: list[str] = []
    for names in wanted:
        present = [name for name in names if name in available]
        found += present
        if not present:
            missing.append(names[0])
    return found, missing


def preset_build_dir(preset: str, args: argparse.Namespace) -> Path:
    root = Path(args.build_dir).resolve() if args.build_dir else CMAKE_BUILD_ROOT
    return root / preset


def clean_build_dir(build_dir: Path) -> None:
    """Delete a build tree, refusing anything that isn't one of ours."""
    if not build_dir.exists():
        return
    protected = [
        ROOT,
        BUILD_ROOT,
        CMAKE_BUILD_ROOT,
        Path(DEFAULT_CLANG_ROOT).resolve(),
        Path(DEFAULT_GCC_BUILD_DIR).resolve(),
        Path(DEFAULT_CLANG_RUNTIMES_DIR).resolve(),
    ]
    if (
        any(build_dir == p or build_dir in p.parents for p in protected)
        or not (build_dir / "CMakeCache.txt").is_file()
    ):
        raise SystemExit(
            f"--clean: refusing to delete {build_dir} (not a CMake build tree)"
        )
    log(f"removing {build_dir}")
    shutil.rmtree(build_dir)


def cached_compiler(build_dir: Path) -> str | None:
    try:
        cache = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^CMAKE_CXX_COMPILER:[A-Z]+=(.*)$", cache, re.MULTILINE)
    return match.group(1) if match else None


def configure(
    spec: CompilerSpec,
    preset: str,
    build_dir: Path,
    args: argparse.Namespace,
    cmake_args: list[str],
) -> CommandResult:
    command = [
        "cmake",
        "--preset",
        preset,
        "-B",
        str(build_dir),
        f"-DCMAKE_CXX_COMPILER={spec.executable}",
        # Always set, so --extra-cxxflag lasts one run rather than staying in
        # the tree's cache. A -DCMAKE_CXX_FLAGS after `--` still wins.
        f"-DCMAKE_CXX_FLAGS={shlex.join(args.extra_cxxflag)}",
    ]
    previous = cached_compiler(build_dir)
    if previous is not None and Path(previous) != spec.executable:
        # CMake can't change the compiler of an existing tree.
        log(f"[{preset}] compiler changed (was {previous}); reconfiguring from scratch")
        command.append("--fresh")
    command.extend(cmake_args)
    return run_command_streamed(command, cwd=ROOT, verbose=args.verbose)


def build_targets(build_dir: Path) -> set[str]:
    """Every target the build tree can build (Ninja's `help` target)."""
    result = run_command(
        ["cmake", "--build", str(build_dir), "--target", "help"],
        cwd=ROOT,
        verbose=False,
    )
    require_success(result, "listing targets")
    return set(re.findall(r"^([^\s:/][^\s:]*): ", result.stdout, re.MULTILINE))


def ctest_tests(build_dir: Path) -> dict[str, dict]:
    """ctest's own list of the tests in a build tree, by name."""
    result = run_command(
        ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"],
        cwd=ROOT,
        verbose=False,
    )
    require_success(result, "listing tests")
    return {test["name"]: test for test in json.loads(result.stdout)["tests"]}


def resolve_selection(
    build_dir: Path, args: argparse.Namespace
) -> tuple[Selection, list[str]]:
    """Match --tests/--benchmarks against what this build tree has.

    Returns the selection and the requested names this compiler doesn't have
    (e.g. a clang_only test under GCC), which the caller reports.
    """
    missing: list[str] = []
    tests: list[str] | None = None
    # --benchmarks on its own skips the tests; with --tests too, both run.
    with_tests = args.tests is not None or args.benchmarks is None
    if with_tests and args.tests:
        wanted = requested_names(args.tests, TESTS_DIR, test_names_for)
        tests, missing = select(wanted, ctest_tests(build_dir))
    elif not with_tests:
        tests = []

    benchmarks: list[str] = []
    if args.benchmarks is not None:
        if args.benchmarks:
            wanted = requested_names(
                args.benchmarks, BENCHMARKS_DIR, lambda path: (path.name.split(".")[0],)
            )
        else:
            wanted = [
                (path.name.split(".")[0],)
                for path in sorted(BENCHMARKS_DIR.rglob("*.cpp"))
            ]
        benchmarks, absent = select(wanted, build_targets(build_dir))
        # With no names, benchmarks this compiler can't build aren't missing.
        if args.benchmarks:
            missing += absent
    return Selection(tests=tests, benchmarks=benchmarks), missing


def build(
    build_dir: Path, selection: Selection, args: argparse.Namespace
) -> CommandResult:
    command = ["cmake", "--build", str(build_dir), "--parallel", str(args.jobs)]
    if args.verbose:
        command.append("--verbose")
    if selection.tests is not None:
        # Only what was asked for. compile_check tests (static_fail/,
        # known_gap/) aren't targets: their test compiles them.
        targets = build_targets(build_dir)
        wanted = [n for n in [*selection.tests, *selection.benchmarks] if n in targets]
        if not wanted:
            return CommandResult(command, 0, "", "")
        command += ["--target", *wanted]
    return run_command_streamed(command, cwd=ROOT, verbose=args.verbose)


def run_ctest(
    build_dir: Path, tests: list[str] | None, args: argparse.Namespace
) -> CommandResult:
    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        # Without this, a -R that matches nothing passes.
        "--no-tests=error",
        "--parallel",
        str(args.jobs),
    ]
    if args.verbose:
        command.append("--verbose")
    if tests is not None:
        command += ["-R", "^(" + "|".join(re.escape(name) for name in tests) + ")$"]
    return run_command_streamed(command, cwd=ROOT, verbose=args.verbose)


def run_benchmark(
    build_dir: Path, name: str, args: argparse.Namespace
) -> CommandResult:
    # Run from the build tree: some benchmarks write CSV files to the
    # working directory.
    working_dir = build_dir / "benchmarks"
    log(f"running benchmark {name} ...")
    return run_command_streamed(
        [str(working_dir / name)], cwd=working_dir, verbose=args.verbose
    )


def start_debugger(build_dir: Path, test: str, args: argparse.Namespace) -> int:
    """Replace this process with the debugger, attached to one test binary."""
    info = ctest_tests(build_dir).get(test, {})
    command = info.get("command") or []
    if not command or Path(command[0]).name != test:
        raise SystemExit(
            f"{test} isn't a test binary that can be debugged (static_fail/ and "
            "known_gap/ tests only compile)."
        )
    properties = {p["name"]: p["value"] for p in info.get("properties", [])}
    working_dir = properties.get("WORKING_DIRECTORY", str(build_dir))
    if args.debugger == "gdb":
        debugger = ["gdb", "--args", *command]
    else:
        debugger = ["lldb", "--", *command]
    log(f"[{working_dir}] $ {shlex.join(debugger)}")
    os.chdir(working_dir)
    try:
        os.execvp(debugger[0], debugger)
    except OSError as error:
        raise SystemExit(f"Can't start {debugger[0]}: {error}") from None
    return 1  # not reached


def run_preset(
    spec: CompilerSpec,
    build_type: str,
    args: argparse.Namespace,
    cmake_args: list[str],
) -> RunResult:
    preset = f"{spec.name}-{build_type}"
    build_dir = preset_build_dir(preset, args)
    result = RunResult(spec.name, build_type, build_dir)
    if args.clean:
        clean_build_dir(build_dir)

    log(f"[{preset}] configuring {build_dir} with {spec.executable}")
    if configure(spec, preset, build_dir, args, cmake_args).returncode != 0:
        result.status = "configure failed"
        return result

    selection, missing = resolve_selection(build_dir, args)
    result.missing = missing
    for name in missing:
        log(
            f"[{preset}] no test or benchmark named {name} with {spec.name}; skipping it"
        )
    if (selection.tests == [] and not selection.benchmarks) and missing:
        result.status = "skipped"
        result.detail = "nothing selected for this compiler"
        return result

    log(f"[{preset}] building")
    if build(build_dir, selection, args).returncode != 0:
        result.status = "build failed"
        return result

    if args.debugger:
        start_debugger(build_dir, (selection.tests or [""])[0], args)

    if args.build_only:
        return result
    if selection.tests != []:
        log(f"[{preset}] testing")
        if run_ctest(build_dir, selection.tests, args).returncode != 0:
            result.status = "tests failed"
    if args.run_executables:
        failed = [
            name
            for name in selection.benchmarks
            if run_benchmark(build_dir, name, args).returncode != 0
        ]
        if failed:
            result.status = "benchmarks failed"
            result.detail = ", ".join(failed)
    return result


def summarize(results: list[RunResult]) -> int:
    print()
    print("Summary")
    print("-------")
    for result in results:
        line = f"{result.compiler}-{result.build_type}: {result.status}"
        if result.detail:
            line += f" ({result.detail})"
        print(line)
    failures = [r for r in results if r.status not in ("passed", "skipped")]
    # A name no compiler has is a typo, not a test for another compiler.
    configured = [r for r in results if r.status != "configure failed"]
    unknown = (
        set.intersection(*(set(r.missing) for r in configured)) if configured else set()
    )
    for name in sorted(unknown):
        print(f"No test or benchmark named {name} for any selected compiler.")
    return len(failures) + len(unknown)


def main() -> int:
    args, cmake_args = parse_args(sys.argv[1:])
    specs = build_specs(args)

    if args.build_compilers:
        log(f"building compiler(s): {', '.join(selected_compilers(args))}")
        build_selected_compilers(specs, args)

    results: list[RunResult] = []
    for compiler_name in selected_compilers(args):
        spec = specs[compiler_name]
        validate_compiler_executable(spec)
        for build_type in selected_build_types(args):
            results.append(run_preset(spec, build_type, args, cmake_args))

    return 1 if summarize(results) else 0


if __name__ == "__main__":
    sys.exit(main())
