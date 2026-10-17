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
    ensure_submodule,
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
        "--sanitize",
        action="store_true",
        help=(
            "Build and test with AddressSanitizer and UBSan, i.e. the "
            "<compiler>-sanitize preset (a Debug build). Clang needs a "
            "toolchain with compiler-rt (--build-compilers builds one)."
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
        "--gcc-sync-ref",
        default="HEAD",
        help="Ref in --gcc-sync-from that contains reflection commits. Default: HEAD.",
    )
    parser.add_argument(
        "--gcc-sync-pin",
        help=(
            "On macOS, the darwin commit the reflection commits are replayed onto. "
            "Default: the gcc-darwin-reflect submodule pin. Required when "
            "--gcc-source-dir is a standalone clone rather than the submodule."
        ),
    )
    # Superseded by the darwin pin; still accepted so old command lines work.
    parser.add_argument("--gcc-sync-base-ref", help=argparse.SUPPRESS)
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
    if args.gcc_sync_base_ref is not None:
        print(
            "warning: --gcc-sync-base-ref is ignored and will be removed; the "
            "macOS GCC sync now starts from the gcc-darwin-reflect pin "
            "(see --gcc-sync-pin).",
            file=sys.stderr,
        )
    if args.sanitize and args.build_type not in (None, "debug"):
        parser.error("--sanitize is a Debug build; drop --build-type.")
    if args.build_type is None:
        args.build_type = "debug" if args.debugger or args.sanitize else "release"
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
    """The preset suffixes to run: build types, or the sanitizer preset."""
    if args.sanitize:
        return ["sanitize"]
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


def gcc_sync_missing_checkout_message(path: Path, role: str, override: str) -> str:
    """Explain a missing gcc sync checkout and how to fix it."""
    lines = [f"gcc sync (macOS): {path} is not a git checkout.", f"It is {role}."]
    if path.parent == ROOT and path.name in {"gcc-mirror", "gcc-darwin-reflect"}:
        lines += [
            "This is a submodule that has not been checked out yet "
            "(normal after a plain 'git clone'). Fix with:",
            f"  git submodule update --init {path.name}",
            "Or check out both GCC trees at once:",
            "  git submodule update --init gcc-mirror gcc-darwin-reflect",
        ]
    else:
        lines.append(f"Point {override} at an existing git checkout.")
    lines.append(
        "To build without applying the reflection commits, pass --no-gcc-sync."
    )
    return "\n".join(lines)


def maybe_sync_gcc_sources_for_macos(
    spec: CompilerSpec, args: argparse.Namespace
) -> None:
    """On macOS, apply new reflection commits from gcc-mirror to darwin GCC.

    The patch stack is every non-merge commit in gcc-mirror that is not
    reachable from the darwin pin (see gcc_sync_target_pin). If the darwin tree
    is exactly where the previous successful sync left it, only newer commits
    are applied; otherwise the tree is reset to its pin and the whole stack is
    replayed, and put back where it was if that fails. The sync state lives in
    the darwin repo's git dir, so all build directories share it.
    Non-macOS runs are unaffected.
    """
    if sys.platform != "darwin" or args.no_gcc_sync:
        return

    if not args.gcc_sync_from:
        return

    sync_from = Path(args.gcc_sync_from).resolve()
    sync_to = spec.source_dir.resolve()

    if sync_from == sync_to:
        return

    # A fresh clone leaves both GCC submodules empty; check them out here.
    for path in (sync_from, sync_to):
        if path in {ROOT / "gcc-mirror", ROOT / "gcc-darwin-reflect"}:
            ensure_submodule(path, args.verbose)

    if not (sync_from / ".git").exists():
        raise SystemExit(
            gcc_sync_missing_checkout_message(
                sync_from,
                role="the source of the reflection commits",
                override="--gcc-sync-from (or REFLECT_GCC_SYNC_FROM)",
            )
        )
    if not (sync_to / ".git").exists():
        raise SystemExit(
            gcc_sync_missing_checkout_message(
                sync_to,
                role="the darwin GCC tree that gets built",
                override="--gcc-source-dir (or REFLECT_GCC_SOURCE_DIR)",
            )
        )

    # Guard against retry loops: if a previous `git am` stopped on conflicts,
    # attempting a new `git am` will fail immediately with rebase-apply present.
    if git_dir_path(sync_to, "rebase-apply", args.verbose).exists():
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
    # Shared by every build directory: it describes the darwin source tree.
    state_file = git_dir_path(sync_to, "gcc-sync-state.json", args.verbose)

    source_head = require_git_output(
        sync_from, ["git", "rev-parse", args.gcc_sync_ref], args.verbose
    )
    target_pin = gcc_sync_target_pin(sync_to, args)
    target_head = require_git_output(
        sync_to, ["git", "rev-parse", "HEAD"], args.verbose
    )

    # Fall back to the per-build-dir file written before the state moved into
    # the darwin repo, so existing checkouts don't pay for a full replay.
    state = read_gcc_sync_state(state_file) or read_gcc_sync_state(
        spec.build_dir / ".gcc_sync_state.json"
    )
    previous = state.get("last_synced_source_commit", "")
    tree_unchanged = (
        bool(previous)
        and state.get("target_pin") == target_pin
        and state.get("target_head") == target_head
    )
    if tree_unchanged and previous == source_head:
        log("[gcc-sync] darwin GCC tree already up to date with gcc-mirror")
        return

    fetch_gcc_sync_source(sync_from, sync_to, source_head, args)

    start_commit = ""
    if tree_unchanged:
        ancestor_check = run_command(
            ["git", "merge-base", "--is-ancestor", previous, source_head],
            cwd=sync_to,
            verbose=args.verbose,
        )
        if ancestor_check.returncode == 0:
            start_commit = previous
    full_replay = not start_commit
    if full_replay:
        start_commit = target_pin

    pending = require_git_output(
        sync_to,
        [
            "git",
            "rev-list",
            "--count",
            "--no-merges",
            f"{start_commit}..{source_head}",
        ],
        args.verbose,
    )
    # Checked before touching the tree, so a refusal leaves it as it was.
    if pending != "0":
        check_gcc_sync_merges(sync_to, start_commit, source_head, args)

    if full_replay and target_head != target_pin:
        log(
            f"[gcc-sync] resetting {sync_to} to its pin {target_pin[:12]} "
            f"(was {target_head[:12]}) and replaying the full patch stack"
        )
        reset_result = run_command(
            ["git", "checkout", "--detach", target_pin],
            cwd=sync_to,
            verbose=args.verbose,
        )
        require_success(reset_result, f"gcc sync: checkout pin in {sync_to}")

    if pending != "0":
        try:
            apply_gcc_sync_patches(
                sync_to, patches_dir, start_commit, source_head, pending, args
            )
        except SystemExit as exc:
            # A failed incremental sync is already back at target_head (git am
            # --abort); a failed full replay is at the bare pin, so put back the
            # previous synced head rather than leave a tree without reflection.
            if not full_replay or target_head == target_pin:
                raise
            restore = run_command(
                ["git", "checkout", "--detach", target_head],
                cwd=sync_to,
                verbose=args.verbose,
            )
            if restore.returncode == 0:
                note = (
                    f"The darwin tree was put back at {target_head[:12]}, "
                    "where it was before this run."
                )
            else:
                note = (
                    f"Putting the darwin tree back at {target_head} also failed. "
                    f"Run 'git checkout --detach {target_head}' in {sync_to}.\n\n"
                    + render_command_failure(restore)
                )
            raise SystemExit(f"{exc.code}\n\n{note}") from None

    state_file.write_text(
        json.dumps(
            {
                "last_synced_source_commit": source_head,
                "target_pin": target_pin,
                "target_head": require_git_output(
                    sync_to, ["git", "rev-parse", "HEAD"], args.verbose
                ),
                "sync_from": str(sync_from),
                "sync_to": str(sync_to),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def git_dir_path(repo: Path, name: str, verbose: bool) -> Path:
    """Resolve `git rev-parse --git-path <name>` for repo to an absolute path."""
    path = Path(
        require_git_output(repo, ["git", "rev-parse", "--git-path", name], verbose)
    )
    return path if path.is_absolute() else repo / path


def read_gcc_sync_state(path: Path) -> dict[str, str]:
    """Read a sync state file; a missing or malformed file counts as no state."""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def gcc_sync_target_pin(sync_to: Path, args: argparse.Namespace) -> str:
    """The darwin commit the reflection commits are replayed onto.

    --gcc-sync-pin wins. Otherwise it is the gitlink in the superproject's
    index (not HEAD), so a pin bumped with `git update-index --cacheinfo`
    takes effect before it is committed.
    """
    if args.gcc_sync_pin:
        return require_git_output(
            sync_to,
            ["git", "rev-parse", "--verify", f"{args.gcc_sync_pin}^{{commit}}"],
            args.verbose,
        )
    superproject = run_command(
        ["git", "rev-parse", "--show-superproject-working-tree"],
        cwd=sync_to,
        verbose=args.verbose,
    )
    if superproject.returncode != 0 or not superproject.stdout.strip():
        raise SystemExit(
            f"gcc sync target is a standalone clone, not a submodule: {sync_to}\n"
            "The sync replays the reflection commits onto a fixed darwin commit, "
            "which for the submodule is its pin. For a standalone clone, give that "
            "commit with --gcc-sync-pin <sha> (normally the pin recorded in this "
            "repo: 'git rev-parse :gcc-darwin-reflect'), or pass --no-gcc-sync to "
            "skip the sync."
        )
    superproject_root = Path(superproject.stdout.strip()).resolve()
    target_rel = sync_to.relative_to(superproject_root).as_posix()
    return require_git_output(
        superproject_root, ["git", "rev-parse", f":{target_rel}"], args.verbose
    )


GCC_SYNC_SOURCE_REF = "refs/gcc-sync/source"


def fetch_gcc_sync_source(
    sync_from: Path, sync_to: Path, source_head: str, args: argparse.Namespace
) -> None:
    """Fetch source_head into sync_to under GCC_SYNC_SOURCE_REF.

    The ref keeps the gcc-mirror commits reachable, so gc can't prune them.
    Protocol v2 lets the fetch name any reachable commit, not only branch tips.
    """
    fetch_result = run_command(
        [
            "git",
            "-c",
            "protocol.version=2",
            "fetch",
            "--no-tags",
            str(sync_from),
            f"+{source_head}:{GCC_SYNC_SOURCE_REF}",
        ],
        cwd=sync_to,
        verbose=args.verbose,
    )
    require_success(fetch_result, f"gcc sync: fetch from {sync_from}")


def check_gcc_sync_merges(
    sync_to: Path, start_commit: str, source_head: str, args: argparse.Namespace
) -> None:
    """Refuse to sync if a merge in the range has changes of its own.

    format-patch skips merge commits, so a manual conflict resolution (or any
    other edit made in a merge) would be silently lost. `git show
    --remerge-diff` is empty for a merge that git would redo identically.
    """
    merges = run_command(
        ["git", "rev-list", "--merges", f"{start_commit}..{source_head}"],
        cwd=sync_to,
        verbose=args.verbose,
    )
    require_success(merges, "gcc sync: list merges")
    resolved = []
    for merge in merges.stdout.split():
        remerge = run_command(
            ["git", "show", "--remerge-diff", "--format=", merge],
            cwd=sync_to,
            verbose=args.verbose,
        )
        require_success(remerge, f"gcc sync: remerge-diff of {merge[:12]}")
        if remerge.stdout.strip():
            resolved.append(merge)
    if resolved:
        raise SystemExit(
            "gcc sync cannot replay gcc-mirror: these merge commits contain "
            "manual conflict resolutions, which a patch replay would drop:\n"
            + "\n".join(f"  {merge}" for merge in resolved)
            + f"\nInspect them with 'git show --remerge-diff <sha>' in {sync_to}. "
            "The darwin tree was not changed. Pass --no-gcc-sync to skip the sync."
        )


def apply_gcc_sync_patches(
    sync_to: Path,
    patches_dir: Path,
    start_commit: str,
    source_head: str,
    pending: str,
    args: argparse.Namespace,
) -> None:
    """Export start_commit..source_head from sync_to's object store and git-am it.

    Conflicts are not auto-resolved: taking either side of a file wholesale
    loses the other side's edits. On failure the session is aborted, which
    returns HEAD to where `git am` started.
    """
    log(f"[gcc-sync] exporting {pending} commit(s) from {sync_to}...")
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
        cwd=sync_to,
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
    # Author dates as committer dates: replaying the same stack on the same pin
    # then yields the same commits, so a repeat replay changes no SHAs.
    apply_result = run_command_streamed(
        [
            "git",
            "am",
            "--3way",
            "--committer-date-is-author-date",
            *[str(path) for path in patch_files],
        ],
        cwd=sync_to,
        verbose=args.verbose,
    )
    if apply_result.returncode == 0:
        return

    unmerged = run_command(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=sync_to,
        verbose=args.verbose,
    )
    conflicted = unmerged.stdout.split() if unmerged.returncode == 0 else []
    abort_result = run_command(
        ["git", "am", "--abort"], cwd=sync_to, verbose=args.verbose
    )
    if abort_result.returncode != 0:
        abort_details = render_command_failure(abort_result)
        raise SystemExit(
            "gcc sync failed while applying patches, and automatic rollback also "
            f"failed in {sync_to}.\n\n{abort_details}"
        )
    conflict_lines = "".join(f"  {path}\n" for path in conflicted)
    raise SystemExit(
        "gcc sync failed: a gcc-mirror patch does not apply cleanly to the darwin "
        "tree (see the 'Patch failed at' line above), so the 'git am' session "
        "was rolled back.\n"
        f"Target: {sync_to}\n"
        + (f"Conflicting files:\n{conflict_lines}" if conflicted else "")
        + f"Exported patch stack: {patches_dir}\n"
        "Fix the source/target divergence, then retry. Use --no-gcc-sync to skip "
        "this step for a build."
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
