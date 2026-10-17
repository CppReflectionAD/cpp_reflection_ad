"""Helpers shared by run_tests.py and the compiler build recipes.

The CI toolchain cache keys hash this file along with each recipe, so a change
here rebuilds both compilers.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CompilerSpec:
    name: str
    source_dir: Path
    build_dir: Path
    binary_dir: Path
    executable: Path


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


def require_success(result: CommandResult, label: str) -> None:
    if result.returncode == 0:
        return
    details = render_command_failure(result)
    raise SystemExit(f"{label} failed\n\n{details}")


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


def ensure_submodule(source_dir: Path, verbose: bool) -> None:
    """Check out a submodule on demand so the repo builds from a bare clone.

    A fresh clone leaves clang-p2996/gcc-mirror as empty submodule dirs; the
    build needs their sources. Idempotent: a no-op once populated.
    """
    if source_dir.is_dir() and (
        (source_dir / ".git").exists() or any(source_dir.iterdir())
    ):
        return

    # Compare full paths, not names: a --clang-source-dir/--gcc-source-dir
    # checkout elsewhere may share a submodule's name but is not that submodule.
    if source_dir not in {
        ROOT / "clang-p2996",
        ROOT / "gcc-mirror",
        ROOT / "gcc-darwin-reflect",
    }:
        raise SystemExit(
            "Compiler source directory is missing or empty: "
            f"{source_dir}\n"
            "This path is not a known submodule, so it is not auto-initialized. "
            "Clone/populate it first, or pass --clang-source-dir/--gcc-source-dir "
            "to an existing tree."
        )

    ensure_directory(source_dir)
    log(f"[{source_dir.name}] checking out submodule (large clone, be patient)...")
    result = run_command_streamed(
        ["git", "submodule", "update", "--init", "--progress", str(source_dir.name)],
        cwd=ROOT,
        verbose=verbose,
    )
    require_success(result, f"submodule checkout ({source_dir.name})")


def strip_binaries(prefix: Path, verbose: bool) -> None:
    """Strip an installed toolchain for the CI cache (Linux, GNU strip).

    Executables and shared libraries lose everything not needed to run or link
    against them; static archives lose only their debug info.
    """
    for path in sorted(prefix.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as file:
            magic = file.read(8)
        if magic.startswith(b"\x7fELF") and (
            ".so" in path.name or path.stat().st_mode & 0o111
        ):
            option = "--strip-unneeded"
        elif magic == b"!<arch>\n":
            option = "--strip-debug"
        else:
            continue
        require_success(
            run_command(["strip", option, str(path)], cwd=None, verbose=verbose),
            f"strip {path}",
        )
