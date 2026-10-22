"""Name, pack and fetch the compiler toolchains CI builds, as release archives.

CI builds each compiler from its submodule pin into the Actions cache
(toolchain-cache-*.yml). toolchain-archive.yml packs each of those builds into
an asset of this repository's `toolchains` release, named after the inputs of
the cache key, and `run_tests.py --fetch-compilers` downloads the assets that
match this checkout and host:

    python3 -m tools.toolchain.archive name gcc [--cache-key KEY]
    python3 -m tools.toolchain.archive pack gcc /opt/toolchain-gcc OUT_DIR
    python3 -m tools.toolchain.archive fetch gcc build/gcc-p2996/artifacts

The cache keys don't hash this file, so editing it rebuilds nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .common import ROOT, log
from .gcc import SOURCE_SUBMODULE as GCC_SOURCE_SUBMODULE

# Where the assets are; REFLECT_TOOLCHAIN_URL overrides it (CI tests an
# archive from a file:// URL before publishing it).
DEFAULT_URL = os.environ.get(
    "REFLECT_TOOLCHAIN_URL",
    "https://github.com/CppReflectionAD/cpp_reflection_ad/releases/download/toolchains",
)
# Written into an unpacked toolchain, holding its asset name. A fetch replaces
# a fetched toolchain but never a local build, and run_tests.py won't build
# over a fetched one.
MARKER = ".toolchain-archive"

SUBMODULES = {"clang": "clang-p2996", "gcc": GCC_SOURCE_SUBMODULE}
# The files the cache key hashes, in the same order (toolchain-cache-*.yml).
RECIPE_FILES = {
    compiler: (
        f".github/scripts/build-{compiler}-toolchain.sh",
        "tools/toolchain/common.py",
        f"tools/toolchain/{compiler}.py",
    )
    for compiler in SUBMODULES
}


def git(*args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"git {' '.join(args)} failed: {error}") from error


def pin(compiler: str) -> str:
    """The submodule commit, from the superproject's tree, as the key reads it."""
    return git("rev-parse", f"HEAD:{SUBMODULES[compiler]}").decode().strip()


def recipe_hash(compiler: str) -> str:
    """The cache key's hash of the recipe, over the committed files.

    Reading HEAD rather than the working tree means an uncommitted recipe
    edit doesn't change which archive matches this checkout.
    """
    digest = hashlib.sha256()
    for path in RECIPE_FILES[compiler]:
        digest.update(git("show", f"HEAD:{path}"))
    return digest.hexdigest()


def platform_id() -> str:
    system = {"linux": "linux", "darwin": "macos"}.get(sys.platform)
    machine = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(platform.machine().lower())
    if system is None or machine is None or (system, machine) == ("macos", "x86_64"):
        raise SystemExit(
            f"CI builds no toolchain for {sys.platform} on {platform.machine()}, "
            "only for Linux x86-64, Linux arm64 and macOS arm64. Build the "
            "compilers with --build-compilers instead."
        )
    return f"{system}-{machine}"


def asset_name(compiler: str) -> str:
    """<compiler>-<pin>-<recipe hash>-<platform>.tar.xz, matching the cache key."""
    return (
        f"{compiler}-{pin(compiler)[:12]}-{recipe_hash(compiler)[:12]}-"
        f"{platform_id()}.tar.xz"
    )


def check_cache_key(compiler: str, cache_key: str) -> None:
    """Fail if the asset name and CI's cache key disagree on pin or recipe."""
    expected = f"-{pin(compiler)[:12]}-{recipe_hash(compiler)[:12]}"
    if expected not in cache_key:
        raise SystemExit(
            f"The cache key {cache_key} doesn't contain {expected}: archive.py "
            f"and toolchain-cache-{compiler}.yml compute the key differently."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(compiler: str, prefix: Path, out_dir: Path, macos_sdk: str) -> Path:
    """Pack an installed toolchain into out_dir, with its checksum.

    On macOS, GCC also gets a .sdk file naming the SDK and linker it was
    built with (macos-sdk-id.sh), which a fetch compares with the host's.
    """
    name = asset_name(compiler)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / name
    log(f"[{compiler}] packing {prefix} into {archive}")
    # tarfile keeps symlinks as symlinks and repeated inodes as hard links.
    with tarfile.open(archive, "w:xz", preset=6) as tar:
        for child in sorted(prefix.iterdir()):
            tar.add(child, arcname=child.name)
    (out_dir / f"{name}.sha256").write_text(f"{sha256_file(archive)}  {name}\n")
    if macos_sdk:
        (out_dir / f"{name}.sdk").write_text(f"{macos_sdk}\n")
    log(f"[{compiler}] {archive.stat().st_size >> 20} MiB")
    return archive


def open_url(url: str, name: str):
    try:
        return urllib.request.urlopen(url)
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(
            f"Can't download {url}: {error}\n"
            f"CI publishes {name} once it has built this pin and recipe on "
            "main. Until then, or on a branch that changes them, build the "
            "compilers with --build-compilers."
        ) from error


def fetch(compiler: str, dest: Path, url: str = DEFAULT_URL) -> None:
    """Download, check and unpack this checkout's toolchain into dest."""
    name = asset_name(compiler)
    marker = dest / MARKER
    if dest.exists() and not marker.is_file():
        raise SystemExit(
            f"{dest} exists but wasn't fetched (is it a local build?). Delete "
            "it, or fetch into another directory with --clang-root or "
            "--gcc-build-dir."
        )
    if marker.is_file() and marker.read_text().strip() == name:
        log(f"[{compiler}] {dest} already holds {name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{dest.name}-") as tmp:
        archive = Path(tmp) / name
        log(f"[{compiler}] downloading {url}/{name}")
        with open_url(f"{url}/{name}", name) as response, archive.open("wb") as file:
            shutil.copyfileobj(response, file, 1 << 20)
        with open_url(f"{url}/{name}.sha256", name) as response:
            expected = response.read().decode().split()[0]
        if sha256_file(archive) != expected:
            raise SystemExit(f"{name} doesn't match its checksum; try again.")

        unpacked = Path(tmp) / "toolchain"
        log(f"[{compiler}] unpacking into {dest}")
        with tarfile.open(archive) as tar:
            # The "data" filter rejects absolute paths and links leaving the
            # tree. Python 3.11 before 3.11.4 doesn't have it.
            if hasattr(tarfile, "data_filter"):
                tar.extractall(unpacked, filter="data")
            else:
                tar.extractall(unpacked)
        (unpacked / MARKER).write_text(f"{name}\n")
        if dest.exists():
            shutil.rmtree(dest)
        unpacked.rename(dest)

    if compiler == "gcc" and sys.platform == "darwin":
        warn_on_other_sdk(url, name)
    log(f"[{compiler}] fetched {name} into {dest}")


def warn_on_other_sdk(url: str, name: str) -> None:
    """Warn when this Mac's SDK or linker isn't the one GCC was built with.

    GCC finds the macOS headers through the SDK path it was configured with,
    and was built for the linker of CI's runner.
    """
    try:
        with urllib.request.urlopen(f"{url}/{name}.sdk") as response:
            built_with = response.read().decode().strip()
    except (urllib.error.URLError, OSError):
        return
    script = ROOT / ".github" / "scripts" / "macos-sdk-id.sh"
    result = subprocess.run([str(script)], capture_output=True, text=True)
    here = result.stdout.strip()
    if result.returncode == 0 and here != built_with:
        print(
            f"warning: this g++ was built with {built_with}, but this Mac has "
            f"{here}. If it can't find the system headers or fails to link, "
            "build GCC with --build-compilers --compiler gcc.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Name, pack or fetch a CI-built compiler toolchain archive."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    name_parser = commands.add_parser(
        "name", help="Print the asset name for this checkout and host."
    )
    name_parser.add_argument("compiler", choices=SUBMODULES)
    name_parser.add_argument(
        "--cache-key", help="Fail unless this cache key has the same pin and recipe."
    )

    pack_parser = commands.add_parser(
        "pack", help="Pack an installed toolchain, with its checksum."
    )
    pack_parser.add_argument("compiler", choices=SUBMODULES)
    pack_parser.add_argument("prefix", type=Path)
    pack_parser.add_argument("out_dir", type=Path)
    pack_parser.add_argument(
        "--macos-sdk", default="", help="SDK and linker GCC was built with (macOS)."
    )

    fetch_parser = commands.add_parser(
        "fetch", help="Download and unpack this checkout's toolchain."
    )
    fetch_parser.add_argument("compiler", choices=SUBMODULES)
    fetch_parser.add_argument("dest", type=Path)
    fetch_parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Where the assets are. Default: {DEFAULT_URL}.",
    )

    args = parser.parse_args()
    if args.command == "name":
        if args.cache_key:
            check_cache_key(args.compiler, args.cache_key)
        print(asset_name(args.compiler))
    elif args.command == "pack":
        pack(args.compiler, args.prefix, args.out_dir, args.macos_sdk)
    else:
        fetch(args.compiler, args.dest.resolve(), args.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
