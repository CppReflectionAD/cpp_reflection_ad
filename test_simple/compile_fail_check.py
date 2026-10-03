#!/usr/bin/env python3
"""Check that a compile-fail test fails with the error(s) it declares.

A compile-fail test (a static_fail/ directory) names the error(s) it must
fail with in one or more

    // EXPECT-ERROR: <text>
    // EXPECT-ERROR-CLANG: <text>   (only when building with clang)
    // EXPECT-ERROR-GCC: <text>     (only when building with gcc)

comments at the top of the file, before any code (see source_directives). It
passes only if compilation fails without the compiler crashing, and each
<text> that applies to the compiler appears literally (not as a regex) in an
error diagnostic of its own: the error's message or the message of one of the
notes the compiler attaches to it (`note:` lines, or the indented `•` lines
of gcc's nested diagnostics), other than the notes that only say where the
error was reached from (`in instantiation of ... requested here`, `in call
to ...`, `required from here`). At least one directive must apply to each
compiler, so a test that fails for an unrelated reason (a broken include, a
typo) does not pass by accident. Anything that reads like a misspelt
directive (`//EXPECT-ERROR:`, `// expected-error:`, `// EXPECT-ERROR <text>`,
`// EXPECT-ERROR-CLNAG:`) or like a directive after code
(`foo(); // EXPECT-ERROR: <text>`) is rejected rather than silently ignored.
The directives (TEST-FLAGS included) are read with
source_directives.read_test_directives before anything is compiled, so a test
without a valid one fails at once.

The compiler must run with COMPILE_ENV and the flags compile_flags gives the
test, so that its output is in the form parsed here, and must only compile the
test (`-c`), not link it, so that its output holds no linker errors.

This is the single implementation of that check: run_tests.py imports it, and
CTest runs it as a script around the build of each compile_check(... TRUE)
target:

    compile_fail_check.py --source <file.cpp> --compiler <clang|gcc> -- <build command...>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path, PurePath
from typing import Sequence

if __package__:
    from . import source_directives
else:  # run as a script; its directory isn't on sys.path under `python3 -P`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import source_directives

# The environment and flags the compiler must run with. In the C locale the
# compilers write English and ASCII quotes (an NLS-enabled gcc otherwise
# translates `error:` and quotes with ‘’ in a UTF-8 locale). The forced
# `<line> |` gutter on echoed source lines keeps a source line such as
# ` * comment` from reading as one of gcc's nested notes; it must come after
# any other diagnostic flags, so that it wins.
COMPILE_ENV = {"LC_ALL": "C"}
DIAGNOSTIC_FLAGS = ("-fdiagnostics-show-line-numbers",)

# Colours and other CSI sequences, and the OSC 8 hyperlinks gcc wraps around
# an option name when URLs are enabled (`-fdiagnostics-urls=always`), which
# end with ST (ESC \) or BEL.
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x1b\x07]*(?:\x1b\\|\x07)"
)
# Quotes are normalized on both sides of a match, so that a directive copied
# from gcc's output in a UTF-8 locale (‘foo’) matches the ASCII quotes it
# prints under COMPILE_ENV, and the other way round.
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
# The header line of a diagnostic, from clang or gcc:
#   <file>:<line>[:<col>]: <kind>: <message>
#   <pseudo-file>: <kind>: <message>   (`<command-line>`, `<built-in>`)
#   <file-or-tool>: <kind>: <message>  (gcc's file-level diagnostics, and
#                                       `clang++`, `cc1plus`; see below; a
#                                       file's path may contain spaces)
#   <kind>: <message>
# Only <message> is searched. Matching the whole line would let text that
# happens to occur in the file path match any error at all, and the
# compilers echo source lines (which may contain `error:` inside a string
# literal) indented under the header, so those never match.
_DIAGNOSTIC_HEADER_RE = re.compile(
    r"^(?:(?:\S.*?:\d+(?::\d+)?|<[^>]+>|(?P<prefix>[^\s:](?:[^:]*[^\s:])?)): )?"
    r"(?P<kind>(?:fatal )?error|sorry, unimplemented|internal compiler error"
    r"|warning|note|remark): (?P<message>.*)$"
)
_ERROR_KINDS = frozenset({"error", "fatal error", "sorry, unimplemented"})
# A <file-or-tool> prefix with a C or C++ source or header extension is a
# file (gcc prints a diagnostic without a line that way); otherwise it is a
# tool, which counts only if it is the compiler: its whole name is a driver
# or cc1 name, with any target prefix and version suffix (`clang++`,
# `x86_64-linux-gnu-g++-14`, `cc1plus`), not another tool that merely
# contains one (`clang-linker-wrapper`, `gcc-ar`) or the build tool
# (`ninja`, `make`).
_SOURCE_FILE_RE = re.compile(
    r"\.(?:c|cc|cp|cpp|cxx|c\+\+|cppm|ixx|ii|h|hh|hpp|hxx|h\+\+|ipp|tpp|tcc|inc|C|H)$"
)
_COMPILER_TOOL_RE = re.compile(
    r"(?:[\w.]+-)*(?:clang(?:\+\+)?|gcc|g\+\+|c\+\+|cc|cc1(?:plus)?)"
    r"(?:-\d+(?:\.\d+)*)?(?:\.exe)?"
)
# A note in gcc's nested diagnostics (how the gcc trunk this repo builds
# prints them by default): an indented bullet line under the error, `•` or
# `*`. Source lines the compilers echo are indented too, but start with the
# `<line> |` or `|` gutter that DIAGNOSTIC_FLAGS asks for.
_NESTED_NOTE_RE = re.compile(r"^ +[\u2022*] (?P<message>.*)$")
# gcc's context lines, which it prints between a diagnostic's notes too
# (`In file included from ...:` before a note in a header): the include
# stack, `<file>: In function ...:` / `At global scope:`, and
# `<file>:<line>:<col>:   required from ...`.
_CONTEXT_RE = re.compile(
    r"^(?:In file included from |In module imported at "
    r"|\S.*?: (?:In |At global scope:)|\S.*?:\d+(?::\d+)?:   )"
)
# A note that only says where the error was reached from, which may name
# any function or type on the way (`ad::inverse<double>`), so it is not
# searched: clang's template instantiation and constexpr call stacks and
# macro expansions, and gcc's `required from` / `In substitution of`
# context, which its nested diagnostics print as bullets. Other notes (the
# reason a constant expression or a constraint failed, a candidate) are.
_BACKTRACE_NOTE_RE = re.compile(
    r"^(?:in instantiation of |in call to |in evaluation of |in implicit "
    r"|in expansion of |in definition of |in 'constexpr' expansion of "
    r"|while |during |required (?:from|by|for) |expanded from |\(skipping "
    r"|(?:\S.*?: )?In (?:instantiation|substitution) of )"
    r"|(?:requested|required|needed) here$"
)
# What a compiler prints when it crashes, which no test may pass by: an ICE
# (gcc), the driver's report that cc1/cc1plus died, a crash in clang's
# frontend.
_CRASH_RE = re.compile(
    r"internal compiler error|signal terminated program"
    r"|frontend command failed|PLEASE submit a bug report"
)


def compile_flags(
    directives: source_directives.TestDirectives, must_fail: bool
) -> tuple[str, ...]:
    """The flags a test is compiled with, after the build's and the user's
    own: its TEST-FLAGS, then, if it must fail to compile, DIAGNOSTIC_FLAGS,
    last so that they win.

    Both runners build a test with these: run_tests.py on its command line,
    CTest through the response file test_directives_cmake.py writes.
    """
    return directives.flags + (DIAGNOSTIC_FLAGS if must_fail else ())


def _crash(returncode: int, lines: list[str]) -> str | None:
    """How the compiler crashed, or None if it didn't."""
    for line in lines:
        if line[:1] not in ("", " ") and _CRASH_RE.search(line):
            return line
    if returncode < 0:  # killed by a signal
        return f"killed by signal {-returncode}"
    return None


def _is_compiler(prefix: str) -> bool:
    name = PurePath(prefix).name
    return bool(_SOURCE_FILE_RE.search(name) or _COMPILER_TOOL_RE.fullmatch(name))


def error_diagnostics(output: str) -> list[tuple[str, ...]]:
    """The error diagnostics in compiler `output`.

    Each is the error's message followed by the messages of the notes that
    follow it (clang, for one, gives the reason a constant expression failed
    only in a note), whether as `note:` lines or as gcc's nested bullet
    lines, except backtrace notes (see _BACKTRACE_NOTE_RE). Notes belong to the diagnostic just before them, so those after a
    warning, or after any other line that isn't indented (a diagnostic of
    another kind, build-tool output) other than gcc's context lines, belong
    to no error. gcc's `sorry, unimplemented:` counts as an error.
    """
    diagnostics: list[list[str]] = []
    current: list[str] | None = None
    for line in _ANSI_ESCAPE_RE.sub("", output).translate(_QUOTES).splitlines():
        match = _DIAGNOSTIC_HEADER_RE.match(line)
        if not match:
            nested = _NESTED_NOTE_RE.match(line)
            if nested:
                if current is not None and not _BACKTRACE_NOTE_RE.match(
                    nested.group("message")
                ):
                    current.append(nested.group("message"))
            elif line[:1] not in ("", " ") and not _CONTEXT_RE.match(line):
                current = None
            continue
        prefix, kind, message = match.group("prefix", "kind", "message")
        if prefix is not None and not _is_compiler(prefix):
            current = None
        elif kind == "note":
            if current is not None and not _BACKTRACE_NOTE_RE.match(message):
                current.append(message)
        elif kind in _ERROR_KINDS:
            current = [message]
            diagnostics.append(current)
        else:
            current = None
    return [tuple(diagnostic) for diagnostic in diagnostics]


def _in(text: str, diagnostic: tuple[str, ...]) -> bool:
    return any(text in message for message in diagnostic)


def _unmatched(
    expected: Sequence[str], diagnostics: list[tuple[str, ...]]
) -> list[int]:
    """The indices of the expected texts left over when each is given an
    error of its own.

    A maximum bipartite matching (augmenting paths), so the outcome does not
    depend on the order of the directives.
    """
    owner: dict[int, int] = {}  # diagnostic index -> expected index

    def claim(index: int, seen: set[int]) -> bool:
        for d, diagnostic in enumerate(diagnostics):
            if d in seen or not _in(expected[index], diagnostic):
                continue
            seen.add(d)
            if d not in owner or claim(owner[d], seen):
                owner[d] = index
                return True
        return False

    return [index for index in range(len(expected)) if not claim(index, set())]


def check(expected: tuple[str, ...], returncode: int, output: str) -> str | None:
    """Return why a compile-fail test failed, or None if it passed.

    `expected` is the test's expected_errors from
    source_directives.read_test_directives, read before compiling.
    """
    if returncode == 0:
        return "expected a compile error, but it compiled"
    crash = _crash(returncode, _ANSI_ESCAPE_RE.sub("", output).splitlines())
    if crash is not None:
        return f"the compiler crashed\n{crash}"
    diagnostics = error_diagnostics(output)
    needles = [text.translate(_QUOTES) for text in expected]
    missing: list[str] = []
    shared: list[str] = []
    for index in _unmatched(needles, diagnostics):
        in_some = any(_in(needles[index], d) for d in diagnostics)
        (shared if in_some else missing).append(expected[index])
    if not missing and not shared:
        return None
    sections = []
    if missing:
        sections.append(
            "missing:\n" + "\n".join(f"  {text}" for text in missing)
        )
    if shared:
        sections.append(
            "only in an error already matched by another directive:\n"
            + "\n".join(f"  {text}" for text in shared)
        )
    summary = (
        "compilation failed, but not with the expected error(s)"
        if missing
        else "compilation failed, but not with a separate error per expected error"
    )
    return "\n".join([summary, *sections])


def main(argv: list[str] | None = None) -> int:
    # No docstring under `python3 -OO`.
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--compiler", choices=source_directives.COMPILERS, required=True
    )
    parser.add_argument("command", nargs="+", help="build command (after --)")
    args = parser.parse_args(argv)

    try:
        expected = source_directives.load_test_directives(
            args.source, args.compiler, must_fail=True
        ).expected_errors
    except source_directives.DirectiveError as error:
        print(f"{args.source}: {error}")
        return 1

    # One stream, so stdout and stderr lines stay whole and in order. The
    # environment reaches the compiler through the build tool; the build
    # adds compile_flags (see test_directives_cmake).
    build = subprocess.run(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env={**os.environ, **COMPILE_ENV},
    )
    sys.stdout.write(build.stdout)
    reason = check(expected, build.returncode, build.stdout)
    if reason is None:
        return 0
    print(f"\n{args.source}: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
