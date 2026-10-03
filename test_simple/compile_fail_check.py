#!/usr/bin/env python3
"""Check that a compile-fail test fails with the error(s) it declares.

A compile-fail test (tests/static_fail/) names the error(s) it must fail with
in one or more

    // EXPECT-ERROR: <text>

comments at the top of the file, before any code (see source_directives). It
passes only if compilation fails and each <text> appears literally (not as a
regex) in an error diagnostic of its own: the error's message or the message
of one of the notes the compiler attaches to it (`note:` lines, or the
indented `•` lines of gcc's nested diagnostics). At least one directive is
required, so a test that fails for an unrelated reason (a broken include, a
typo) does not pass by accident. Anything that reads like a misspelt directive
(`//EXPECT-ERROR:`, `// expected-error:`, `// EXPECT-ERROR <text>`) or like a
directive after code (`foo(); // EXPECT-ERROR: <text>`) is rejected rather
than silently ignored. The directives are read before anything is compiled, so
a test without a valid one fails at once.

This is the single implementation of that check: run_tests.py imports it, and
CTest runs it as a script around the build of each compile_check(... TRUE)
target:

    compile_fail_check.py --source <file.cpp> -- <build command...>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePath

if __package__:
    from . import source_directives
else:  # run as a script; its directory isn't on sys.path under `python3 -P`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import source_directives

DirectiveError = source_directives.DirectiveError
read_source = source_directives.read_source

EXPECT_ERROR_DIRECTIVE = "// EXPECT-ERROR:"
EXPECT_ERROR = source_directives.DirectiveFamily(
    pattern=re.compile(r"// (?P<name>EXPECT-ERROR):"),
    # A `//` comment (including `///` and `//!`) that reads like an
    # EXPECT-ERROR directive: the hyphen or underscore spelling with or
    # without a colon, or the spaced spelling with one (so prose like
    # "expected errors are listed below" is left alone).
    near_miss=re.compile(
        r"//[/!]*\s*expect(?:ed)?(?:[-_]errors?\b|\s+errors?\s*:)", re.IGNORECASE
    ),
    usage=f"{EXPECT_ERROR_DIRECTIVE} <text>",
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# The header line of a diagnostic, from clang or gcc:
#   <file>:<line>[:<col>]: <kind>: <message>
#   <pseudo-file>: <kind>: <message>   (`<command-line>`, `<built-in>`)
#   <tool>: <kind>: <message>          (`clang++`, `cc1plus`; see below)
#   <kind>: <message>
# Only <message> is searched. Matching the whole line would let text that
# happens to occur in the file path match any error at all, and the
# compilers echo source lines (which may contain `error:` inside a string
# literal) indented under the header, so those never match.
_DIAGNOSTIC_HEADER_RE = re.compile(
    r"^(?:(?:\S.*?:\d+(?::\d+)?|<[^>]+>|(?P<tool>[\w./+-]+)): )?"
    r"(?P<kind>(?:fatal )?error|warning|note|remark): (?P<message>.*)$"
)
# A <tool> header counts only if the tool is the compiler (driver or cc1),
# not the linker (`collect2`, `ld.lld`) or the build tool (`ninja`, `make`);
# nor does the driver's report that the linker failed.
_COMPILER_TOOL_RE = re.compile(r"clang|gcc|g\+\+|c\+\+|cc1")
_LINKER_FAILED = "linker command failed"
# A note in gcc's nested diagnostics (how the gcc trunk this repo builds
# prints them by default): an indented bullet line under the error, `•` or,
# outside a UTF-8 locale, `*`. Source lines the compilers echo are indented
# too, but start with a `<line> |` or `|` gutter.
_NESTED_NOTE_RE = re.compile(r"^ +[\u2022*] (?P<message>.*)$")


def parse_expected_errors(source: str) -> tuple[str, ...]:
    """Return the <text> of every `// EXPECT-ERROR: <text>` in `source`."""
    directives = source_directives.read_directives(source, EXPECT_ERROR)
    for directive in directives:
        if not directive.text:
            raise DirectiveError(
                f"line {directive.line}: empty `{EXPECT_ERROR_DIRECTIVE}` directive"
            )
    expected = [directive.text for directive in directives]
    if not expected:
        raise DirectiveError(
            f"no `{EXPECT_ERROR_DIRECTIVE} <text>` directive; a compile-fail "
            "test must say which error it expects"
        )
    return tuple(expected)


def error_diagnostics(output: str) -> list[tuple[str, ...]]:
    """The error diagnostics in compiler `output`.

    Each is the error's message followed by the messages of the notes that
    follow it (clang, for one, gives the reason a constant expression failed
    only in a note), whether as `note:` lines or as gcc's nested bullet
    lines. Notes after a warning belong to the warning.
    """
    diagnostics: list[list[str]] = []
    current: list[str] | None = None
    for line in _ANSI_ESCAPE_RE.sub("", output).splitlines():
        match = _DIAGNOSTIC_HEADER_RE.match(line)
        if not match:
            nested = _NESTED_NOTE_RE.match(line)
            if nested and current is not None:
                current.append(nested.group("message"))
            continue
        tool, kind, message = match.group("tool", "kind", "message")
        if tool is not None and (
            not _COMPILER_TOOL_RE.search(PurePath(tool).name)
            or message.startswith(_LINKER_FAILED)
        ):
            current = None
        elif kind == "note":
            if current is not None:
                current.append(message)
        elif kind.endswith("error"):
            current = [message]
            diagnostics.append(current)
        else:
            current = None
    return [tuple(diagnostic) for diagnostic in diagnostics]


def _unmatched(
    expected: tuple[str, ...], diagnostics: list[tuple[str, ...]]
) -> list[str]:
    """The expected texts left over when each is given an error of its own.

    A maximum bipartite matching (augmenting paths), so the outcome does not
    depend on the order of the directives.
    """
    owner: dict[int, int] = {}  # diagnostic index -> expected index

    def claim(index: int, seen: set[int]) -> bool:
        for d, diagnostic in enumerate(diagnostics):
            if d in seen or not any(expected[index] in m for m in diagnostic):
                continue
            seen.add(d)
            if d not in owner or claim(owner[d], seen):
                owner[d] = index
                return True
        return False

    return [text for index, text in enumerate(expected) if not claim(index, set())]


def check(expected: tuple[str, ...], returncode: int, output: str) -> str | None:
    """Return why a compile-fail test failed, or None if it passed.

    `expected` is what parse_expected_errors returned, before compiling.
    """
    if returncode == 0:
        return "expected a compile error, but it compiled"
    diagnostics = error_diagnostics(output)
    missing = [
        text
        for text in expected
        if not any(text in message for d in diagnostics for message in d)
    ]
    if missing:
        return "compilation failed, but not with the expected error(s)\nmissing:\n" + (
            "\n".join(f"  {text}" for text in missing)
        )
    unmatched = _unmatched(expected, diagnostics)
    if unmatched:
        return (
            "compilation failed, but not with a separate error per expected error"
            "\nonly in an error already matched by another directive:\n"
            + "\n".join(f"  {text}" for text in unmatched)
        )
    return None


def main(argv: list[str] | None = None) -> int:
    # No docstring under `python3 -OO`.
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("command", nargs="+", help="build command (after --)")
    args = parser.parse_args(argv)

    try:
        expected = parse_expected_errors(read_source(args.source))
    except OSError as error:
        print(f"{args.source}: cannot read the test: {error}")
        return 1
    except DirectiveError as error:
        print(f"{args.source}: {error}")
        return 1

    # One stream, so stdout and stderr lines stay whole and in order.
    build = subprocess.run(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    sys.stdout.write(build.stdout)
    reason = check(expected, build.returncode, build.stdout)
    if reason is None:
        return 0
    print(f"\n{args.source}: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
