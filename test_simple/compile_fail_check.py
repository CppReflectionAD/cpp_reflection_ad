#!/usr/bin/env python3
"""Check that a compile-fail test fails with the error(s) it declares.

A compile-fail test (tests/static_fail/) names the error(s) it must fail with
in one or more

    // EXPECT-ERROR: <text>

lines in the comment block at the top of the file. It passes only if
compilation fails and each <text> appears literally (not as a regex) in an
error diagnostic of its own: the error's message or the message of one of the
notes the compiler attaches to it. At least one directive is required, so a
test that fails for an unrelated reason (a broken include, a typo) does not
pass by accident. Anything that reads like a misspelt directive
(`//EXPECT-ERROR:`, `// expected-error:`, `// EXPECT-ERROR <text>`) or like a
directive outside that comment block (`foo(); // EXPECT-ERROR: <text>`) is
rejected rather than silently ignored. The directives are read before anything
is compiled, so a test without a valid one fails at once.

This is the single implementation of that rule: run_tests.py imports it, and
CTest runs it as a script around the build of each compile_check(... TRUE)
target:

    compile_fail_check.py --source <file.cpp> -- <build command...>
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path


EXPECT_ERROR_DIRECTIVE = "// EXPECT-ERROR:"
# A `//` comment (including `///` and `//!`) that reads like an EXPECT-ERROR
# directive: the hyphen or underscore spelling with or without a colon, or the
# spaced spelling with one (so prose like "expected errors are listed below"
# is left alone). A line that matches this but is not an exact directive is a
# mistake, not something to skip.
_NEAR_MISS_RE = re.compile(
    r"//[/!]*\s*expect(?:ed)?(?:[-_]errors?\b|\s+errors?\s*:)", re.IGNORECASE
)
# String and character literals, and `/* ... */` comments that close on the
# same line, are blanked out before a line of code is searched for a
# directive-like comment.
_NOT_A_COMMENT_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])+\'|/\*.*?\*/')
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# The header line of a diagnostic, from clang or gcc:
#   <file>:<line>[:<col>]: <kind>: <message>
#   <pseudo-file>: <kind>: <message>   (`<command-line>`, `<built-in>`)
#   <tool>: <kind>: <message>          (`clang++`, `cc1plus`, `/usr/bin/ld`)
#   <kind>: <message>
# Only <message> is searched. Matching the whole line would let text that
# happens to occur in the file path match any error at all, and the
# compilers echo source lines (which may contain `error:` inside a string
# literal) indented under the header, so those never match.
_DIAGNOSTIC_HEADER_RE = re.compile(
    r"^(?:(?:\S.*?:\d+(?::\d+)?|<[^>]+>|[\w./+-]+): )?"
    r"(?P<kind>(?:fatal )?error|warning|note|remark): (?P<message>.*)$"
)


class DirectiveError(ValueError):
    """The test's `// EXPECT-ERROR:` directives are missing or malformed."""


def leading_comment_lines(source: str) -> list[tuple[int, str]]:
    """Return (line number, stripped line) for the comment block atop `source`.

    The block is every line before the first one that is neither blank nor a
    `//` comment; it is where test directives (`// EXPECT-ERROR:` here,
    `// TEST-FLAGS:` in run_tests.py) go. Only the block is read, not the
    whole file.
    """
    lines = []
    for number, line in enumerate(io.StringIO(source), start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            break
        lines.append((number, stripped))
    return lines


def read_comment_directives(source: str, directive: str) -> list[tuple[int, str]]:
    """Return (line number, text) for every `<directive> <text>` line in `source`.

    The one placement rule for test directives: the directive starts a line
    of the comment block at the top of the file (see leading_comment_lines),
    after optional indentation; <text> is the rest of that line, stripped.
    """
    return [
        (number, line[len(directive) :].strip())
        for number, line in leading_comment_lines(source)
        if line.startswith(directive)
    ]


def parse_expected_errors(source: str) -> tuple[str, ...]:
    """Return the <text> of every `// EXPECT-ERROR: <text>` in `source`."""
    header = leading_comment_lines(source)
    expected = []
    for number, line in header:
        if line.startswith(EXPECT_ERROR_DIRECTIVE):
            text = line[len(EXPECT_ERROR_DIRECTIVE) :].strip()
            if not text:
                raise DirectiveError(
                    f"line {number}: empty `{EXPECT_ERROR_DIRECTIVE}` directive"
                )
            expected.append(text)
        elif _NEAR_MISS_RE.match(line):
            raise DirectiveError(
                f"line {number}: malformed directive {line!r}; write "
                f"`{EXPECT_ERROR_DIRECTIVE} <text>`"
            )
    body = list(io.StringIO(source))[len(header) :]
    for number, line in enumerate(body, start=len(header) + 1):
        if _NEAR_MISS_RE.search(_NOT_A_COMMENT_RE.sub("", line)):
            raise DirectiveError(
                f"line {number}: directive {line.strip()!r} is below the "
                f"first line of code; put `{EXPECT_ERROR_DIRECTIVE} <text>` "
                "in the comment block at the top of the file"
            )
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
    only in a note). Notes after a warning belong to the warning.
    """
    diagnostics: list[list[str]] = []
    current: list[str] | None = None
    for line in _ANSI_ESCAPE_RE.sub("", output).splitlines():
        match = _DIAGNOSTIC_HEADER_RE.match(line)
        if not match:
            continue
        kind, message = match.group("kind", "message")
        if kind == "note":
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


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("command", nargs="+", help="build command (after --)")
    args = parser.parse_args(argv)

    try:
        expected = parse_expected_errors(read_source(args.source))
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
