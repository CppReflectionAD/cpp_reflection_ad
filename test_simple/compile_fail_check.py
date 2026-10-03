#!/usr/bin/env python3
"""Check that a compile-fail test fails with the error(s) it declares.

A compile-fail test (tests/static_fail/) names the error(s) it must fail with
in one or more

    // EXPECT-ERROR: <text>

comments, each on a line of its own, anywhere in the file. It passes only if
compilation fails and every <text> appears literally (not as a regex) in the
message of an error diagnostic. At least one directive is required, so a test
that fails for an unrelated reason (a broken include, a typo) does not pass by
accident, and anything that looks like a misspelt or misplaced directive
(`//EXPECT-ERROR:`, `// expect_error:`, `// EXPECT-ERROR:` after code) is
rejected rather than silently ignored. The directives are read before anything
is compiled, so a test without a valid one fails at once.

This is the single implementation of that rule: run_tests.py imports it, and
CTest runs it as a script around the build of each compile_check(... TRUE)
target:

    compile_fail_check.py --source <file.cpp> -- <build command...>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


EXPECT_ERROR_DIRECTIVE = "// EXPECT-ERROR:"
# Anything that reads like an EXPECT-ERROR directive. A line that matches
# this but is not an exact directive is a mistake, not something to skip.
_NEAR_MISS_RE = re.compile(r"\bexpect[-_ ]?errors?\s*:", re.IGNORECASE)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# The header line of an error diagnostic, from clang or gcc:
#   <file>:<line>[:<col>]: [fatal ]error: <message>
#   <pseudo-file>: [fatal ]error: <message>   (`<command-line>`, `<built-in>`)
#   <tool>: [fatal ]error: <message>          (`clang++`, `cc1plus`, `/usr/bin/ld`)
#   [fatal ]error: <message>
# Only <message> is searched. Matching the whole line would let text that
# happens to occur in the file path match any error at all, and the
# compilers echo source lines (which may contain `error:` inside a string
# literal) indented under the header, so those never match.
_ERROR_HEADER_RE = re.compile(
    r"^(?:(?:\S.*?:\d+(?::\d+)?|<[^>]+>|[\w./+-]+): )?"
    r"(?:fatal )?error: (?P<message>.*)$"
)


class DirectiveError(ValueError):
    """The test's `// EXPECT-ERROR:` directives are missing or malformed."""


def read_comment_directives(source: str, directive: str) -> list[tuple[int, str]]:
    """Return (line number, text) for every `<directive> <text>` line in `source`.

    The one placement rule for test directives (`// EXPECT-ERROR:` here,
    `// TEST-FLAGS:` in run_tests.py): the directive starts a line of its own,
    after optional indentation, anywhere in the file; <text> is the rest of
    that line, stripped.
    """
    directives = []
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(directive):
            directives.append((number, stripped[len(directive) :].strip()))
    return directives


def parse_expected_errors(source: str) -> tuple[str, ...]:
    """Return the <text> of every `// EXPECT-ERROR: <text>` in `source`."""
    directives = dict(read_comment_directives(source, EXPECT_ERROR_DIRECTIVE))
    for number, line in enumerate(source.splitlines(), start=1):
        if number not in directives and _NEAR_MISS_RE.search(line):
            raise DirectiveError(
                f"line {number}: malformed directive {line.strip()!r}; write "
                f"`{EXPECT_ERROR_DIRECTIVE} <text>` on a line of its own"
            )
    for number, text in directives.items():
        if not text:
            raise DirectiveError(
                f"line {number}: empty `{EXPECT_ERROR_DIRECTIVE}` directive"
            )
    if not directives:
        raise DirectiveError(
            f"no `{EXPECT_ERROR_DIRECTIVE} <text>` directive; a compile-fail "
            "test must say which error it expects"
        )
    return tuple(directives.values())


def error_messages(output: str) -> list[str]:
    """The messages of the error diagnostics in compiler `output`."""
    messages = []
    for line in _ANSI_ESCAPE_RE.sub("", output).splitlines():
        match = _ERROR_HEADER_RE.match(line)
        if match:
            messages.append(match.group("message"))
    return messages


def check(expected: tuple[str, ...], returncode: int, output: str) -> str | None:
    """Return why a compile-fail test failed, or None if it passed.

    `expected` is what parse_expected_errors returned, before compiling.
    """
    if returncode == 0:
        return "expected a compile error, but it compiled"
    messages = error_messages(output)
    missing = [
        text for text in expected if not any(text in message for message in messages)
    ]
    if missing:
        return "compilation failed, but not with the expected error(s)\nmissing:\n" + (
            "\n".join(f"  {text}" for text in missing)
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
