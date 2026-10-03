#!/usr/bin/env python3
"""Check that a compile-fail test fails with the error(s) it declares.

A compile-fail test (tests/static_fail/) names the error(s) it must fail with
in one or more

    // EXPECT-ERROR: <text>

comments anywhere in the file. It passes only if compilation fails and every
<text> appears literally (not as a regex) in the message of an error
diagnostic. At least one directive is required, so a test that fails for an
unrelated reason (a broken include, a typo) does not pass by accident.

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
_DIRECTIVE_RE = re.compile(r"^[ \t]*" + re.escape(EXPECT_ERROR_DIRECTIVE) + r"(.*)$")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# The header line of an error diagnostic, from clang or gcc:
#   <file>:<line>[:<col>]: [fatal ]error: <message>
#   <tool>: [fatal ]error: <message>      (e.g. `clang++:`, `cc1plus:`)
# Only <message> is searched. Matching the whole line would let text that
# happens to occur in the file path match any error at all, and the
# compilers echo source lines (which may contain `error:` inside a string
# literal) indented under the header, so those never match.
_ERROR_HEADER_RE = re.compile(
    r"^(?:\S.*?:\d+(?::\d+)?|[\w.+-]+): (?:fatal )?error: (?P<message>.*)$"
)


class DirectiveError(ValueError):
    """The test's `// EXPECT-ERROR:` directives are missing or malformed."""


def parse_expected_errors(source: str) -> tuple[str, ...]:
    """Return the <text> of every `// EXPECT-ERROR: <text>` in `source`."""
    expected = []
    for line in source.splitlines():
        match = _DIRECTIVE_RE.match(line)
        if match:
            text = match.group(1).strip()
            if not text:
                raise DirectiveError(f"empty `{EXPECT_ERROR_DIRECTIVE}` directive")
            expected.append(text)
    if not expected:
        raise DirectiveError(
            f"no `{EXPECT_ERROR_DIRECTIVE} <text>` directive; a compile-fail "
            "test must say which error it expects"
        )
    return tuple(expected)


def error_messages(output: str) -> list[str]:
    """The messages of the error diagnostics in compiler `output`."""
    messages = []
    for line in _ANSI_ESCAPE_RE.sub("", output).splitlines():
        match = _ERROR_HEADER_RE.match(line)
        if match:
            messages.append(match.group("message"))
    return messages


def check(source: str, returncode: int, output: str) -> str | None:
    """Return why a compile-fail test failed, or None if it passed."""
    try:
        expected = parse_expected_errors(source)
    except DirectiveError as error:
        return str(error)
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

    # One stream, so stdout and stderr lines stay whole and in order.
    build = subprocess.run(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    sys.stdout.write(build.stdout)
    reason = check(read_source(args.source), build.returncode, build.stdout)
    if reason is None:
        return 0
    print(f"\n{args.source}: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
