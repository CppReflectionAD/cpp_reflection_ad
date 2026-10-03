#!/usr/bin/env python3
"""Write the response file that gives a CTest test the flags its directives say.

The build runs this for each test (test_flags in test_simple_cmake.cmake),
before compiling it and again whenever the test or the user's flags change,
so that CTest builds a test with the same flags as run_tests.py, and rejects
the same invalid directives, from the same reader
(source_directives.read_test_directives):

    test_directives_cmake.py --compiler <clang|gcc> --source <file.cpp>
        --user-flags-file <file> --output <file.rsp> [--must-fail]

The response file, which the compiler reads as `@<file.rsp>` after the
target's own options (when compiling the test, and when linking it if it is
an executable), holds compile_fail_check.compile_flags: the test's
TEST-FLAGS for the compiler, then the user's flags (CMAKE_CXX_FLAGS and
those of the build type, which test_simple_cmake.cmake writes to <file>;
split as a shell would), so that they override the test's, as
--extra-cxxflag does in run_tests.py, then, for a test that must fail to
compile, the diagnostic flags. It is rewritten only when they change, so
that a change to the test that leaves its flags alone, or to these scripts,
doesn't recompile every test.

If the test's directives are invalid, or the user's flags can't be read or
split, this prints why, removes the response file and fails, so building
the test fails.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

if __package__:
    from . import compile_fail_check, source_directives
else:  # run as a script; its directory isn't on sys.path under `python3 -P`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compile_fail_check
    import source_directives

# What gcc's and clang's (GNU) response file readers take as other than a
# plain character; a backslash before it means the character itself in both.
# (Not shlex.quote: both readers take a backslash inside single quotes as an
# escape too, so `'a\b'` would read as `ab`.)
_RESPONSE_FILE_SPECIAL_RE = re.compile(r"""([\s'"\\])""")


def response_file_argument(argument: str) -> str:
    """`argument` as one argument in a response file."""
    if not argument:
        return "''"
    return _RESPONSE_FILE_SPECIAL_RE.sub(r"\\\1", argument)


def test_flags(
    path: Path, compiler: str, must_fail: bool, user_flags: list[str]
) -> list[str]:
    """The flags to compile the test at `path` with.

    Raises DirectiveError if its directives are invalid.
    """
    directives = source_directives.load_test_directives(path, compiler, must_fail)
    return list(compile_fail_check.compile_flags(directives, must_fail, user_flags))


def _fail(output: Path, reason: str) -> int:
    """Print `reason`, remove the response file `output`, so that the build
    runs this again once the cause is fixed, and return the exit status."""
    try:
        output.unlink()  # (missing_ok needs Python 3.8)
    except FileNotFoundError:
        pass
    print(reason, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    # No docstring under `python3 -OO`.
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--compiler", choices=source_directives.COMPILERS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--user-flags-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--must-fail", action="store_true")
    args = parser.parse_args(argv)

    try:
        user_flags = shlex.split(args.user_flags_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # unreadable, or unbalanced quotes
        return _fail(
            args.output,
            f"can't read the user's flags (CMAKE_CXX_FLAGS and those of the "
            f"build type) from {args.user_flags_file}: {error}",
        )
    try:
        flags = test_flags(args.source, args.compiler, args.must_fail, user_flags)
    except source_directives.DirectiveError as error:
        return _fail(args.output, f"{args.source}: {error}")
    content = "".join(f"{response_file_argument(flag)}\n" for flag in flags)
    try:
        unchanged = args.output.read_text(encoding="utf-8") == content
    except OSError:
        unchanged = False
    if not unchanged:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
