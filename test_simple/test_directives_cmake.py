#!/usr/bin/env python3
"""Write the response file that gives a CTest test the flags its directives say.

The build runs this for each test (test_flags in test_simple_cmake.cmake),
before compiling it and again whenever the test changes, so that CTest builds
a test with the same flags as run_tests.py, and rejects the same invalid
directives, from the same reader (source_directives.read_test_directives):

    test_directives_cmake.py --compiler <clang|gcc> --source <file.cpp>
        --output <file.rsp> [--must-fail]

The response file, which the compiler reads as `@<file.rsp>` after
CMAKE_CXX_FLAGS and the target's own options, holds
compile_fail_check.compile_flags: the test's TEST-FLAGS for the compiler,
then, for a test that must fail to compile, the diagnostic flags. It is
rewritten only when they change, so that a change to the test that leaves
its flags alone, or to these scripts, doesn't recompile every test.

If the test's directives are invalid, this prints why, removes the response
file and fails, so building the test fails.
"""

from __future__ import annotations

import argparse
import re
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


def test_flags(path: Path, compiler: str, must_fail: bool) -> list[str]:
    """The flags to compile the test at `path` with.

    Raises DirectiveError if its directives are invalid.
    """
    directives = source_directives.load_test_directives(path, compiler, must_fail)
    return list(compile_fail_check.compile_flags(directives, must_fail))


def main(argv: list[str] | None = None) -> int:
    # No docstring under `python3 -OO`.
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--compiler", choices=source_directives.COMPILERS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--must-fail", action="store_true")
    args = parser.parse_args(argv)

    try:
        flags = test_flags(args.source, args.compiler, args.must_fail)
    except source_directives.DirectiveError as error:
        # No response file, so the build reruns this once the test is fixed.
        try:
            args.output.unlink()  # (missing_ok needs Python 3.8)
        except FileNotFoundError:
            pass
        print(f"{args.source}: {error}", file=sys.stderr)
        return 1
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
