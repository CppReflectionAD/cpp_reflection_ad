"""Read `// NAME: <text>` test directives from the top of a C++ source file.

A test declares things about itself (the error it must fail with, extra
compile flags) in `//` comments that come before its first line of code:

    // EXPECT-ERROR: ad::inverse requires an explicit inverse plan
    // TEST-FLAGS: -O2
    // TEST-FLAGS-CLANG: -fconstexpr-steps=16000000
    #include "..."

A directive may have a `-<COMPILER>` variant for each of COMPILERS, which
applies only when building with that compiler.

This is the single implementation of what a test declares, shared by both
test runners: run_tests.py and CTest (through compile_fail_check.py and
test_directives_cmake.py) read every test with read_test_directives. A
`//` comment at the top of the file that reads like a directive but is
misspelt (including a variant for a compiler not in COMPILERS), a directive
(its name and a colon) in a `/* */` comment or after the first code, and an
EXPECT-ERROR in a test that must compile are all rejected rather than
silently ignored. Comments are found with a small C++ lexer (see
lex_comments), so text inside string literals and raw strings is never
mistaken for a directive. A file is lexed once (read_comments), and each
family's directives are read from the result (read_directives).
"""

from __future__ import annotations

import bisect
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

# The compilers a directive can have a `-<COMPILER>` variant for; also the
# compilers run_tests.py builds with.
COMPILERS = ("clang", "gcc")


class DirectiveError(ValueError):
    """A test's directives can't be read: the file is unreadable, or its
    directives are missing or malformed."""


class Directive(NamedTuple):
    line: int
    name: str  # e.g. "EXPECT-ERROR", "TEST-FLAGS-CLANG"
    text: str  # the rest of the comment, stripped


@dataclass(frozen=True)
class DirectiveFamily:
    """The directives a reader looks for, and what reads like a mistake.

    `pattern` matches the `// NAME:` start of a well-formed directive, with
    the name in group "name"; `near_miss` matches the start of a `//` comment
    at the top of the file that reads like one (it should also match every
    well-formed one). `misplaced` matches the directive's own name, in upper
    case and with `-` or `_`, then any suffix (`S`, `-GCC`, `_CLNAG`) and a
    colon: a `//` comment after code, or a line of a `/* */` comment, that
    starts with it is a directive in the wrong place. (Prose such as
    `// expected error: none` or `// EXPECT-ERROR handling below` after code
    is left alone.)
    `usage` is how a well-formed directive is written, for error messages.
    """

    pattern: re.Pattern[str]
    near_miss: re.Pattern[str]
    misplaced: re.Pattern[str]
    usage: str

    @classmethod
    def with_compiler_variants(
        cls, name: str, argument: str, near_miss: re.Pattern[str]
    ) -> DirectiveFamily:
        """`// NAME: <argument>`, plus `// NAME-<COMPILER>:` for each of
        COMPILERS. A suffix naming any other compiler is not well formed, so
        it is reported if `near_miss` matches it."""
        variants = "|".join(compiler.upper() for compiler in COMPILERS)
        spelled = "[-_]".join(re.escape(word) for word in name.split("-"))
        return cls(
            pattern=re.compile(rf"// (?P<name>{re.escape(name)}(?:-(?:{variants}))?):"),
            near_miss=near_miss,
            misplaced=re.compile(rf"{spelled}[-\w]*\s*:"),
            usage=f"`// {name}: <{argument}>`, or `// {name}-<{variants}>: "
            f"<{argument}>` for one compiler",
        )


def for_compiler(
    directives: Sequence[Directive], name: str, compiler: str
) -> list[Directive]:
    """The directives in `directives` that apply to `compiler`: those named
    `name` and those named `name-<COMPILER>`, in order."""
    if compiler not in COMPILERS:
        raise ValueError(f"unknown compiler {compiler!r}; expected one of {COMPILERS}")
    names = (name, f"{name}-{compiler.upper()}")
    return [directive for directive in directives if directive.name in names]


class Comment(NamedTuple):
    line: int  # where it starts
    # From `//` to the end of the line, or from `/*` to `*/`; line splices
    # removed.
    text: str
    before_code: bool  # whether it comes before the first token of code
    block: bool = False  # a `/* */` comment


# A backslash-newline, which joins two lines before comments and literals are
# recognized (compilers allow whitespace between the two).
_SPLICE_RE = re.compile(r"\\[ \t]*\r?\n")
# The `//` of a comment, with any `///` / `//!` extra and the space after it.
_COMMENT_START_RE = re.compile(r"//[/!]*\s*")
_RAW_STRING_PREFIXES = frozenset({"R", "LR", "uR", "UR", "u8R"})
_RAW_DELIMITER_RE = re.compile(r'[^\s()\\"]{0,16}\(')


def read_source(path: Path) -> str:
    """The text of a test; a UTF-8 byte order mark is dropped.

    Raises OSError if the file can't be read.
    """
    return path.read_text(encoding="utf-8-sig", errors="replace")


def read_comments(path: Path) -> tuple[Comment, ...]:
    """The comments of the test at `path` (see lex_comments).

    Raises DirectiveError if the file can't be read.
    """
    try:
        source = read_source(path)
    except OSError as error:
        raise DirectiveError(f"cannot read the test: {error}") from None
    return lex_comments(source)


def lex_comments(source: str) -> tuple[Comment, ...]:
    """The `//` and `/* */` comments in C++ `source`, in order.

    A state machine over the characters of the file: code, `//` comment,
    `/* */` comment, string or character literal, raw string. It applies
    line splices (backslash-newline) everywhere except inside raw strings,
    reads `'` inside a number (`1'000`, `0x1'FF`) as a digit separator, and
    needs the exact R/LR/uR/UR/u8R prefix for a raw string. A quote whose
    literal isn't closed on its line (as in `#error don't` or prose inside
    `#if 0`) is a stray character, so a `//` comment after it is still
    found. It does not otherwise tokenize or preprocess: a `//` inside
    `#include <...>` would read as a comment.
    """
    newlines = [m.start() for m in re.finditer("\n", source)]
    n = len(source)

    def line_of(position: int) -> int:
        return bisect.bisect_left(newlines, position) + 1

    def skip_splices(i: int) -> int:
        while True:
            match = _SPLICE_RE.match(source, i)
            if match is None:
                return i
            i = match.end()

    def char_at(i: int) -> str:
        return source[i] if i < n else ""

    def literal_end(i: int, quote: str) -> int | None:
        # i is just past the opening quote; None if the line ends first.
        while True:
            i = skip_splices(i)
            ch = char_at(i)
            if ch in ("", "\n"):
                return None
            if ch == quote:
                return i + 1
            i = skip_splices(i + 1) + 1 if ch == "\\" else i + 1

    comments: list[Comment] = []
    seen_code = False
    i = 0
    while True:
        i = skip_splices(i)
        ch = char_at(i)
        if not ch:
            break
        after = skip_splices(i + 1)
        if ch == "/" and char_at(after) == "/":
            end = after + 1
            while True:
                end = skip_splices(end)
                if char_at(end) in ("", "\n"):
                    break
                end += 1
            text = _SPLICE_RE.sub("", source[i:end]).rstrip("\r")
            comments.append(Comment(line_of(i), text, not seen_code))
            i = end
        elif ch == "/" and char_at(after) == "*":
            end = after + 1
            while True:
                end = skip_splices(end)
                if end >= n:
                    break
                if source[end] == "*":
                    close = skip_splices(end + 1)
                    if char_at(close) == "/":
                        end = close + 1
                        break
                end += 1
            text = _SPLICE_RE.sub("", source[i:end])
            comments.append(Comment(line_of(i), text, not seen_code, block=True))
            i = end
        elif ch.isspace():
            i += 1
        else:
            seen_code = True
            if ch.isalpha() or ch == "_":
                identifier = []
                end = i
                while True:
                    end = skip_splices(end)
                    c = char_at(end)
                    if not (c.isalnum() or c == "_"):
                        break
                    identifier.append(c)
                    end += 1
                prefix = "".join(identifier)
                if char_at(end) == '"' and prefix in _RAW_STRING_PREFIXES:
                    # No splices in a raw string: `)delim"` must be literal.
                    opening = _RAW_DELIMITER_RE.match(source, end + 1)
                    if opening is None:  # malformed; lex it as a plain string
                        closed = literal_end(end + 1, '"')
                        i = end + 1 if closed is None else closed
                    else:
                        delimiter = opening.group()[:-1]
                        close = source.find(f'){delimiter}"', opening.end())
                        i = n if close == -1 else close + len(delimiter) + 2
                else:
                    # A prefix (L, u8, ...) is followed by its literal, which
                    # the next pass through the loop reads.
                    i = end
            elif ch.isdigit() or (ch == "." and char_at(after).isdigit()):
                # A pp-number: digits, letters, `_`, `.`, an exponent sign,
                # and `'` digit separators.
                end = after
                previous = ch
                while True:
                    end = skip_splices(end)
                    c = char_at(end)
                    if c in ("+", "-") and previous in "eEpP":
                        pass
                    elif c == "'":
                        following = skip_splices(end + 1)
                        if not (char_at(following).isalnum() or char_at(following) == "_"):
                            break
                    elif not (c.isalnum() or c in ("_", ".")):
                        break
                    previous = c
                    end += 1
                i = end
            elif ch in ('"', "'"):
                closed = literal_end(i + 1, ch)
                i = i + 1 if closed is None else closed
            else:
                i += 1
    return tuple(comments)


# The `/*` and `*` decoration a line of a `/* */` comment may start with, and
# the `*/` and white space it may end with.
_BLOCK_LINE_START_RE = re.compile(r"[ \t]*(?:/\*+)?[ \t*]*")
_BLOCK_LINE_END_RE = re.compile(r"\s*(?:\*+/)?\s*$")


def read_directives(
    comments: Sequence[Comment], family: DirectiveFamily
) -> list[Directive]:
    """Every directive of `family` among a file's `comments`, in order.

    A directive is a `//` comment that comes before the first token of code
    (after blank lines and other comments, `/* */` ones included) and starts
    with `family.pattern`; its text is the rest of the comment, stripped.
    Raises DirectiveError for a comment that reads like a directive but is
    not one: a `//` comment at the top of the file that matches
    `family.near_miss` (a misspelling), a `//` comment after code that
    starts with `family.misplaced`, or a line of a `/* */` comment that does
    (after the `/*` and any `*` decoration; so `/* // EXPECT-ERROR: x */`,
    a commented-out directive, is left alone).
    """
    directives = []
    for comment in comments:
        if comment.block:
            for offset, line in enumerate(comment.text.split("\n")):
                start = _BLOCK_LINE_START_RE.match(line).end()
                if family.misplaced.match(line, start):
                    directive = _BLOCK_LINE_END_RE.sub("", line[start:])
                    raise DirectiveError(
                        f"line {comment.line + offset}: directive "
                        f"{directive!r} in a `/* */` comment; "
                        f"directives are `//` comments: write {family.usage}"
                    )
            continue
        match = family.pattern.match(comment.text)
        if comment.before_code:
            if match:
                directives.append(
                    Directive(
                        comment.line,
                        match.group("name"),
                        comment.text[match.end() :].strip(),
                    )
                )
            elif family.near_miss.match(comment.text):
                raise DirectiveError(
                    f"line {comment.line}: malformed directive "
                    f"{comment.text!r}; write {family.usage}"
                )
        elif family.misplaced.match(
            comment.text, _COMMENT_START_RE.match(comment.text).end()
        ):
            raise DirectiveError(
                f"line {comment.line}: directive {comment.text!r} comes after "
                "code; directives go in the comments at the top of the file, "
                "before any code"
            )
    return directives


def _near_miss(first: str, second: str) -> re.Pattern[str]:
    """A `//` comment (including `///` and `//!`) that reads like a directive
    named by the words `first` and `second` (upper-case regexes): the two
    joined by `-` or `_`, in any case, with any suffix (`S`, `_GCC`,
    `-CLNAG`), then a colon; that in upper case without the colon; or the
    two spaced apart, in upper case, then a colon. So prose that only uses
    the words (`// test_flags.cpp: ...`, `// Expected error: none`,
    `// expect_error() ...`) is left alone."""
    joined = rf"{first}[-_]{second}"
    return re.compile(
        rf"//[/!]*\s*(?:(?i:{joined})[-\w]*\s*:|{joined}[-\w]*(?:\s|$)"
        rf"|{first}\s+{second}\s*:)"
    )


# `// EXPECT-ERROR: <text>`: an error a compile-fail test must fail with
# (see compile_fail_check).
EXPECT_ERROR = DirectiveFamily.with_compiler_variants(
    "EXPECT-ERROR", "text", near_miss=_near_miss("EXPECT(?:ED)?", "ERRORS?")
)
# `// TEST-FLAGS: <flags>`: extra flags to compile the test with (e.g. -O2
# for a benchmark). A misspelt one is an error rather than flags silently
# dropped.
TEST_FLAGS = DirectiveFamily.with_compiler_variants(
    "TEST-FLAGS", "flags", near_miss=_near_miss("TEST", "FLAGS?")
)


class TestDirectives(NamedTuple):
    """What a test declares for one compiler."""

    # The flags of its `// TEST-FLAGS:` directives, then those of its
    # `// TEST-FLAGS-<COMPILER>:` ones, each in file order.
    flags: tuple[str, ...]
    # The <text> of each `// EXPECT-ERROR:` / `-<COMPILER>:` directive; ()
    # for a test that must compile.
    expected_errors: tuple[str, ...]


def read_test_directives(
    comments: Sequence[Comment], compiler: str, must_fail: bool
) -> TestDirectives:
    """The directives of a test (`comments` from read_comments) that apply
    to `compiler`.

    A test that must fail to compile (`must_fail`, a static_fail/ test) must
    have at least one non-empty EXPECT-ERROR directive for `compiler`; any
    other test must have none. A long list of flags may be split over
    several TEST-FLAGS directives. Raises DirectiveError if a directive is
    malformed, misplaced or empty, its flags don't parse, or the rules above
    are broken; the directives for every compiler are checked, not only
    those for `compiler`, so a test is valid for all compilers or for none.
    """
    flag_directives = read_directives(comments, TEST_FLAGS)
    parsed: dict[Directive, list[str]] = {}
    for directive in flag_directives:
        try:
            parsed[directive] = shlex.split(directive.text)
        except ValueError as error:
            raise DirectiveError(f"line {directive.line}: {error}") from None
    # The shared directives' flags first, so a compiler's own can override
    # them.
    flags = [
        flag
        for name in ("TEST-FLAGS", f"TEST-FLAGS-{compiler.upper()}")
        for directive in flag_directives
        if directive.name == name
        for flag in parsed[directive]
    ]

    error_directives = read_directives(comments, EXPECT_ERROR)
    if not must_fail:
        if error_directives:
            stray = error_directives[0]
            raise DirectiveError(
                f"line {stray.line}: `// {stray.name}:` in a test that must "
                "compile; compile-fail tests go in a static_fail/ directory"
            )
        return TestDirectives(tuple(flags), ())
    for directive in error_directives:
        if not directive.text:
            raise DirectiveError(
                f"line {directive.line}: empty `// {directive.name}:` directive"
            )
    expected = for_compiler(error_directives, "EXPECT-ERROR", compiler)
    if not expected:
        raise DirectiveError(
            f"no `// EXPECT-ERROR: <text>` directive for {compiler}; a "
            "compile-fail test must say which error it expects"
        )
    return TestDirectives(tuple(flags), tuple(d.text for d in expected))


def load_test_directives(path: Path, compiler: str, must_fail: bool) -> TestDirectives:
    """read_test_directives for the test at `path`.

    Raises DirectiveError if it can't be read, too.
    """
    return read_test_directives(read_comments(path), compiler, must_fail)
