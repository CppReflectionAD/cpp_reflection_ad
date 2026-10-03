"""Read `// NAME: <text>` test directives from the top of a C++ source file.

A test declares things about itself (the error it must fail with, extra
compile flags) in `//` comments that come before its first line of code:

    // EXPECT-ERROR: ad::inverse requires an explicit inverse plan
    // TEST-FLAGS: -O2
    #include "..."

This is the single implementation of that placement rule, shared by
run_tests.py (`// TEST-FLAGS:`) and compile_fail_check.py (`// EXPECT-ERROR:`).
A `//` comment anywhere in the file that reads like a directive of a family
but is misspelt, or that comes after the first code, is rejected rather than
silently ignored. Comments are found with a small C++ lexer (see
_lex_comments), so text inside string literals, raw strings and `/* */`
comments is never mistaken for a directive.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


class DirectiveError(ValueError):
    """A test's directives are missing or malformed."""


class Directive(NamedTuple):
    line: int
    name: str  # e.g. "EXPECT-ERROR", "TEST-FLAGS-CLANG"
    text: str  # the rest of the comment, stripped


@dataclass(frozen=True)
class DirectiveFamily:
    """The directives a reader looks for, and what reads like a mistake.

    `pattern` matches the `// NAME:` start of a well-formed directive, with
    the name in group "name"; `near_miss` matches the start of a `//` comment
    that reads like one (it should also match every well-formed one). `usage`
    is how a well-formed directive is written, for error messages.
    """

    pattern: re.Pattern[str]
    near_miss: re.Pattern[str]
    usage: str


class _Comment(NamedTuple):
    line: int  # where it starts
    text: str  # from `//` to the end of the line, line splices removed
    before_code: bool  # whether it comes before the first token of code


# A backslash-newline, which joins two lines before comments and literals are
# recognized (compilers allow whitespace between the two).
_SPLICE_RE = re.compile(r"\\[ \t]*\r?\n")
_RAW_STRING_PREFIXES = frozenset({"R", "LR", "uR", "UR", "u8R"})
_RAW_DELIMITER_RE = re.compile(r'[^\s()\\"]{0,16}\(')


def read_source(path: Path) -> str:
    """The text of a test; a UTF-8 byte order mark is dropped.

    Raises OSError if the file can't be read.
    """
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _lex_comments(source: str) -> list[_Comment]:
    """The `//` comments in C++ `source`, in order.

    A state machine over the characters of the file: code, `//` comment,
    `/* */` comment, string or character literal, raw string. It applies
    line splices (backslash-newline) everywhere except inside raw strings,
    reads `'` inside a number (`1'000`, `0x1'FF`) as a digit separator, ends
    an unterminated string or character literal at the end of its line (as
    in `#error don't` or prose inside `#if 0`), and needs the exact
    R/LR/uR/UR/u8R prefix for a raw string. It does not otherwise tokenize
    or preprocess: a `//` inside `#include <...>` would read as a comment.
    """
    if source.startswith("\ufeff"):
        source = source[1:]
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

    def literal_end(i: int, quote: str) -> int:
        # i is just past the opening quote.
        while True:
            i = skip_splices(i)
            ch = char_at(i)
            if ch in ("", "\n"):
                return i
            if ch == quote:
                return i + 1
            i = skip_splices(i + 1) + 1 if ch == "\\" else i + 1

    comments: list[_Comment] = []
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
            comments.append(_Comment(line_of(i), text, not seen_code))
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
                        i = literal_end(end + 1, '"')
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
                i = literal_end(i + 1, ch)
            else:
                i += 1
    return comments


def read_directives(source: str, family: DirectiveFamily) -> list[Directive]:
    """Every directive of `family` in C++ `source`, in order.

    A directive is a `//` comment that comes before the first token of code
    (after blank lines and other comments, `/* */` ones included) and starts
    with `family.pattern`; its text is the rest of the comment, stripped.
    Raises DirectiveError for a `//` comment that matches `family.near_miss`
    but is not such a directive: a misspelling, or a directive after code.
    """
    directives = []
    for comment in _lex_comments(source):
        match = family.pattern.match(comment.text)
        if comment.before_code and match:
            directives.append(
                Directive(
                    comment.line,
                    match.group("name"),
                    comment.text[match.end() :].strip(),
                )
            )
        elif family.near_miss.match(comment.text):
            if comment.before_code:
                raise DirectiveError(
                    f"line {comment.line}: malformed directive "
                    f"{comment.text!r}; write `{family.usage}`"
                )
            raise DirectiveError(
                f"line {comment.line}: directive {comment.text!r} comes after "
                f"code; put `{family.usage}` in the comments at the top of the "
                "file, before any code"
            )
    return directives

