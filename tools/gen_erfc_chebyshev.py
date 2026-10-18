#!/usr/bin/env python3
"""Generate cx_erfc.hpp's Chebyshev table, or check it.

    uv run --group scripts tools/gen_erfc_chebyshev.py          # print it
    uv run --group scripts tools/gen_erfc_chebyshev.py --check  # compare

For x >= 0.5, cx::erfc computes erfc(x) = exp(-x^2) * g(x), with
g(x) = exp(x^2) * erfc(x) smooth and slowly varying. g is fitted on the
dyadic intervals [2^(i-1), 2^i], i = 0..5 (so [0.5, 1] up to [16, 32]), by
interpolation at Chebyshev nodes in 300-bit arithmetic. Each interval's
degree is the least that keeps the relative error below 2^-60 on 401 points,
plus one. The coefficients are printed with repr, so each is the double
nearest the exact one.

Paste the output over the table in cx_erfc.hpp. --check regenerates the table
and fails unless cx_erfc.hpp holds the same numbers; pre-commit runs it. It
compares numbers rather than text, since clang-format reflows the table.
"""

import re
import sys
from pathlib import Path

import mpmath

mpmath.mp.prec = 300

HEADER = Path(__file__).resolve().parent.parent / "include/cx_std/cx_erfc.hpp"


def g(x):
    return mpmath.exp(x * x) * mpmath.erfc(x)


def chebyshev(f, a, b, n):
    """Coefficients of the degree-n interpolant at Chebyshev nodes, c[0]
    halved, in the variable u = (x - (a + b) / 2) / ((b - a) / 2)."""
    N = n + 1
    theta = [mpmath.pi * (j + mpmath.mpf(1) / 2) / N for j in range(N)]
    fs = [f((b - a) / 2 * mpmath.cos(t) + (a + b) / 2) for t in theta]
    c = [
        2 * sum(fs[j] * mpmath.cos(k * theta[j]) for j in range(N)) / N
        for k in range(N)
    ]
    c[0] /= 2
    return c


def clenshaw(c, u):
    b1 = b2 = mpmath.mpf(0)
    for k in range(len(c) - 1, 0, -1):
        b1, b2 = c[k] + 2 * u * b1 - b2, b1
    return c[0] + u * b1 - b2


def max_rel_error(c, a, b):
    worst = mpmath.mpf(0)
    for i in range(401):
        u = mpmath.mpf(2 * i) / 400 - 1
        x = (b - a) / 2 * u + (a + b) / 2
        worst = max(worst, abs(clenshaw(c, u) / g(x) - 1))
    return worst


def table():
    pieces = []
    for i in range(6):
        a, b = mpmath.mpf(2) ** (i - 1), mpmath.mpf(2) ** i
        n = 8
        while max_rel_error(chebyshev(g, a, b, n), a, b) >= mpmath.mpf(2) ** -60:
            n += 1
        pieces.append(chebyshev(g, a, b, n + 1))
    width = max(len(c) for c in pieces)
    lines = [
        f"inline constexpr std::size_t kErfcPieces = {len(pieces)};",
        f"inline constexpr std::size_t kErfcMaxTerms = {width};",
        "inline constexpr std::size_t erfc_terms[kErfcPieces] = {"
        + ", ".join(str(len(c)) for c in pieces)
        + "};",
        "inline constexpr double erfc_chebyshev[kErfcPieces][kErfcMaxTerms] = {",
    ]
    for i, c in enumerate(pieces):
        lines.append(f"    // [{float(2.0 ** (i - 1))}, {float(2.0**i)}]")
        lines.append("    {" + ", ".join(repr(float(v)) for v in c) + "},")
    lines.append("};")
    return "\n".join(lines) + "\n"


def numbers(text):
    """The numbers in a table, in order, without comments or array bounds."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"\[kErfc\w+\]", "", text)
    return [float(t) for t in re.findall(r"[-+]?\d[\d.]*(?:e[-+]?\d+)?", text)]


def check(generated):
    text = HEADER.read_text()
    start = text.index("inline constexpr std::size_t kErfcPieces")
    end = text.index("};", text.index("erfc_chebyshev[kErfcPieces]")) + 2
    if numbers(text[start:end]) != numbers(generated):
        print(
            f"{HEADER.name}: the Chebyshev table differs from what "
            f"{Path(__file__).name} generates. Paste its output over the table.",
            file=sys.stderr,
        )
        return 1
    return 0


def main():
    generated = table()
    if "--check" in sys.argv[1:]:
        return check(generated)
    print(generated, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
