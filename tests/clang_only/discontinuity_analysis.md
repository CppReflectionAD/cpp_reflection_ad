# Discontinuity analysis

`discontinuity_analysis.hpp` finds the points where a function jumps along one
of its arguments, and by how much. Its header comment describes the API, the
method and the errors it raises. This file lists what it knowingly gets wrong
or refuses.

## Known limitations

- **A removable singularity on a branch is reported as an error**, because the
  function is evaluated at the root rather than as a limit. For example,
  `s > 0 ? s*log(s) : 0` has a jump of 0 at 0, but the right side evaluates
  `0 * −inf`.
- **A condition that varies continuously but not affinely is a compile error,
  even when it has no jump.** `exp(s) - 2 ? 1 : 0` and `sqrt(s) ? 1 : 0` change
  only at a single point, so they have no jump, but they fail the affine
  `static_assert`. This is the same conservative stance as for `s*s > k`: the
  analysis can't place the point, so it refuses rather than assumes. Telling
  these apart from `max(s - k, 0) ? 1 : 0`, which does jump, would need more
  than affine solving.
- **A jump within the rounding error of evaluating the function at its point
  is not reported**, since it cannot be told from rounding at a kink there.
  The bound is the branch values' rounding plus their slope times how far the
  exact root can be from the point. It is 0 where those are computed exactly,
  and a few ulps of the values involved otherwise. For example,
  `s/3 > k/3 ? (s - k)*1e9 + 1e-5 : 0` at k = 100 loses the 1e-5: the gap's
  rounding puts the root within about 1.5 ulps of 100, and 1e9 times that is
  2e-5. The error bounds (midpoint-radius `Ball`s) do not know that `s/3` and
  `k/3` round alike at s = k.
- **Two real jumps within 4 ulps of each other are reported as one**, their
  sum, at one of their roots. That is the price of treating roots that close
  as one crossing written two ways. Roots more than 4 ulps apart are measured
  independently.
- **An infinite slope is an error even where the crossing never flips.**
  `s * (k - k) > 0` at k = inf is NaN for every target, but its slope is NaN
  too, so it is rejected rather than told apart from `s / k > 1` at k = 0.
- **An overflow at target 0 is an error even where the crossing never flips.**
  `(s + 1e308) + 1e308 > 0` throws, though its root (about −2e308) is not a
  double, because it cannot be told apart from `(s - k) * 1e10 > 0` at
  k = 1e300, which does jump.
- **A guard is judged per condition, not jointly.** `s > 1 && s < 0` is "may be
  true", because each operand may be. Such a dead branch is still rooted,
  which matters only if it then evaluates a pole.
- **Compile time grows with the square of the number of crossings**, because
  each point is measured by evaluating the whole function twice, in `Ball`s,
  plus the gap of each crossing in its group. With clang-p2996 at -O0, one
  consteval analysis of a sum of 40 digitals compiles in about 2.5 s, of 80 in
  about 7 s, and of 160 in about 34 s.
- **The results can differ from the function's own `<cmath>` in the last few
  bits.** exp/log/erfc/sin/cos come from `cx_std` in every entry point (sqrt is
  exact), so the consteval and `_rt` entry points agree exactly with each other
  instead. That holds under this repo's flags (no `-march`/`-mfma`): with FMA
  enabled, clang's default `-ffp-contract=on` could fuse `cx_std`'s
  multiply-adds at run time, and `_rt` could then differ by an ulp.
- **GCC cannot build this header** (it has no `?:` reflection), so its tests
  are clang-only, and there are no `tests/static_fail` cases for its compile
  errors: `static_fail` also runs on GCC, where they would pass for the wrong
  reason.
