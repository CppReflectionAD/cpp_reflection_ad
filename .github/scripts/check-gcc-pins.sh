#!/usr/bin/env bash
# Checks that the gcc-darwin-reflect pin has every reflection commit in the
# gcc-mirror pin (#94 C11). The darwin fork's p4395 branch cherry-picks them,
# so this is `git cherry`, which matches commits by content and skips merge
# commits, not an ancestry check. It passes when every line is `-`. See
# "Bumping a compiler pin" in CONTRIBUTING.md for how to fix a failure.
#
# It fetches both pins from the URLs in .gitmodules into a scratch repository,
# so it needs neither submodule checked out. The pins are read from the index,
# so a staged bump is checked before it's committed.
#
# Usage: check-gcc-pins.sh (from anywhere in the repository)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
darwin="$(git rev-parse :gcc-darwin-reflect)"
mirror="$(git rev-parse :gcc-mirror)"
darwin_url="$(git config -f .gitmodules submodule.gcc-darwin-reflect.url)"
mirror_url="$(git config -f .gitmodules submodule.gcc-mirror.url)"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export GIT_DIR="$work"
git init -q --bare
# Otherwise a fetch can leave a background gc rewriting the shallow file that
# the next fetch reads ("shallow file has changed since we read it").
git config gc.auto 0
git config maintenance.auto false
fail() { echo "check-gcc-pins: $*" >&2; exit 1; }

# GCC's history is too big to fetch whole, so fetch both pins shallow and
# deepen until the range git cherry compares is complete: the pins have a merge
# base, and no shallow boundary is inside either side of the range. A boundary
# there hides its parents, so git cherry could skip commits behind it, or reach
# commits below the base through a merge and list them as missing.
complete() {
  base="$(git merge-base "$darwin" "$mirror")" || return 1
  [[ -s $work/shallow ]] || return 0
  ! git rev-list "^$base" "$darwin" "$mirror" | grep -qxFf "$work/shallow"
}
depth=256
while :; do
  echo "Fetching both pins at depth $depth"
  git fetch -q --no-tags --depth="$depth" "$darwin_url" "$darwin"
  git fetch -q --no-tags --depth="$depth" "$mirror_url" "$mirror"
  complete && break
  ((depth < 16384)) || fail "no complete range between the pins within $depth commits"
  depth=$((depth * 2))
done

echo "Merge base: $(git log -1 --format='%h %cs %s' "$base")"
cherry="$(git cherry -v "$darwin" "$mirror" "$base")"
echo "$cherry"
if grep -q '^+' <<<"$cherry"; then
  fail "the gcc-darwin-reflect pin $darwin lacks the gcc-mirror commits" \
    "marked + above. Bring them over to p4395 as described in" \
    "\"Bumping a compiler pin\" in CONTRIBUTING.md."
fi
echo "ok: the gcc-darwin-reflect pin has every gcc-mirror commit since their merge base"
