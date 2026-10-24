#!/usr/bin/env bash
# Checks that every submodule's pin is on the branch its .gitmodules entry
# names (#162 step 9): the pin must be an ancestor of that branch's tip in the
# repository at the entry's URL. The forks' p4395 branches can't be deleted or
# force-pushed, so a pin on one stays fetchable. A pin that is only on some
# other branch, or only in a pull request, can disappear.
#
# It asks GitHub's compare API, so it clones nothing and needs gh with a token
# (GH_TOKEN). The submodules come from .gitmodules and the pins from the index,
# so a renamed or added submodule, or a staged bump, needs no change here.
#
# Usage: check-submodule-pins.sh (from anywhere in the repository)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
failed=0
err="$(mktemp)"
trap 'rm -f "$err"' EXIT
fail() { echo "check-submodule-pins: $*" >&2; failed=1; }

while read -r key path; do
  name="${key#submodule.}"
  name="${name%.path}"
  url="$(git config -f .gitmodules "submodule.$name.url")"
  branch="$(git config -f .gitmodules "submodule.$name.branch" || true)"
  pin="$(git rev-parse ":$path")"
  if [[ -z $branch ]]; then
    fail "$name: .gitmodules names no branch, so there is nothing to check $pin against"
    continue
  fi
  if [[ ! $url =~ ^https://github\.com/([^/]+)/([^/]+)$ ]]; then
    fail "$name: $url is not an https://github.com/<owner>/<repo> URL"
    continue
  fi
  repo="${BASH_REMATCH[1]}/${BASH_REMATCH[2]%.git}"

  # status is how the branch compares with the pin: "ahead" or "identical"
  # means the pin is an ancestor of the branch's tip.
  if ! status="$(gh api "repos/$repo/compare/$pin...$branch" --jq .status 2>"$err")"; then
    fail "$name: GitHub can't compare $pin with $branch in $repo ($(tail -n 1 "$err"))." \
      "Is the pin pushed there, and does the branch exist?"
    continue
  fi
  case "$status" in
    ahead | identical)
      echo "ok: $name pin $pin is on $repo $branch" ;;
    behind)
      fail "$name: pin $pin is ahead of $repo $branch. Merge it into $branch first." ;;
    *)
      fail "$name: pin $pin is not on $repo $branch ($status). Pin a commit on" \
        "$branch, or merge the pinned commit into it with a merge commit." ;;
  esac
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')

exit "$failed"
