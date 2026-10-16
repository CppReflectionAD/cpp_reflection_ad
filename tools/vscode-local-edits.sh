#!/bin/sh
# Keep personal edits to the shared .vscode/ files out of `git status`, or
# (with --undo) track them again.
#
# .git/info/exclude can't do this: it only hides untracked files. This marks
# the files skip-worktree instead, so git leaves your copies alone. If a pull
# changes one of them, git stops with "would be overwritten"; run --undo,
# stash or drop your edits, pull, then run this again.
set -eu

cd "$(git rev-parse --show-toplevel)"
files=".vscode/settings.json .vscode/launch.json .vscode/extensions.json"

case "${1-}" in
    "")     flag=--skip-worktree ;;
    --undo) flag=--no-skip-worktree ;;
    *)      echo "usage: $0 [--undo]" >&2; exit 2 ;;
esac

# shellcheck disable=SC2086 # the file names have no spaces
git update-index $flag $files
git ls-files -v $files
