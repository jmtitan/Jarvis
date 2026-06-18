#!/usr/bin/env bash
# Push freshly-generated flow canvases to the shared git remote.
# Run on the machine where the Claude task generated them (WSL core or cloud host).
#
# Path-scoped: commits ONLY docs/flow/ and pushes on the current branch, so your
# other staged/unstaged work is left completely alone.
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
cd "$repo"

git add -- docs/flow
if git diff --cached --quiet -- docs/flow; then
  echo "No canvas changes to push."
  exit 0
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
git commit -m "docs(flow): refresh canvases" -- docs/flow
git push origin "$branch"
echo "Pushed canvases on '$branch'."
