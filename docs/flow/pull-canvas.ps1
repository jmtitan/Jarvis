# Pull the latest flow canvases from the shared git remote onto THIS machine.
# One-way + overwrite: only docs/flow/ is touched; all other local changes are left
# alone. No local layout edits are preserved (by design - canvases are consumed as-is).
#
# Usage:  pwsh docs/flow/pull-canvas.ps1                # uses the current branch
#         pwsh docs/flow/pull-canvas.ps1 -Branch flow   # a dedicated canvas branch
param([string]$Branch = "")

$repo = (git -C $PSScriptRoot rev-parse --show-toplevel).Trim()

git -C $repo fetch origin --quiet
if (-not $Branch) { $Branch = (git -C $repo rev-parse --abbrev-ref HEAD).Trim() }

git -C $repo checkout "origin/$Branch" -- docs/flow 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "No docs/flow on origin/$Branch yet. Push it from the remote first: bash docs/flow/push-canvas.sh"
    exit 1
}

git -C $repo reset -q -- docs/flow    # unstage; leave files in place for Obsidian
Write-Host "Canvases updated from origin/$Branch  ->  $repo\docs\flow"
