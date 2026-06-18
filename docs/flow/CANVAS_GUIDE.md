# Canvas Flow Guide

Conventions for generating Obsidian `.canvas` flowcharts of this codebase.
Hand this file to Claude together with the target module and say:
*"Create/refresh `docs/flow/<name>.canvas` for <module/flow>, following `docs/flow/CANVAS_GUIDE.md`."*

This is the portable, engine-agnostic artifact. It travels **with the code** so any
Claude task — local or on the WSL/SSH core — produces consistent diagrams.

## Why we keep canvases (not just code)
- **Review the flow, not the diff.** A flowchart is faster to audit than a patch.
- **Context recall.** Claude re-reads the canvas to regain the whole-system picture before a change.
- **Architecture-smell detector.** Many bugs aren't logic errors — the canvas reveals a step that *shouldn't exist in this flow*. Fix the flow, don't patch the patch.

## Two canvases per project (required)
Every project keeps exactly two canonical canvases in `docs/flow/`:

1. **`architecture.canvas` — current architecture.** Components, process/boundary
   swimlanes, data stores, external services, and how they connect. The "what it is
   right now" structure map. Refresh whenever the structure changes.
2. **`status.canvas` — goals · approach · task progress.** A board:
   🎯 Goals · 🧭 Approach/method · ✅ Done · 🚧 In progress · ⏳ Next, plus a ⚠️ Attention lane.
   **Altitude = human/PM level: what & how, not the code.** OMIT code identifiers
   (`file:line`, function/class names), commit hashes, and git/GitHub operations
   (branch names, "uncommitted/untracked", "commit/merge") — those belong in
   `architecture.canvas` and the code. Still *derive* the truth from real signals,
   but phrase cards as **capabilities and progress**, not mechanics:
   - 🎯 Goals ← README roadmap / project intent
   - 🧭 Approach ← the principles/strategy guiding the work (e.g. "offline-first", "heavy work split to the core")
   - ✅ Done ← shipped capabilities (derive from history; state the feature, not the commit)
   - 🚧 In progress ← what's actively being built (derive from branch/working state; describe the task)
   - ⏳ Next ← unfinished goals, backlog
   - ⚠️ Attention ← blockers, regressions, things to verify — in plain language

Optional: per-flow deep-dives (e.g. `voice-pipeline.canvas`, `memory-subsystem.canvas`)
when one flow needs more detail than the architecture canvas can hold.

- Commit canvases with the code change they describe (same PR).

## JSON Canvas format (the whole spec you need)
A `.canvas` file is JSON with `nodes` and `edges`.

```json
{
  "nodes": [
    {"id":"a","type":"text","x":0,"y":0,"width":260,"height":120,"color":"5","text":"**Title**\n`file.py:42`"},
    {"id":"g","type":"group","x":-40,"y":-40,"width":700,"height":300,"label":"Lane name","color":"5"},
    {"id":"r","type":"link","x":0,"y":-200,"width":300,"height":90,"url":"https://github.com/..."}
  ],
  "edges": [
    {"id":"e1","fromNode":"a","fromSide":"right","toNode":"b","toSide":"left","label":"calls"}
  ]
}
```
- Node `type`: `text` (markdown), `group` (labeled swimlane/background), `link` (url), `file` (vault path).
- Sides: `top|right|bottom|left`. Coordinates: `x`→right, `y`→down.

## Node conventions
- **One concept per node.** A node = a module, function, or pipeline step.
- **Always cite `file:line`** in the node text (e.g. `` `main.py:639` ``). The canvas doubles as a clickable code index.
- **Use `text` + `link` nodes, NOT `file` nodes.** Source code is not in the vault, so `file` nodes would dangle. `text` nodes (self-contained) + a `link` node to the GitHub file keep the canvas portable to the vault and to remote machines.
- First line = **bold title**; following lines = the responsibility / key calls.

## Layout rules (minimize the manual nudge)
- **Grid.** Step x by ~320, y by ~150. Standard node 260×120; widen to 340–460 when text is long.
- **Swimlanes = `group` nodes.** One group per process/boundary (here: *Windows desktop* vs *WSL core*). Place step nodes inside; the group is just a labeled background.
- **Direction is consistent.** Capture→send left→right on top; the remote hop is a vertical edge; the return leg flows back below. A little edge-crossing is fine — nudge in Obsidian.

## Color legend (JSON Canvas presets "1"–"6")
| color | preset | use |
|---|---|---|
| `5` cyan | Windows desktop / client | the hardware-bound side |
| `6` purple | WSL core / server | LLM + memory process |
| `4` green | memory / persistence | stores, vector search, prompt build |
| `2` orange | LLM call | Ollama generate |
| `1` red | error / interrupt | barge-in, failure paths |

## Edge conventions
- Label every edge with the **call or event** (`core_client.chat(text)`, `② reply {id,text}`).
- Number a request/response round trip ① ② so the direction reads at a glance.
- Color a cross-boundary edge to match the boundary it crosses (e.g. the WS hop = purple).

## When to (re)generate
- After a bug fix or a feature that changes a flow.
- After writing/updating module docs.
- Scope regeneration to the **changed flow only** — keep canvases living, not stale snapshots.

## Remote / SSH note
Generating a canvas is just writing this JSON file — no Obsidian API needed on the
generating machine. The core runs in WSL (`~/workspace/Jarvis/core`); a Claude task
there can write `docs/flow/*.canvas` with plain file writes. The files reach this
machine via **git, one-way** (see below) — the Obsidian REST API path was dropped
(OneDrive dehydrates vault files). This guide + these conventions are what later get
extracted into a reusable plugin/skill.

## Syncing remote → local (git, one-way)
The remote generates canvases; this machine only renders them. Transport = git, one
direction, overwrite (no local layout edits to preserve). Works whether the remote is
WSL or a cloud SSH host — both push to the shared remote and this machine pulls.

- **Remote** (after a regen): `bash docs/flow/push-canvas.sh` — commits *only*
  `docs/flow/` and pushes on the current branch (your other work is untouched).
- **This machine**: `pwsh docs/flow/pull-canvas.ps1` — fetches and overwrites *only*
  `docs/flow/`; everything else is left alone. Schedule it every few minutes for
  hands-off refresh.
- **WSL shortcut**: if the remote is WSL on this same box, you can skip git and copy
  straight from `\\wsl$\<distro>\home\<you>\workspace\Jarvis\docs\flow\`.
