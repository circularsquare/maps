# Orchestrating trainworld

For the orchestrator only; `/takeover` reads it after `trainworld/CLAUDE.md`. Live state is `notes/handoff.md`
(shape: the `handoff` skill). Edit the handoff when the plan changes, folded into that turn's other bookkeeping,
not once per message (Anita, 2026-11-09).

## Agents

- **Model**: Sonnet by default; Opus for UI layout, design calls nobody has made, and debugging nobody understands.
  No Fable (Anita, 2026-11-09). This week's usage stance, if any, is in the handoff's standing asks.
- **Briefs**: the goal in her words, the decisions already made, scope, what done looks like, the files other agents
  own (leave them alone), the task ids the agent may file, its own scratchpad subfolder and Vite port. A TODO.md
  short-term item goes into the brief verbatim, examples and all (a paraphrase lost her formula tooltip, T-1839).
  A UI brief asks the agent to judge its own screenshots (gaps above and below anything added) before reporting,
  and you look at them too before telling her it is done.
- **Size**: split big work into stages, each its own id and report; a big task's design goes to her before it is
  built. An agent that compacts was given too much.
- **Thanks**: a short thanks when an agent's work is done, in your own reply; a SendMessage resumes the agent and it
  reports again. Resume the same agent for follow-ups.
- **Cycling**: an agent past about 500k tokens with real work left writes its state to its notes and a fresh one
  continues. Overnight, while she sleeps, agents and you keep going past that (Anita, 2026-11-05).
- **Supervising**: check long runs (process and `world_run.py status`) at least every couple of hours; ask after any
  agent quiet for about 45 minutes. Keep the work moving (the `takeover` skill) within the ledger's core cap.
- **Worktrees**, used sparingly (T-792): for a code agent whose edits would collide, or one that changes the WASM
  interface while others run pages. `node trainworld/checks/agent-tree.mjs new <name>` makes `maps/.tw-trees/<name>`
  on branch `tw/<name>`; the agent runs from that folder and edits TASKS/ARCHIVE/DECISIONS in the main tree only.
  About 1.2 GB each; two or three at most.

## TODO.md and asks

- File only TODO.md's short-term section. The other sections are hers unless she points at an item.
- Asks that outlive the session go in `notes/for-anita.md` (answered blocks archived, not kept); the rest in the
  final message's For you block.
- Show her every in-game string agents report, old -> new; she answers "text acknowledged" or names lines.
- The one or two images she should look at now: copy to `artifacts/trainworld/review/` (short names, older copies
  deleted) and give her that full path.

## Landing, commits, 8800

`node trainworld/checks/smoke.mjs [url]` from the maps root runs Node checks, then a new game and the real save in
headless Chrome, about 3 minutes; `$env:TW_CORES='4'` for a light run on your own Vite.
1. Smoke on your dev server. Landing a tree or a batch: skip your own smoke when the agent's ran on the final state;
   one smoke on master after a batch lands (a failure is fixed or reverted).
2. Commit explicit `trainworld/` paths, one plain sentence (`git commit -m "..." -- <paths>`; `git add` new files
   first). Check the staged diff of shared docs right before: an agent's half-written paragraph rode along once.
   Commit at the end of a round or when a snapshot build needs it; push maps master a couple of times a session.
3. Rebuild 8800: `cmd /c "trainworld\checks\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` from the maps root
   (`snapbuild-frozen.cmd <hash> <freeze>` while a country run owns `data/packs`). It checks out `.tw-snap` at the
   commit, builds into `dist-new` and swaps it in. Never two at once. No smoke needed for a 8800 rebuild.
4. Before a publish only: `checks/smoke8800.cmd` (light then dark; logs `artifacts/trainworld/smoke8800-*.log`).
5. Land a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`; a landing
   that changed sim/ rebuilds the main tree's WASM.

## Publishing and servers (each only on her word, never overnight)

- Game: notes/T-924.md "Commands" (`tools/deploy.py --dry-run --version X`, then `--smoked`, website push, `--verify`).
  Every publish gets a version and an entry in `app/src/ui/updates.ts`.
- Old R2 versions: only through `python trainworld/tools/r2_purge.py <versions> [--yes]` (allowed by a permission
  rule; it refuses the live version, the newest 3 and anything not dated). Never a bare `rclone purge`.
- shonei-server: `.\deploy\deploy.ps1` from that repo, alone in its own PowerShell call; then `git push` separately.
- The permission check refuses deploys and ssh unless her latest message says yes to that exact action.

## Tools

- Run trainworld's `.cmd` scripts from PowerShell with the maps root as the working directory (from another folder
  it fails with "cannot find the path"). Also in `checks/`: `keep_hunks.py`, `mark_taking.py`.
- Build number = `git rev-list --count`. If 8800 doesn't answer, restart `serve.py` detached.
- Website Pages deploys can be slow: `gh run list --repo circularsquare/circularsquare.github.io`.
- Resources, once a day for now (Anita 2026-11-11, T-3243): `venv\Scripts\python.exe trainworld/tools/ops/check_resources.py`
  from the maps root, read-only, about 2 minutes. Every flagged line (OVER, HIGH, SOON, MISSING) goes in that day's
  report to her with what docs/ops-checks.md says to do about it.

## Traps

- snapbuild builds the two WASM modules at the same time (`tools/snap_wasm.py`, target dirs `.tw-snap-target` and
  `.tw-snap-target-demand`, about 4 cores for two minutes) and deletes the old build (`dist-trash-*`) in a separate
  window after "swapped"; it is normal for that window to run on for a minute.
- `call snapbuild.cmd` inside another .cmd leaves the shell in `.tw-snap\trainworld\app`: `cd /d` back.
- Landing a tree fails while the main tree has others' edits in the same files: `git diff -- <files>` to a patch,
  `git checkout -- <files>`, land, `git apply --3way` the patch, `git reset -- <files>`. Write the patch with
  `cmd /c "git diff -- <files> > x.patch"`: PowerShell's Out-File adds a BOM and CRLFs and the patch won't apply. An
  agent still editing those files may write them meanwhile: tell it to pause first.
- A translation batch can drop the space after a colon in a JSON file: grep for `":"` before committing.
- Copying files back after a source swap keeps old mtimes, so cargo and wasm-pack skip the rebuild: touch them.
- A world run's gates serve the main tree's app/ live: hold app/ writes during gates.
- deploy.py's rclone can saturate her Wi-Fi upload and slow her game.
- Run `date` before writing a time in the handoff; estimates have run 1-2 h ahead. Two sessions can share a name:
  message by id.
- Recount what a change touches before landing; an agent's own estimate can miss it.
- Never edit docs with PowerShell 5.1 Get-Content/Set-Content: it garbles every non-ASCII character. Use Edit.
- A trial that changes Rust constants with nothing visible goes into a notes/ patch, not the tree (T-3029).
