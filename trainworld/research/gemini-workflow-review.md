# Fresh-Eyes Review: Workflow, Tooling, Specs & Checks (Gemini, 2026-11-24)

Read-only audit of `trainworld/` and `.claude/` from a first-time outside perspective. No code or docs were modified. Companion to `research/claude-md-review.md`.

---

## Summary

The core game architecture (`SPEC.md`, `docs/code-tour.md`, dual-WASM + native Rust, code-linked `wiki/`) is strong. The friction is concentrated in four areas:
1. **AWS security relies on a Claude-only regex hook** over an `AdministratorAccess` key.
2. **Harness overhead** (an unmatched `PostToolUse` Python hook running on every tool call, transcript-scraping hooks, and 19 live worktrees).
3. **Doc & check sprawl** (~312k words in `docs/`, 52k lines in `checks/`, concrete drift in `SPEC.md`, `interface.md`, `handoff.md`, and `TASKS.md`).
4. **Dead weight** (`pipeline/nocar.py`, 5 `BROKEN` checks, 3-copy `wiki/` in git, JS `sheetLines.ts` duplicate).

---

## 1. Security / Safety (High Priority)

### 1.1 AWS: `docs/aws.md` / `awslib.py` still point at Jean's old `AdministratorAccess` profile
- **Status (updated after checking with Anita)**: Anita is no longer using Jean's account (`[default]`, account `429583250612`) and already set up her own AWS account (`537239093801`) under profile `anita-own` with a non-admin IAM user `bench` (`arn:aws:iam::537239093801:user/bench`, confirmed denied `iam:ListAttachedUserPolicies`).
- **Remaining cleanup**:
  - Anita will remove Jean's `[default]` key from `~/.aws/credentials` (and let Jean know about the old `anita` admin key).
  - Update `trainworld/tools/aws/awslib.py` (`PROFILE`, `ACCOUNT`, `USER_ARN`, `BUCKET`) and `trainworld/docs/aws.md` when ready to switch `tools/aws/` over to `anita-own`, keeping `guard.py` as defense-in-depth for spot/region/cost rules.

---

## 2. Claude Workflow & Hook Overhead

### 2.1 Unmatched `PostToolUse` hook spawns Python on every tool call
- **Where**: `.claude/settings.json:181-191` → `.claude/hooks/orchestrator.py` (`reply-nudge`).
- **Issue**: No tool matcher is set, so every `Read`, `Edit`, `Grep`, `Glob`, `Bash`, etc. across `maps`—including inside subagents, which spawn `venv/Scripts/python.exe` on Windows only to exit at line 181—pays Python startup overhead. For the orchestrator, `waiting_message()` (`orchestrator.py:131-176`) seeks and decodes up to **3 MB** (`NUDGE_TAIL`) of transcript JSONL on every tool call.
- **Suggested fix**: Either narrow the matcher (e.g., to `Bash|Agent` or `Stop` only), or skip spawning Python for subagents / throttle via a lightweight timestamp check, or drop `reply-nudge` if the `Stop` hook / `watch.py` already covers it.

### 2.2 Cycling rule mismatch between `orchestrator.py` and `docs/orchestrator.md`
- **Where**: `.claude/hooks/orchestrator.py:105` vs `trainworld/docs/orchestrator.md:75-79`.
- **Issue**: The hook tells the orchestrator to run `next_orchestrator.py <project>` directly, omitting both the `notes/standby.txt` check and `--advisor fable`. Also `docs/orchestrator.md:38` ("no Fable unless she asks") vs line 78 (`--advisor fable`) is easy to misread.
- **Suggested fix**: Have `orchestrator.py:105` point to the `/handoff` skill (or mirror the `standby.txt` + `--advisor` wording) and clarify "no Fable as worker model; orchestrator uses `--advisor fable`".

### 2.3 Worktree accumulation & `cores.py` shadowing
- **Where**: `.tw-trees/` (17 worktrees + `.tw-snap` + main = **19 worktrees**, ~20+ GB), `trainworld/notes/handoff.md:55` (`.tw-trees/arrivefirst` junction landmine), `cores.py`.
- **Issue**:
  - Stale worktrees accumulate (`orchestrator.md:51` says ~1.2 GB each, remove when done).
  - `.tw-trees/arrivefirst` is left behind as a documented trap ("never delete it by hand").
  - Because `cores.py` is tracked in git, every worktree has its own copy; scripts calling `./cores.py` inside a worktree write to an isolated `cores.json` unless they explicitly climb out of `.tw-trees/` like `agent_cargo.py:17-22`.
- **Suggested fix**:
  - Clean up `.tw-trees/arrivefirst` safely via `agent-tree.mjs` (or unlink junctions first, then remove) and prune idle/finished worktrees.
  - Patch `cores.py` itself to resolve `cores.json` via `git rev-parse --git-common-dir` (or walk up past `.tw-trees/`), so every caller automatically shares one ledger.

### 2.4 Inconsistent locking across task/ask/changelog tools & uncommitted clutter
- **Where**: `trainworld/tools/{tasks,asks,changelog}.py`, `trainworld/notes/handoff.md:51-54`.
- **Issue**:
  - `asks.py:70-90` uses an `O_EXCL` lockfile with no stale-lock expiry; `changelog.py:52-81` expires after 30s; `tasks.py` has no lockfile (only a `PermissionError` retry on `os.replace`).
  - 15+ files are intentionally left uncommitted in `main`, causing repeated landing friction (`git diff > x.patch` / `checkout` / `apply --3way` in `docs/orchestrator.md:186-189`).
- **Suggested fix**: Share one small lockfile helper (with stale-PID/timeout expiry) across `tasks.py`, `asks.py`, and `changelog.py`. For runtime/local files that are never meant to be committed (`notes/board-live.json`, `notes/asks-log.jsonl`, etc.), `.gitignore` them instead of carrying them as permanent dirty status.

### 2.5 Minor script/settings polish
- `trainworld/tools/ops/reserve_cores.ps1:2` comment says `claude.exe stays at normal priority`, but lines 33 & 49 set it to `AboveNormal` — align comment or code.
- `.claude/settings.json:26-91` repeats ~10 Python scripts 4x each for path/quote permutations and hardcodes `C:/Users/anita/projects/maps/venv/Scripts/python.exe`. Worth keeping in mind if the repo/venv ever moves.

---

## 3. Spec & Documentation Drift / Bloat

`trainworld/docs/` is currently **32 files, 17,560 lines (~312,000 words)**—larger than the codebase—across four overlapping layers (`SPEC.md`, `docs/*.md`, `docs/demand-plain.md`, `wiki/`).

### Concrete drift to fix now (cheap)
1. **`SPEC.md:89`**: Still says *"What is covered (2026-10-22, about 800 zones)"*, whereas `docs/code-tour.md:32` says *"a little over 4,000 zones"*. Update or point to `docs/data-register.md` / registry count so `SPEC.md` stops going stale.
2. **`docs/interface.md:10`**: Still summarizes `ui-panels.md` as *"Zen Maru Gothic, ..., no transitions"*, contradicting `DECISIONS.md:187` (Nunito for UI text) and `DECISIONS.md:203` (transitions allowed).
3. **`notes/handoff.md`**: Carries an `## In flight` section despite `handoff/SKILL.md` saying live state comes from `handoff.py status`, and lists two conflicting 8800 commits in the same file (`8800 = ef99d979` on line 19 vs `809a0d16` on line 27).
4. **`TASKS.md:6,17,116`**: Header says P4 tasks live under `**Someday**` at the end and don't count as open, yet `[P4]` tasks (`T-3706`, `T-1738`) sit near the top under `## World and countries`. Add a check in `tasks.py check` or move them.

### Medium-term doc hygiene
- **Stop committing generated `wiki/html/`** (3 copies of the wiki are currently tracked: `wiki/*.md`, `wiki/read/*.md`, `wiki/html/*.html`). Keep `wiki/read/` for agent reads if helpful, and generate `wiki/html/` at build/deploy time.
- **Keep `SPEC.md` as a true summary**: Section 6 (`SPEC.md:231-318`) has grown into a dense multi-page wall of `[T-####]` citations that duplicates `docs/track-and-service.md`. Trim inline task-ID history when next editing.
- **Trim chronological history from `docs/demand-worldwide.md`**: Currently reads as a dated country log ("The first country outside the US: Canada (T-277, 2026-10-13)") rather than current rules.

---

## 4. Checks & Dead Code

1. **5 `BROKEN` checks still in `checks/`**: `checks/INDEX.md` flags `t1201-fare-rows.mjs`, `t2162-diagram-names.mjs`, `t2289-diagram-intercity.mjs`, `t2407-blocks.mjs`, `t2626-profit-estimate.mjs` as `BROKEN` (several test the removed unsheeted diagram). Move to `checks/old/` or delete.
2. **One-off task check sprawl**: `checks/` has **382+ files (~52,000 lines)**—bigger than `pipeline/`—with only 19 in `smoke.mjs` and many `t####-look.mjs` / zero-run scripts. Worth a periodic archive pass to `checks/old/`.
3. **Dead pipeline step (`pipeline/nocar.py`)**: Still runs and is documented in `docs/world-run.md:23,116,134,387` even though `sim/src/demand/api.rs` stopped reading car-free household arrays in T-918 (already noted in `docs/code-tour.md:138`). Remove from `world_run.py` and `docs/world-run.md`.
4. **JS carry-solve duplicate (`app/src/diagram/sheetLines.ts`)**: Kept in sync with Rust `sim/src/diagram.rs` solely because 3 Node checks (`t2580-sheet-water`, `t3358-river-banks`, `t3364-off-sheet-water`) call it directly (`docs/code-tour.md:135`). Pointing those 3 checks at WASM lets you delete the duplicate.
5. **Repo clutter**: Delete `trainworld/bash.exe.stackdump` and `trainworld/grep.exe.stackdump`.

---

## Suggested Split: Claude Orchestrator vs. Gemini

- **Best for the Claude orchestrator (owns live state & Claude hooks)**:
  - `.claude/settings.json` (`PostToolUse` hook fix) & `.claude/hooks/orchestrator.py`
  - Pruning `.tw-trees/` (including `.tw-trees/arrivefirst`) & cleaning up uncommitted `main` files
  - `notes/handoff.md` & `TASKS.md` P4 placement
  - Landing/committing anything into `master` while other Claude agents are active
- **Good standalone tasks for Gemini (self-contained, low collision risk)**:
  - Fixing `cores.py` to resolve the shared repo root from inside worktrees + unifying the lockfile helper in `tasks.py`/`asks.py`/`changelog.py`
  - Removing dead `pipeline/nocar.py` from `world_run.py` and `docs/world-run.md`
  - Retiring the 5 `BROKEN` checks / migrating the 3 `sheetLines.ts` checks to WASM so the JS duplicate can be deleted
  - Doc drift cleanup (`SPEC.md`, `docs/interface.md`, trimming `docs/demand-worldwide.md`, or splitting `demandClient.ts` later)
- **Anita (+ friend)**:
  - AWS IAM scoping (already done for `anita-own` / `bench`; remove Jean's `[default]` key when ready).

---

## Pattern A: Calling Gemini (`agy`) directly from Claude's orchestrator

Anita installed the Antigravity CLI (`C:\Users\anita\AppData\Local\agy\bin\agy.exe`, already on User PATH). A read-only wrapper is ready at `trainworld/tools/gemini_audit.py`:

```powershell
# One-shot read-only audit written to a file (keeps orchestrator context small):
venv\Scripts\python.exe trainworld/tools/gemini_audit.py --brief <brief.md> --out trainworld/research/<out>.md

# Inline prompt (prints to stdout) or JSON output (includes conversation ID for follow-ups):
venv\Scripts\python.exe trainworld/tools/gemini_audit.py "Audit <area> for bugs or drift" --json
venv\Scripts\python.exe trainworld/tools/gemini_audit.py --resume <conv-id> "Follow-up question"
```

- Runs `agy -p` with `--mode plan --sandbox --print-timeout 15m` and `CREATE_NO_WINDOW`, so it cannot edit files, open windows, or hang indefinitely.
- Optional: add `trainworld/tools/gemini_audit.py` to `.claude/settings.json` allowlist if Claude should be able to invoke audits without prompting Anita.
