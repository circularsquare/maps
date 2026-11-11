# Rules cleanup proposal (2026-11-11, agent rulesaudit for maps-04)

**Applied 2026-11-11 by agent rulescleanup** (Anita: "lets do all of it", plus "keep stuff slim"). Differences from the
plan below: the compact instructions stay in maps/CLAUDE.md (Claude Code reads them there when compacting); global edit
2 lost its "Check names a doc..." clause, which her new "Never send her to read code" paragraph already covers; the
religiondots shapes went to a new memory file (`reference_religiondots_concurrent_builds.md`), not religiondots' notes.

What a trainworld agent reads before working today, by size:

| File | Size | Who reads it |
|---|---|---|
| `~/.claude/CLAUDE.md` (hers) | 14.9 KB | everyone |
| `maps/CLAUDE.md` | 5.1 KB | everyone in maps |
| `trainworld/CLAUDE.md` | 26.1 KB, 293 lines | every trainworld agent |
| memory `MEMORY.md` index | 16.1 KB, 134 lines | everyone in maps (files behind it on demand) |
| `trainworld/notes/handoff.md` | 13.9 KB, 147 lines | the orchestrator |

## 1. Why agents keep contradicting earlier rules

- **The same rule lives in three to five places, and only some copies get updated.** The core cap is written as
  10 (maps/CLAUDE.md, memory `feedback_cap_cpu`), 14 and "12-15 overnight" (trainworld/CLAUDE.md), 12 (the ledger
  and handoff) and 4 (isochrone). Pushing is "never unasked" in one memory and "a couple of times a session" in
  trainworld/CLAUDE.md. Image links are "absolute path" in her global file and "workspace-relative" in two memories.
  Whichever copy an agent read last wins.
- **New exceptions get added to the end of long paragraphs.** "The wiki is not player-facing" (added today) is
  sentence nine of a 17-line paragraph that opens "Anita reads every player-facing string before it lands". Her
  global file defines user-facing text as "anything a player or site visitor reads", and `wiki/RULES.md` tells wiki
  writers to follow that section. So by two of three files the wiki *is* player-facing. The same paragraph also says
  both "before it lands" and "landing needs no wait".
- **Rules that bind subagents sit in the orchestrator's handoff only.** "Estimates in hours, not agent-days" is a
  standing ask in handoff.md; the agents who write the estimates (notes/T-2921.md, T-3137.md say "agent-days") never
  read that file. Same for "no deploys overnight" and "server deploys need her word each time".
- **Some rules exist nowhere at all, and an example points the wrong way.** No file says she does not read
  trainworld's code; the only hint is a memory line ("She won't write code"). Her global "For you" block uses
  `HazeLayer.cs:88` as the model Check item, so an agent naming `app/src/game/achievements.ts` was following the
  example. Nothing says to ask a question only when its answer is needed next, and the global Questions heading
  lists "a push", which invites asking about deploys early.
- **Orchestrator procedure fills the file every agent reads.** About 110 of trainworld/CLAUDE.md's 293 lines are for
  the orchestrator, data runs or one kind of task (commit steps, trees, snapbuild, pack rebuilds, joins). The rules
  that slip most (talking to her, what waits for her word, player text) are at lines 133-177 and 283, after 70 lines
  of doc hygiene.
- **Memory files go stale quietly.** They are written once and not touched when CLAUDE.md changes: 9 of the ~30
  memories that touch trainworld, UI or orchestration now contradict a newer rule (list in part 5).

## 2. Findings

Kinds: dup = duplicate, contra = contradiction or ambiguity, stale, narrow (costs every agent, rarely applies), order.
"tw" = trainworld/CLAUDE.md, "maps" = maps/CLAUDE.md, "global" = her ~/.claude/CLAUDE.md.

| # | Where | What | Kind | Proposed fix |
|---|---|---|---|---|
| 1 | maps "Sharing CPU cores"; tw "CPU" (l.104) and "Task size" (l.280); memory `feedback_cap_cpu`; handoff; isochrone/CLAUDE.md | Core cap stated as 10, 14, 12-15, 12 and 4; the ledger says 12 | contra, dup | No number in any file: "the cap is whatever `python cores.py list` says; changes go through `cores.py cap`". Keep isochrone's own 4 as its share. |
| 2 | tw "Player-facing text" (l.133-149); global "User-facing text"; `wiki/RULES.md` | Wiki excluded from review only at the end of the paragraph; global and RULES.md read as including it | contra, order | Split into "voice" (applies to wiki too) and "review" (in-game strings only), with an explicit not-reviewed list: wiki, zone/region/realm names, docs, commit messages. Global edit 1 in part 4. |
| 3 | tw l.135 vs l.138 | "reads every string before it lands" vs "landing needs no wait" | contra | "She sees every in-game string you add or change; landing does not wait for her." |
| 4 | handoff standing asks only | Estimates in hours from how long agents really take; never agent-days | narrow-placed, missing | Move into tw "Talking to Anita" and the design-note brief template: "Sizes in hours of agent time, checked against similar past tasks (T-3137 C1: 'two to three days', took about 45 min)." |
| 5 | nowhere in tw; memory `project_trainworld`; global Check example | She does not read the game's code | missing, contra | tw: "Never point her at a code file. Quote the value or list in the message (the `NATIONS` thresholds as a short table) or name the place on 8800." Global edit 2. |
| 6 | handoff (Now, How to, Lessons, standing asks); global "a push"; memory `feedback_trainworld_8800_first` | Deploy and publish rules spread over six spots; asked about a server deploy long before it was ready | dup, contra | One "What waits for her word" list near the top of tw (publish, website push, server deploy each time, never overnight). Ask only when the thing is ready and she is around. Global edit 2. |
| 7 | global "Git commits"; memories `feedback_isochrone_commits`, `feedback_transituniversalis_commits` | Global lists shonei, shonei-server and trainworld as the only commit exceptions; isochrone and transituniversalis orchestrators commit too | contra | Global edit 3. Delete both memories (their CLAUDE.md files carry it). |
| 8 | memory `feedback_trainworld_commits` | "never push unasked"; tw and global now allow the orchestrator a couple of pushes a session | stale | Delete. |
| 9 | memory `feedback_link_images`, `feedback_no_file_cards`, MEMORY.md index | Image links "workspace-relative"; global says the full absolute path once | contra | Merge into one memory that defers to global. |
| 10 | memory `feedback_copy_review_images` (+ index line "renders go in the project root") vs tw l.130 | Project root vs `artifacts/trainworld/review/` | contra | Merged memory says: the project's review folder; trainworld's is `artifacts/trainworld/review/`. |
| 11 | memory `project_trainworld` vs tw file table and TODO.md's own header | "agents add only under 'Added by agents'" vs "agents never edit it" | stale | Fix the memory line. |
| 12 | tw file table (l.22) vs TODO.md | Rule names a "long-term section"; TODO.md now has short term / for anita / other player suggestions / long term / context | contra | "File only the short-term section. The others are hers unless she points at an item." Move to the orchestrator doc (only the orchestrator files TODO items). |
| 13 | memory `feedback_map_ui_style` | "uppercase 11px letter-spaced panel titles" (global forbids uppercase); "no animations on state changes" (dropped 2026-10-29) | stale, contra | Reword: sentence-case titles, motion allowed per `feedback_no_transitions`. |
| 14 | memory `feedback_run_it_for_her` | "The ~3-minute rule in her global CLAUDE.md still holds: ask before anything longer" (that rule is gone) | stale | Delete; global "Long-running commands" covers it. |
| 15 | memory `feedback_long_scripts` | Restates global "Long-running commands" word for word | dup | Delete. |
| 16 | tw "Task size" (l.267-280); memories `feedback_opus_for_ui`, `feedback_orchestrator_on_opus`, `feedback_no_fable_subagents`; handoff usage note; global tier section | Model choice in five places: "Sonnet default", "UI to Opus", "Opus more widely but not gung-ho" (11-10), "Sonnet by default again" (after v0.2) | dup, contra | One line in the orchestrator doc: "Sonnet by default; Opus for UI layout, design calls and debugging nobody understands; no Fable. This week's usage stance: handoff." Merge the three memories into one. |
| 17 | maps "Keep the work moving" vs handoff "Usage this week: fewer agents" | Start everything that can go ahead, vs fewer agents this week | contra | Keep both; say in maps that a usage stance in the handoff is a stated reason to hold back. |
| 18 | tw "Working here": countryside zones 0.5M, built cities not fixed, intercity, simple prices; panels show what can change, every quantity has a tooltip, negative money red | Already in `docs/zones.md` (0.5M, T-788) and `docs/ui-panels.md` (panels, tooltips, red money) | dup | Delete from tw; one line pointing at ui-panels.md "Look" for UI principles. Move "no dividers within a section" and "live feel" into ui-panels.md and SPEC (not there now). |
| 19 | tw l.213-215 | New countries: Korea, Canada, Mexico, Turkey, Vietnam (2026-10-24) | stale (likely) | All five are in `docs/data-register.md`. Drop the list; keep "whole world, tier 0 everywhere, tailoring later". |
| 20 | tw "Before a commit", worktree landing, long-run checks, handoff cadence, commits, serving snapshot, subagent treatment and cycling | Orchestrator-only, ~70 lines every subagent reads | narrow | Move to a new `trainworld/docs/orchestrator.md`, which `/takeover` reads in step 1. |
| 21 | tw "While a join runs", "Pack rebuilds", "Calibration of new cities", "Long runs are checked", AWS line | Data-run rules, ~25 lines | narrow | Move to `docs/world-run.md` (the runner's doc); keep one tw line: "Data runs, packs, joins and AWS: read docs/world-run.md first." |
| 22 | tw "CPU" (l.109-113) | `TW_CORES_WANT`, `TW_CORE_MASK`, `TW_NOPIN`, the 2026-10-27 story | narrow | Move to the header of `checks/cores.mjs`; tw keeps 3 lines. |
| 23 | tw "Bash" (l.125-129); memory `reference_gitbash_cmd_start` | `cmd /c start /b` through Git Bash, 5 lines in tw | dup | tw: one line ("pin processes from PowerShell or a .cmd, never `cmd /c start ... /b` through Git Bash"). |
| 24 | tw "Serving" (l.286) | `serve.py add trainworld ...`: already registered | stale | Delete; snapshot details go to the orchestrator doc. |
| 25 | tw "Token counts" (l.281) | `token_usage.py` when asked | narrow | Move to code-map.md checks list. |
| 26 | tw "Keep your context lean" (l.152-160) | 9 lines, partly orchestrator-only | narrow | 3 lines for agents (foreground waits under 4 min, background longer, final checks in the foreground); rest to the orchestrator doc. |
| 27 | handoff.md | 147 lines; the handoff skill says under 60. "Running" lists done and committed work; "Today's work" repeats ARCHIVE; "Keep 4 cores free for the next few hours" (22:32 yesterday) | stale, narrow | Trim to the skill's shape. "Lessons" and "How to" move to `docs/orchestrator.md`. Standing asks keep only ones that are not in tw. |
| 28 | `notes/for-anita.md` | "Open now (2026-11-04)": a week old, mostly answered | stale | Archive answered blocks; keep only what she still owes an answer on. |
| 29 | global "Final messages" vs tw "Reports to the orchestrator are short" | A subagent reading global writes a long standalone message with its own "For you" block, then the orchestrator relays asks that the project rules exclude (the wiki review) | contra | Global edit 4. |
| 30 | tw l.98 "no commits" vs global "shonei-server: commit freely" | community3 (a trainworld subagent) committed in shonei-server | contra (small) | tw: "A trainworld brief that touches shonei-server may say to commit there; pushing and deploying it are the orchestrator's, on her word." |
| 31 | memory `reference_concurrent_sessions` (12.5 KB) | Seven religiondots and seoulriders failure stories behind a general index line | narrow | Keep the general part (~20 lines: a file changing under you is normal; check the file before the code; never whole-file Write a shared file). Move the religiondots shapes into religiondots' own notes. |
| 32 | memory `feedback_shell_prompt_friction` | Most of it repeats global "Keep Bash commands auto-approvable" | dup | Trim to what global lacks: if the Bash classifier blocks a script, run it from PowerShell; no pipe on the invocation, redirect to a file. |
| 33 | `trainworld/tools/aws/guard.py` (seen while auditing) | Refused a read-only Bash call because a path list contained `tools/aws` followed by another path | tool bug | File a small task: match `aws` only as the command word. |
| 34 | tw order | Doc hygiene (47 lines) comes before the rules that bite | order | New order in part 3: talking to her, what waits for her, player text, then the rest. |

## 3. New outlines

### trainworld/CLAUDE.md: 293 lines, 26 KB -> about 140 lines, 14 KB (-45%)

1. **Read first** (5 lines): this file, SPEC.md, grep TASKS.md for your id, wiki then docs, code-map.md. As now.
2. **Talking to Anita** (10 lines, new section): reports go to the orchestrator, short; she does not read the code,
   so quote values and name places on 8800; estimates in hours of real agent time; ask a question when its answer
   is needed next; images: one file in `artifacts/trainworld/review/`, full path. *From: findings 4, 5, 6, 29.*
3. **What waits for her word** (8 lines, new): publishing and the website push; every server deploy, never
   overnight; adding a complicating factor to demand; a choice that changes what the game is; anything marked
   "open" in the spec; a hold she has set (named in handoff). Everything else you decide, record in DECISIONS.md and
   tell her after: rail facts, sensible defaults, zone and region names; a slip gets a fix, not a new process.
   *From: tw l.172-177, l.222-224, l.291-293; handoff; memories decide_more, 8800_first.*
4. **Player text** (10 lines): voice: global "User-facing text", plus "trips" for demand and "rides/riders" for the
   player's system; this applies to the wiki too. Review: list every in-game string you add or change, old -> new;
   landing does not wait; she answers "text acknowledged" or names lines. Not reviewed: wiki, zone/region/realm names,
   docs, commit messages. Watch list (object as subject, dropped words, hovers that repeat what is visible); a
   gesture with no visible control gets its instruction in a tag. *From: tw l.133-149, finding 2 and 3.*
5. **Files** (12 lines): the table, one line per row (docs/ row cut to "one file per area; index in docs/interface.md
   and docs/demand.md"). TODO.md row: "Anita's; agents never edit it".
6. **Doc rules** (24 lines): as now, each rule cut to two lines; the archive-band list becomes "unresolved notes/
   links are in notes/archive/<band>/".
7. **Working as an agent** (30 lines): before starting; own Vite on 6100-6199; a tree only if your brief gives one;
   headless Chrome with its own profile; the real save; typecheck, no full build, no commits (shonei-server note);
   Rust in 3 lines; CPU in 3 lines (ledger cap, Playwright self-claims, cap a page's demand workers to your claim);
   one line on `cmd /c start`; screenshots to `artifacts/trainworld/`; check your own screenshots before reporting
   UI work; refused permission: make it safer and retry; stop at a clean point; context lean in 3 lines; data runs:
   read docs/world-run.md first.
8. **Game principles** (16 lines): whole world, tier 0 everywhere, tailoring later; demand simple on purpose (A 85%,
   B 50%, notes/T-636.md); every demand task has a "done when"; general and global sources, look outward; demand rules
   hold worldwide (docs/demand-worldwide.md); approximations welcome when a player can't tell; buses settled; saves
   need no migration effort; UI principles live in docs/ui-panels.md "Look".
9. **Build and run** (4 lines): as now.

Where the moved rules go:

| Moved | To |
|---|---|
| Before a commit, smoke, snapbuild, smoke8800, landing trees, commits and pushes, serving snapshot, handoff cadence, TASKS id hand-out, TODO filing (verbatim briefs, which sections), long-run supervision (every couple of hours, ask after 45 quiet minutes), subagent treatment, thanks, 500k cycling and overnight, model choice | new `trainworld/docs/orchestrator.md` (~90 lines, read by `/takeover`), together with handoff.md's "How to" and "Lessons" |
| Joins, pack rebuild backups, calibration of new cities, long runs detached and checked, AWS | `docs/world-run.md` |
| Countryside zone floor, built cities not fixed, intercity option 2 | already in `docs/zones.md` / `docs/regional.md`; delete |
| Simple prices | `docs/regional.md` (fares) if not there; delete from tw |
| No dividers within a section, live feel | `docs/ui-panels.md` "Look" |
| `TW_CORES_WANT` / mask / nopin | header comment of `checks/cores.mjs` |
| `token_usage.py` | code-map.md checks list |

### maps/CLAUDE.md: 73 lines, 5.1 KB -> about 38 lines, 2.8 KB (-45%)

1. **AWS** (3 lines): as now.
2. **Serving maps locally** (12 lines): serve.py is the one server; the three commands; register a map when you start
   on it; relative URLs; no servers of your own. The Range and open-per-request detail shrinks to one clause.
3. **Sharing CPU cores** (7 lines): claim, release, list; "the cap is the ledger's; never write a number into a doc,
   use `cores.py cap`"; Playwright in trainworld claims for itself.
4. **Orchestrator sessions** (8 lines): handoff.md is live state; `/takeover`, `/handoff`; hooks; a usage stance in the
   handoff counts as a stated reason to hold back. "Keep the work moving" in full and the compact instructions move
   to the takeover skill's closing section (only orchestrators need them).
5. **Where rules live** (3 lines, new): a project rule lives in that project's CLAUDE.md or docs, once; when a rule
   changes, change it there and delete copies (handoff, memory); a memory file links to the project file instead of
   restating it.

### notes/handoff.md: 147 lines -> under 60, per the handoff skill

State, In flight, Long runs, Next, Standing asks that are not already in tw/CLAUDE.md. Done work goes to ARCHIVE, lessons
and commands to `docs/orchestrator.md`.

## 4. Proposed edits to her global CLAUDE.md

Four small ones. Each is optional.

**Edit 1: say the style rules are about voice, not review.** Reason: finding 2; the wiki was put up for review.

Old:
> For anything a player or site visitor reads: UI labels, buttons, tooltips,
> alerts, site copy, link previews, map labels. The aim is text that doesn't read
> as AI-written. Not for chat replies, code comments, specs or commits.

New:
> For anything a player or site visitor reads: UI labels, buttons, tooltips,
> alerts, site copy, link previews, map labels. The aim is text that doesn't read
> as AI-written. Not for chat replies, code comments, specs or commits. This
> section is about how text reads; which text she reviews string by string is
> each project's own rule.

**Edit 2: Check items in a form she reads; ask when it's time.** Reason: findings 5 and 6 (a .ts file named for her;
a deploy asked about hours early). Also folds in memory `feedback_for_you_testable`, which can then go.

Old:
> - Three headings, no others: **Questions** (anything you need her to answer or
>   decide — a pick between options, a priority, a risk call, a push, something
>   only she knows), **Test** (playtest, or a subjective visual call), **Check** (a
>   named file or constant worth her spot-check). Use the ones you have items for.

New:
> - Three headings, no others: **Questions** (anything you need her to answer or
>   decide — a pick between options, a priority, a risk call, a push, something
>   only she knows), **Test** (playtest, or a subjective visual call), **Check** (a
>   named file or constant worth her spot-check). Use the ones you have items for.
> - Ask when the answer is needed next, not ahead: a push or deploy once the thing
>   is ready and she is around. Test and Check items only once she can do them
>   (built where she plays, or published). Check names a doc, an image or the value
>   itself; a code file only in projects where she writes code.

**Edit 3: list every orchestrator that commits.** Reason: finding 7.

Old:
> **Exceptions: `shonei`, `shonei-server`, and trainworld's orchestrator.** In
> shonei and shonei-server (private repos) commit freely as you go. In maps, the
> trainworld orchestrator commits `trainworld/` paths without asking (its
> subagents never commit).

New:
> **Exceptions: `shonei`, `shonei-server`, and the orchestrators of trainworld,
> isochrone and transituniversalis.** In shonei and shonei-server (private repos)
> commit freely as you go. Those orchestrators commit their own project's paths
> without asking (their subagents never commit).

**Edit 4: a subagent's final message is for its orchestrator.** Reason: finding 29; subagents follow "Final messages"
literally and fill a "For you" block the orchestrator then relays.

Old:
> Your final message is the only part of the session she reliably reads. She is
> usually context-switching between several agents and did not watch the tool
> calls, so write it for someone seeing the work for the first time.

New:
> Your final message is the only part of the session she reliably reads. She is
> usually context-switching between several agents and did not watch the tool
> calls, so write it for someone seeing the work for the first time. (A subagent's
> final message goes to the agent that started it: follow its brief and the
> project's rules for reports, and mark what is for her so it can be relayed.)

Not proposed: moving the long "User-facing text" section out to a file. It is the rule agents slip on most, so
loading it on demand would likely make things worse.

## 5. Memory files to merge or delete

All in `C:\Users\anita\.claude\projects\c--Users-anita-projects-maps\memory\`. Update MEMORY.md's index lines to match;
together this removes about 15 index lines (16.1 KB -> about 14 KB) and several contradictions.

**Delete** (the rule now lives in a CLAUDE.md, or is stale):

| File | Reason |
|---|---|
| `feedback_trainworld_commits.md` | Says never push; tw and global now allow the orchestrator a couple of pushes |
| `feedback_trainworld_8800_first.md` | In tw ("publishing waits for her word") and the orchestrator doc (rebuild 8800 after landings) |
| `feedback_trainworld_decide_more.md` | In tw "What waits for her word" |
| `feedback_player_text_review.md` | In tw "Player text"; its "likely all her games" is contradicted by transituniversalis |
| `feedback_opus_for_ui.md` | Model line in the orchestrator doc; "check your own screenshots" in tw agent rules |
| `feedback_long_scripts.md` | Restates global "Long-running commands" |
| `feedback_run_it_for_her.md` | Cites a ~3-minute rule that no longer exists; global covers the rest |
| `feedback_isochrone_commits.md` | In isochrone/CLAUDE.md and global edit 3 |
| `feedback_transituniversalis_commits.md`, `feedback_transituniversalis_nudges.md`, `feedback_transituniversalis_no_text_review.md` | All three are in transituniversalis/CLAUDE.md (l.57-76) |
| `feedback_for_you_testable.md` | Folded into global edit 2 (keep it if edit 2 is declined) |

**Merge:**

| Files | Into | Reason |
|---|---|---|
| `feedback_link_images.md`, `feedback_no_file_cards.md`, `feedback_copy_review_images.md` | one `feedback_showing_files.md` | They disagree with global on relative vs absolute paths and on where review copies go; one file, deferring to global, review copies in the project's review folder |
| `feedback_orchestrator_on_opus.md`, `feedback_no_fable_subagents.md` | one `feedback_orchestrator_models.md` | Same topic; both small |
| `feedback_minimal_ui_text.md` | `feedback_map_ui_style.md` | Same topic; trainworld's copy is in docs/ui-panels.md |

**Reword:**

| File | Change |
|---|---|
| `feedback_cap_cpu.md` | Drop every number: "the cap is the ledger's; summed over all subagents; BLAS env caps" |
| `feedback_map_ui_style.md` | Sentence-case panel titles (global bans uppercase); motion allowed (`feedback_no_transitions`) |
| `project_trainworld.md` | TODO.md: "agents never edit it"; replace the four working-rule links with "rules: trainworld/CLAUDE.md" |
| `reference_concurrent_sessions.md` | Keep the general ~20 lines; move the religiondots shapes to religiondots' notes |
| `feedback_shell_prompt_friction.md` | Keep only the PowerShell fallback and the no-pipe advice |

Kept as they are (general, still true, not duplicated): `feedback_orchestrating_subagents`, `feedback_next_up_outline`,
`feedback_question_rules_and_ping`, `feedback_permission_rejections`, `feedback_use_idle_compute`,
`feedback_delete_downloads_when_done`, `feedback_trust_gut_on_local_rail`, `reference_gitbash_cmd_start`,
`feedback_no_transitions`, `feedback_requery_pid_before_stop`, `feedback_no_identity_in_requests`.

## Expected size

| Reader | Today | After |
|---|---|---|
| trainworld subagent (global + maps + tw + index) | 62 KB, about 15.5k tokens | about 45 KB, about 11k tokens (-27%) |
| orchestrator (adds handoff, and orchestrator.md on takeover) | 76 KB | about 58 KB (-24%) |
| trainworld/CLAUDE.md alone | 26.1 KB | about 14 KB (-45%) |
| maps/CLAUDE.md alone | 5.1 KB | about 2.8 KB (-45%) |

The size cut helps, but most of the gain is from having each rule in one place, near the top when it bites often.
