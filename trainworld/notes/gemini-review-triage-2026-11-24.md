# Triage of Gemini's trainworld review (2026-11-24)

Source: research/gemini-workflow-review.md. Each point checked against the code, docs, TASKS.md, ARCHIVE.md and DECISIONS.md as they are tonight. 23 points: 11 worth doing (grouped into 6 tasks below), 3 already done, 3 already filed, 6 not worth it.

## Worth doing, best value for effort first

Proposed head lines for the orchestrator to file (ids left blank).

1. Rules for Gemini before it works in the repo (review "Suggested split")
- T-???? [P2] {L:checks} {waits:Anita} Rules for Gemini agents in maps
  - Need from Anita: Will Gemini get tasks in maps, and if so may it ever touch AWS? I'd say tasks yes, AWS no.
  - Context: None of the Claude hooks run for Gemini: the AWS guard, the refusal to edit TASKS/DECISIONS in a tree, the agent guard. It reads its own instructions file, not CLAUDE.md, unless told to. Its review was well behaved, but a working agent would need the same rules.
  - Goal: Gemini's instructions point at the maps and trainworld CLAUDE.md files and say no AWS, no commits, worktrees only through agent-tree.mjs. About 1 h.
  - Who decides: Anita (her rules, and who works on the project).

2. A scoped AWS key (review 1.1)
- T-???? [P2] {L:world-tooling} {waits:Anita} AWS key limited to our bucket and machines
  - Need from Anita: Ask your friend if you may make a second IAM user with only our rights in their account; the agent writes the policy.
  - Context: User anita has AdministratorAccess on the friend's account. tools/aws/guard.py only sees commands typed into Claude's shell: a Python script file using boto3, or any non-Claude agent, goes straight past it. T-4166 (your own account) has 5 vCPUs, too few for builds.
  - Goal: The key in ~/.aws/credentials can only use the anitabuilder bucket and tagged machines in us-east-2; tools/aws/ still works end to end; docs/aws.md says so. About 1 h agent time.
  - Who decides: Anita and her friend.

3. cores.py finds the main tree's ledger itself (review 2.3, second half)
- T-???? [P2] {L:checks} cores.py uses one ledger from any worktree
  - Context: cores.py keeps cores.json beside itself, so a claim made with a worktree's copy goes to a ledger nobody reads and can quietly go over the cap. cores.mjs, agent_cargo.py, perf_bench.mjs and edit_bench.mjs each work around it on their own.
  - Goal: cores.py resolves the main tree's cores.json (walk up past .tw-trees/ or git's common dir); the four workarounds removed. About 0.5 h.
  - Who decides: orchestrator.

4. The 500k hook says the same as docs/orchestrator.md (review 2.2)
- T-???? [P2] {L:checks} Cycling hook text matches the orchestrator doc
  - Context: The stop hook (.claude/hooks/orchestrator.py, the context message) tells the orchestrator to run next_orchestrator.py <project>, skipping the standby.txt check and --advisor fable that docs/orchestrator.md "Your own cycling" asks for. Its line 38 "no Fable unless she asks" is about agent models, which reads as a contradiction.
  - Goal: The hook points at the project's "Your own cycling" section instead of repeating steps; line 38 says it is about agents. About 0.25 h.
  - Who decides: orchestrator (it follows her existing rule).

5. Small drift fixes, one batch (review 3.1, 3.2, 3.3, 3.4, 2.5, 4.5)
- T-???? [P3] {L:cleanup} Stale lines in SPEC, docs and handoff
  - Context: SPEC.md section 3 still says "about 800 zones" (the code tour says over 4,000). docs/interface.md's redirect table lists "Zen Maru Gothic ... no transitions" as if current. handoff.md gives two 8800 commits and warns about .tw-trees/arrivefirst, which is gone. reserve_cores.ps1's top comment says claude.exe stays at normal priority; the code raises it.
  - Goal: Each line says what is true now; TASKS.md's header says P4 tasks anywhere are parked; the two untracked .stackdump files deleted. About 0.5 h.
  - Who decides: orchestrator.

6. Prune finished worktrees and branches (review 2.3, first half)
- T-???? [P3] {L:checks} Remove finished worktrees and old tw branches
  - Context: 19 worktrees at about 1.2 GB each, a leftover empty .tw-trees/rampport folder, and 34 tw/* branches, 14 with no tree. handoff.md keeps coreslider, onewaylag, split6, stationids and timetables on purpose until after 0.4.6; others may be done.
  - Goal: Every tree whose agent is done and whose work landed is removed with agent-tree.mjs; merged branches with no tree deleted. About 0.5 h.
  - Who decides: orchestrator.

## Already done

- wiki/html is not committed (review 3, "3 copies"): git tracks wiki/*.md and wiki/read/ only; wiki/RULES.md says to commit read/.
- Runtime files in git (review 2.4): notes/board-live.json is already in trainworld/.gitignore. asks-log, task-history, perf and edit ledgers are records, committed in the bookkeeping commits.
- .tw-trees/arrivefirst (review 2.3): the folder is gone; only handoff.md's warning is left (in task 5).

## Already filed

- pipeline/nocar.py (review 4.3): T-926, Drop the unread nocar layer, waiting for the next data batch.
- The 5 BROKEN checks (review 4.1): T-3697 covers t2162, t2289 and t2407; t1201 (fare rows moved) and t2626 (too slow) fit T-3457.
- Check sprawl (review 4.2): T-3457, Slim the standing checks.

## Not worth it

- PostToolUse reply-nudge on every tool call (2.1): measured about 0.1 s a call, 0.06 s of it Python starting; the 3 MB transcript read adds about 0.02 s, and subagents already stop at the first line. Small beside a turn. If anyone touches it: check the session is an orchestrator before reading the transcript.
- One shared lock helper (2.4): no lost write found in ARCHIVE or DECISIONS; agents also edit TASKS.md with Edit, which a lock in tasks.py would not cover. asks.py's lock gives up after 10 s and says to delete it.
- settings.json repeats scripts in several path forms (2.5): that is how permission prefixes match; harmless.
- Trim SPEC section 6 (3): 92 lines, under the 150-line move rule; task ids are how rules trace back to their reasons.
- Trim docs/demand-worldwide.md history (3): the dated country headings are citations; the sections under them are current per-country rules.
- Delete the JavaScript carry solve in sheetLines.ts (4.4): only the solver part is duplicated, not the 1,264-line layout file, and the JS solver also backs ?perfOff=carryWasm for comparisons. Moving three checks to WASM saves little.
