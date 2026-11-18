# Orchestrator handoff, trainworld

Orchestrator: session 59c85afe-f6fc-46fa-b1d6-fdaa946e7f95, name maps-4f, since 2026-11-18 12:08 UTC
Updated: 2026-11-18 13:10 UTC (batch landed, smoke passed, 8800 = d5714bee; workflow research running)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5; she's up.

## State
- v0.4.1 LIVE since ~05:35 UTC. Unpublished changelog: `changelog.py list` (c001-c014 plus this batch's items).
- Night run finished: T-3511 (12 enclave rows) and T-3455 (all 28 coastal rows, Malaysia's 4 after a rerun) shipped
  to data/packs; committed 1ef0cee1 with osm_units.reclaimed's fix. Undo points data/work/pre-T3455/{pre,prejoin} and
  pre-T3455my/prejoin can go once 8800 checks clean.
- Landed and archived: T-3482 a9bbcb5f, T-3645 6e542f1b, T-3648 0e9da2cc, T-3650 021b9b38, T-3649 d5714bee
  (changelog c015 T-3645, c016 T-3650 reworded, c017 T-3649; T-3482 is c014). Smoke on master PASSED (29 checks).
- 8800 = d5714bee (swapped 13:06 UTC): has the night run's zones and every landing so far.
- Up next (board): T-3604, T-2136, T-1080, T-930; plus T-3620, T-3623 now the run is over.
- workflow (Opus, background, no tree): research + ranked proposal for her ask (workflow improvements, code/test
  trimming, agent tools) -> trainworld/research/workflow-2026-11-18.md. A design to her; nothing built until she picks.

## In flight
- tramoneway (maps-ca's, Opus, .tw-trees/tramoneway, port 6194): T-3647 (P1) dattera's Amsterdam tram save on
  one-way stops; reports reach maps-ca, which relays. Also asked to say if T-3651 (Zui, passing tracks over side
  platforms) is the same cause. His real save needs her word for ssh.


## Waiting on Anita (asks.json)
- T-3298 walking change limit 300 m -> 600 m? (recommended 600 m for all)
- T-3630 start menu settings A/B/C (artifacts/trainworld/review/t3630-landing.png; recommended B)
- Text: T-3649's help Q/A sent to her 12:45 (no ack yet); earlier strings in maps-ca's 12:08 report.
- CinnamonVII's PR 21: close with a link after the next open-source sync; draft the reply for her first.
- Filed from her pads, not started: T-3652 Community Newest, T-3653 time controls (proposal first), T-3654 map
  drawing (design first), T-3655 held line keeps its old rate.

## Next
1. Her ask (13:00 UTC): after current work, workflow improvements, code/test trimming, agent tools. Send her the
   workflow agent's ranked proposal when it reports. Meanwhile T-3620/T-3623/T-3604 (pipeline, core cap) may start.
2. Delete data/work/pre-T3455/{pre,prejoin} and pre-T3455my/prejoin once she's played the new zones on 8800.
3. Leftover .tw-trees: touchrule (empty, locked), shenzhen (dirty pipeline/osm_units.py, whose?), tt2 (stale).
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json.

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- She edits each update's changelog on the board before publish (unless ~2 items) (docs/orchestrator.md).
- Lean testing: concise time-boxed checkers, no checker for small UI, smoke per batch.
- One phrase per wait in player text: loads / shows up / calculating trips / memory.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
