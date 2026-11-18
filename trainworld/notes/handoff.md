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
- Up next (board): T-2136, T-1080, T-930; T-3620 (Caribbean etc.) held: she'll say when (11-18 morning, "i will lyk").
- cairo (Opus): T-3623 code LANDED 5e36f4ed (places.py: node in a small outline hole kept, same-named area merges
  into its node); 26 packs change. cairo now shipping in the MAIN tree (backup data/work/pre-T3623, places on 26 ids,
  start_difficulty --fresh, zone_aliases, changelog). Commit its outputs after; archive T-3623. Filed T-3660
  (downtown Cairo unit in no zone, next night run) and T-3661 (zone centres on the wrong town).
- T-3604 LANDED 2936bc8e (6 shared download places on reopen; 20/20 all-China restarts clean; rest factor 1.12;
  c022 reworded to not claim the stuck pool, which wasn't reproduced). Not on 8800 yet.
- T-3583 closed: done by T-3650. T-3298 closed on her word (300 m stays; passages), DECISIONS.
- T-3647 proposals not built (from tramoneway): a string for "track runs the other way"; "Run trains one way on this
  track?" reads backwards when turning it off. She sends dattera's reply herself.
- LANDED: T-3647 79783877 (tramcheck PASS; c018, c019 reworded), T-3658 87158649 (c020; not reproduced live,
  her Test item). T-3623 shipped to data/packs (26 places) and archived 58be946a. 8800 = 58be946a, master pushed.
  T-3662 filed (switching one kerbside piece breaks lines with no refusal; master too).
- shonei-server DEPLOYED 13:30 UTC on her yes and pushed (4d07666: leaderboard country tables + T-3395 stages 1
  and 3, save levels 19-21, playtime cap). T-3395's game side (stage 2, on 8800) may ship with the next publish;
  close T-3395 after a day of clean scans (she leaves it to the scan's emails).
- T-3657 free fares: put off by her (P3, needs more thought: 0x on most lines still cheats a game-wide ratio).
- T-3659 filed: contributors' speed PRs (caelum418 23, 24; CinnamonVII 18, 19) for the next performance round.
- landing (Opus, .tw-trees/landing, port 6191): T-3630 she picked C + footer language + gear mini settings;
  intercity becomes a setting everywhere, the checkbox replacing her 11-10 menu rule (tell her; DECISIONS).
- She keeps both T-3647 rule changes; she'll send dattera's reply herself later.
- Workflow research DONE: trainworld/research/workflow-2026-11-18.md (337584e5), 7 ranked proposals sent to her
  ~14:20 UTC; waits on her picks (recommended start: 4 test builds, 6 landing guards, then 1 play/wait, 5 handoff
  status + asks.py, 3 report inbox; 2 is her setup). T-3663 filed (t1550 hover-tag lines fail).

## In flight
- Running: workflow (research), restarts (T-3604), landing (T-3630), cairo (T-3623 shipping in the main tree).
- sideplat (Opus, .tw-trees/sideplat, port 6192): T-3651 (Zui, passing tracks over side platforms; not T-3647's
  cause; her 14:00 "other people are having this issue"); also gathering similar inbox reports. Checker before landing.


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
