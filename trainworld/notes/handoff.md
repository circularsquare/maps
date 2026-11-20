# Orchestrator handoff, trainworld

Orchestrator: session f4ff94a8-c06c-4457-a593-caeb3269d7c2, name maps-b4 [173f33], since 2026-11-20 15:01 UTC
Updated: 2026-11-20 18:52 UTC (maps-b4 past 500k, starting its successor)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = 89d6ca0d. Master pushed through ac137c93; ~15 landings since, push soon.
1. Evening update (her: "wait until this evening"). Before it: land T-3735 (tree weekenddep) if weekendchk passes
   and T-3883 (tree rampbug) if rampbugchk passes (both checkers report via inbox; finish with --task). If
   rampbug fails, ramps (T-3027, c025) wait: hide c025 or hold. Then smoke on master, 8800 rebuild, ping her, publish
   skill (sort, highlights, playtest list, translations incl. T-3822/3823 hovers + addCity + today's strings, new
   zones image for PK/BD, smoke8800). Server first, her yes: shonei-server 2754ba1, fd294e7, 43360be, 39e1bd2 (not
   pushed; 39e1bd2's follow-the-game test needs T-3735 landed; if T-3735 doesn't ship, revert 39e1bd2 or ship it
   anyway: level 32 accepted is harmless). After publish: post PR 24 reply (notes/T-3659.md, her OK), open-source sync.
2. Running agents (their reports reach the inbox; follow-ups to fresh agents on their tree): hourturn (T-3804 st1,
   ~1 h left, edits clock.worker.ts; checker before landing), trainrate (T-3550; checker before landing), tramhwy
   (T-3831), frontier (T-3887 chart, 3-4 h), weekendchk, rampbugchk.
3. Then: T-3856 plain buses (after frontier's first chart; rerun it after), T-3888 walk tiers design (her call),
   T-3378 BRT design to her; held T-3655, T-3646 until trainrate lands; T-2374 clock worker split after hourturn.
   Board queue: `tasks.py upnext list`. Track rework design (T-3444) after cleanup + edit speed, her order.
4. T-3395: run `tools/ops/save_health.py` after the 11-21 04:00 UTC scan; clean = close it.
5. Leftover trees ttchk2base, touchrule, timekeys.

## Her words
- 11-20 15:20 UTC: "cleaning up some of the minor seeming tasks in code/docs cleanup, and edit speed ... not sure
  how many of them still make sense / would be valuable to do. and after that ... full track rework."
- 11-20 ~19:10 UTC buses: frontier chart first (T-3887, frontier), walk tiers design (T-3888), plain bus no boarding
  cost/85%/near-free/0.8x fare (T-3856, after the chart), BRT hybrid line upgrade + stations (T-3378), trams on
  highways now (T-3831, tramhwy). Her words in those tasks.
- 11-20 17:55 UTC she played League and wanted it lighter: cap 6 and tell agents at once; back to 14 on her word.
- 11-20 ~17:45 UTC: PR 24 reply (notes/T-3659.md) "looks good": post it after tonight's open-source sync.
- 11-20 ~17:35 UTC: profit hover just "profit per weekday"; city/country profit boards (T-3885); losses unlisted;
  old ramps no limit ok; HSR ramps still 160 after build on 8800 (T-3883 P1, rampbug: blocks tonight's T-3027).
- 11-20 ~16:30 UTC: text ack T-3695 and T-3027; c025 in her words; schematic "looks really fantastic".
- 11-20 ~16:00 UTC: T-3869 text ack. 13:29-14:55: "ping me when it's all on 8800" (save fix still to come).
- 11-20 ~14:00 UTC: no minor track work (T-3662, T-3353 wait on the rework design). 400 km per ferry leg stays.
- 11-20 ~14:45 UTC: slopes "kinda separate from track redesign ... so we can do this soon": 200 m per level = 160,
  400 m = 320, floors trains 5% / trams 8% (T-3027); real terrain as cost later (T-3871, P4).
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply herself.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
