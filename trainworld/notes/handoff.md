# Orchestrator handoff, trainworld

Orchestrator: session f4ff94a8-c06c-4457-a593-caeb3269d7c2, name maps-b4 [173f33], since 2026-11-20 15:01 UTC
Updated: 2026-11-20 16:20 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = 7637e0e0. Landed since, not on 8800: T-3846, 3847, 3845, 3870, 3867. Master pushed
   through fe638e5b. Update this evening (her 16:30: "wait until this evening ... a lot of other minor improvements"); publish skill.
1. Old tab maps-a3's agents (reports via inbox; follow-ups to fresh agents): gapfill (T-3764; join done 15:5x, all
   338 rows shipped; running after-join checks + smoke on the main tree: hold landings until its report, then
   commit its files), rampspeed (T-3027: checker before landing). Mine: hourturn (T-3804 st 1), modesplit (T-3695),
   reportsfix (T-3841), streetspeeds (T-3460).
2. Landing queue after gapfill: bigload (T-3868, no checker; then a fresh agent for its option B); riverfix (T-3839,
   no checker; her "good to ship"; c018 reword in its notes); islands1 (T-2216 st 1 PASSED; land WITHOUT --task,
   rewrite T-2216 to stages 2-3); savefail (T-3869, text acked; rebase after bigload, then a checker); one smoke,
   8800 rebuild, ping her, push. After gapfill's
   commit, tell hourturn, modesplit, streetspeeds to recopy the main tree's app/src/game/*.json (PK/BD changed them).
3. T-3395: 11-20 scan (pass 1, not done) found 0 new damage. Run `tools/ops/save_health.py` after the 11-21
   04:00 UTC scan; clean = close it (the two rev-1 games of ue83342ed78f8fc4c are old 11-19 damage).
4. Up next is the board queue (`tasks.py upnext list`). Her order 11-20 15:20: cleanup and edit speed first (both
   reviews applied: notes/review-{edit-speed,cleanup}-2026-11-20.md; cleanup pieces table there), then the track
   rework design (T-3444). T-3873 net.rs split: her yes 16:30, starts when rampspeed lands. Zone fixes now P4.
5. Next update: unpublished changelog c001-c015 plus tonight's; untranslated: T-3822's and T-3823's hover strings,
   diagramView.addCity; use the publish skill (review agent, sort, playtest list, translations, new zones image).
6. Leftover trees ttchk2base, touchrule, timekeys. Steam (T-3729): her shots in artifacts/trainworld/steam.

## Her words
- 11-20 15:20 UTC: "cleaning up some of the minor seeming tasks in code/docs cleanup, and edit speed ... not sure
  how many of them still make sense / would be valuable to do. and after that ... full track rework."
- 11-20 ~16:00 UTC: T-3869 text ack ("The browser ran out of memory saving this game...").
- 11-20 ~15:50 UTC: transit split by mode in every mode bar, toggle default off (T-3695, P2, agent modesplit).
- 11-20 13:29-14:55 UTC, several times: "ping me when it's all on 8800", also when the Mexico City fix (T-3868)
  and the save fix (T-3869) are on it. Ping her at the next 8800 rebuild, naming what is on it.
- 11-20 ~13:40 UTC: TODO.md "for anita" and player suggestions all filed (T-3850-T-3866); 400 km per ferry leg stays.
- 11-20 ~14:00 UTC: "i'd still rather not work too muhc on track stuff": no minor track work (T-3662, T-3353 wait on
  the rework design; T-2920 stays P4).
- 11-20 ~14:45 UTC: slopes "kinda separate from track redesign ... so we can do this soon": 200 m per level = 160,
  400 m = 320, floors trains 5% / trams 8% (T-3027); real terrain as cost later (T-3871, P4).
- 11-20 early: board changelog review is NOT the publish text OK; Resend stays free; intercity time "keep it as is".
- 11-19: "player complaints take precedent over other new features" (actual bugs); a full playtest list before
  every publish; fares by city: country or city multiples, never both. 11-18: small easy wins over design-heavy.
- Share previews: free Worker, careful with Cloudflare. Mod support (T-3377): never say publicly it's for vibe-coders.
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
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
