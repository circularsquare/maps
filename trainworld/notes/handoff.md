# Orchestrator handoff, trainworld

Orchestrator: session d15915ac-6dd5-4586-bd98-843cbc616824, name maps-ac [1ef926], since 2026-11-21 15:53 UTC
Updated: 2026-11-21 19:55 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
0. LIVE v0.4.5. Cap 14. Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at takeover). No publish yet ("no rush").
   Before the next publish: fresh freeze (place index), translations (T-3986, T-3991, T-3999, T-4003, T-4010 keys),
   server learns save level 37+ (busways; boardonly may add one), batchchk11. Standby tabs: notes/standby.txt.
1. Master 28531e0f, unchecked since batch 10: oneway T-4000 (own check PASS), pausebp, placeindex, brtdash,
   hovers, kinds2, brtwait, helppanels. 8800 = 182e8e8e (lacks helppanels). batchchk11 after busways lands (add:
   a stop on a station joined to nothing does nothing and says nothing).
2. TRACK (her top priority): stage 1 T-3948 landed b10862a0 (guardchk PASS). tw/track cut at ac49304b. Running:
   bundles T-3949 stage 2a (Opus, tree from tw/track); landinto T-4016 (Sonnet: agent-tree `--into tw/track`,
   `sync-track`); newerlevel T-4014 (Opus, P1: refuse saves from a newer level; never upload over newer).
3. BRT (her other priority): busways T-3378 stage 4 in tree (level 37); buswaychk: 4 PASS, saves FAIL (a level-37
   save opens silently as rail on master: general fix is T-4014, now P1-ish, start after trackprep lands);
   buswayfix (Sonnet) rebasing onto master + Blueprint total fix; then land.
   boardonly T-4013 done in tree (archived): ALSO takes level 37 (LBO1). After busways lands: rebase it to 38, then
   its own save checker (it also fixes a demand clamp: riders no longer get off at one-track stops a train passes
   on the other track; may shift riders a little), then land. Server must learn 37 and 38 before publish. Open her call: any bus 60 km/h on busways (built) or BRT only.
4. Running: guardchk, buswaychk, boardonly T-4013 (Opus), splitb3 T-3373 B stage 3 (Opus). oomretry T-3933 landed
   6fcccc9a. Held: T-3932 (splitb3's worker pool). Up next on the board: draw from it when below 4.
5. Open asks: T-3742 nudge design (a-..-16), T-4008 soft rule (a-..-17); T-2264 help wording; busway speed.
   Later: T-3937 quiet machine; T-3931 held for T-3231; T-3655 faster full-track edits; T-3939 B's 140 MB.
## Her words
- 11-21 ~19:20 UTC: BRT stations come with a waiting room, shown as bought (T-4012, brtwait running); tram icon's
  top piece bigger; "text ack" (kinds2 strings); boarding-only / alighting-only stops filed (T-4013).
- 11-21 ~18:45 UTC: main priorities now: keep the track rework moving, finish and polish BRT. She's out with remote
  control on this session; standby orchestrator tabs (notes/standby.txt, docs/orchestrator.md "Your own cycling").
- 11-21 ~17:50 UTC: "text ack" for everything listed to then (viewonly, BRT stations, modeldown, brtblueprint,
  community, stationkinds). Search must reach far towns, "general" against crashes (T-4004 built). Server deploy yes.
- 11-21 ~17:00 UTC: closed T-3365; levels -5 to +5 with the track rework, Settings toggle (T-3691). Station totals by
  mode with station timetables (T-3780). Exact sums OK ("<0.5% is never that much"). Community: open paused,
  blueprints hidden, no leave confirm (T-4002); bookmarks maybe (T-4005); raise limits (T-4006, deployed).
- 11-21 ~16:40 UTC: bus reach 1 km (T-3995); no borrowed-search bus stops (T-3992 closed, "a lie"); BRT built border
  inside the colour, dashes fixed on the ground (T-3996); station tool train/bus/port, open-ground stations (T-3999).
- 11-21 ~16:10 UTC: Report as a flag with a confirm (T-3991); Discord and leaderboards in Community (T-3990);
  itch.io $0-or-donate, $5 suggested (T-3988).
- 11-21 ~15:55 UTC: track rework all calls made (notes/T-3444-design/answers-2026-11-21.md). 38 stages
  T-3948..T-3985 on tw/track.
- 11-21 ~14:05 UTC: keep 4-12 agents. Count every Auto stop ("if people are seeing bus stops the bus stops have to
  be real").
- 11-21 ~12:50 UTC: BRT stations for BRT lines only; T-3373 B go, the memory limit rises (players set it: T-3932; a
  load out of memory retries lower: T-3933). Never ask again whether big saves get more workers.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply and the T-3945 reply herself (draft in notes/T-3945.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- Commit messages through PowerShell: single quotes, or "$2M" becomes empty (49fa812e lost "US$2M").
- Landing scripts: wait.mjs refuses waits over 170 s; loop it.
