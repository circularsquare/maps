# Orchestrator handoff, trainworld

Orchestrator: session 2ba1fd73-4c91-45ba-b838-fe8f908d4b21, name maps-18 [8ecfe1], since 2026-11-11 11:10
Updated: 2026-11-12 ~04:50 UTC (she went to bed; may start a new orchestrator: maps-18 then lands its running agents)

Read trainworld/CLAUDE.md and docs/orchestrator.md first. The 11-11/12 log this replaced is in git history
(handoff.md at its previous commit).

## Live
- **Live = v0.3.1** (tw/release-031 head 71009033, worktree .tw-trees/hotfix031; data 2026-11-11b; website cc14a89).
  v0.3.1 = T-3239 units fix + Community search/time played/hidden names/start-menu button + T-3238 save hang fix.
  I published its changelog without her text ack; she had one line removed (republished). Memory: a deploy yes is
  not a text ack.
- **Server** (shonei-server, deployed 02:54Z 11-12, pushed): caps 30k accounts / 16 GB games / 3 backups (T-3251),
  boards fb9588d. **Committed, not deployed (her word, morning):** efb2cd6 (every board = best game there, T-3248
  round 3), plus whatever commview (T-3258 random likes tiebreak, T-3259 her 60-min exemption) and flags (T-3269)
  commit. Push after deploy (she allowed pushing deployed commits).
- **Off-site backup installed** (T-2918): nightly 03:26Z to R2 bucket shonei-backups, 7 kept (privacy page says a
  week), rclone 1.76 from rclone.org; key at ~/shonei-backup-key.txt (she emailed herself a copy). README 8ac8f5c.
- **Cloudflare**: cache rules "trainsworld assets" + "trainworld data" (Browser TTL respect origin), Smart Shield /
  Tiered Cache on. Ops token at ~/cloudflare-ops-token.txt (zone read, cache rules edit, zone analytics, R2, account
  analytics). tools/ops/check_resources.py reads real R2 counts; run daily, report flagged lines.
- **R2 reads, 04:35Z check**: trainworld data now mostly HIT. The ~800k GetObject an hour left are **noritetsu's
  pmtiles** (data.anita.garden/noritetsu/data/*.pmtiles, 1-40 MB, `no-cache`, unversioned names, under her "data
  bypass" rule): jp 287k, us 52k, id 56k, cn 29k per 30 min. Proposal to her: a cache rule for /noritetsu/data/ with a
  short edge TTL (~2 h, ignore cache-control), placed after "data bypass"; pmtiles' etag check copes with a change.

## 8800 and master
- 8800 = 9ced7beb (satellite fade), smoke PASSED. Master has since: 3ea3c6e7, f25f2180 + 1e18c190 (T-3240/3254 pull,
  config "on": false), 2e0638d2, 1f89d798, 58cf2a89.
- Landed today (all on 8800): T-3242 names, T-3239 + option A, T-3238, T-3212 split, T-3241 rebuild across stations,
  T-3250 return trains drawn, T-3249 satellite + T-3267 fade, T-3225 30 auto real networks.

## Running agents (resume by id; all report to maps-18; no commits except in shonei-server)
- buildfix a4d09f6f5baf12d0c (6143): T-3260 Ctrl straight, T-3261 quad snap, T-3262 type switch resets count / HSR
  single, T-3263 road snap lag. May touch sim/src/track/wasm/draw.rs.
- commview afcbcaeb2cb0fb5be (6144): T-3257 Community "View" (strings to her before shipping), T-3258, T-3259
  (shonei-server commits).
- hexterrain aaac72d73e4b15746 (6145, Sonnet): T-3266 back to squares; pricing follows (sim cost.rs/params.rs).
- sbahn ae8f4f1cf4c2bb5e9 (6146): T-3265 commuter rail missing in real networks; rebuilds data/packs/real saves.
- split ae432976a77e94311 (6135): T-3212 follow-up, no thickening when zoomed out (her test).
- zoombtn a89c7c2c54b523ba9 (6148, Sonnet): T-3268 zoom buttons + Settings switch; Privacy · Terms spacing.
- flags aee74445698fcc4c7 (6149): T-3269 flags by names (server + client).
- trainsdraw a924318684a130567 (6147): T-3264 proposal only (general fix for trains drawn through each other).
- aimpact a0a866213491b49cb (6141/6142): T-3253 option A impact before/after on saves; answers her 3 questions.
- pull2 a1242c4c91eb0c0c7: T-3254 small Poland-only pull under 300 km (her "very slight ... keep it pretty small").
- rbprep a75a6269120e3d2c7: T-3244 build (light-pass runbook, --airports, market_zones pool, gate split, Wikidata
  prefill via airports' fill_wikidata_boardings, campus outline fetch, baselines). Scratch only until go.
- airports accd664886e499af4: T-2896 items 1-5 (incl. airport trips crossing zone lines, her yes), scratch only.

## World rebuild: her go for overnight 11-12, on 10 cores, once fully prepared
Checklist before rbprep starts the light pass (it must ask first): rbprep's speed-ups proven + baselines saved;
airports' items 1-5 landed; pull config `"on": true` with pull2's Poland factor; T-3218 attendance 0.8 applied;
T-2112 seams.py before the joins; T-2627 market step under its limit; campus outlines fetched. During the run: no
WASM landings (the gate's Vite serves live app/), 8800 rebuilds via snapbuild-frozen.cmd, nothing else writes
data/packs (sbahn's data/packs/real saves excepted, finish them first). Light pass ~4 h with speed-ups (T-3244).

## Waiting on Anita (morning)
- Server deploy (efb2cd6 + commview + flags commits).
- Noritetsu pmtiles cache rule (above).
- Test satellite at 10/40/100% light and dark (she said morning).
- Strings to review when agents report: Community View, track tool hints, zoom buttons, flags, satellite ("Map",
  "Satellite", hover "Low resolution outside Japan and the US").
- Translation batch for all new strings before the next publish.
- Longer-term TODO items not filed (flags/zoom now filed): leaderboard map view, brainstorm, intercity-only mode,
  more languages, offline, advanced quad track. She chose zoom buttons + flags for now.
- Her calls 11-12: option A stays (conceptually fine); wealth tiers parked; Community Biggest left alone; China pull
  0.6; Poland slight; undershoot rule for all demand fits; hex terrain back to squares; best game on every board.

Next free id T-3270.

## Her calls 11-10/11 (for the record; detail in DECISIONS and ARCHIVE)
Single track: two-way by default, margin 240 s; a terminus branch stays two-way with its flat badge. Double track is
lanes in one piece, never deletable by halves. Flyover ramps 4%. Margins: trams ~40/h, terminals 30/h, junction moves
120 s. Community v1 = share, share link, open a copy (on by default), country leaderboards; copies named
"<title> - <name>"; Biggest points-only with the sandbox note. Click standard as notes/T-3149.md "Proposal". Track
widths 0.71x / 1.24x. 11-11: train count cap = running + 1; HSR and tram quad yes (tram quad no street running);
buses and trams on any road in the data; snap to roads (no discount, below grade too); T-3161 option 1; T-3164 text ok;
text copied word for word from her message needs no review (CLAUDE.md); hotfix v0.2.1 done; rebuild speedups 1-3 yes;
R2 purges via the guarded script; "small changes all look fantastic", "target speed / auto flyovers feel really nice";
4-6 min rebuilds are "not so bad", only low-hanging fruit. v0.3 notes are her 16 lines (tram parenthesis dropped);
single track is not a nerf; only the nerf figures red; Privacy · Terms in the signed-in panel; privacy/terms text ok.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- Cores: ledger cap 14.
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
