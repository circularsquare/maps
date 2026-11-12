# Orchestrator handoff, trainworld

Orchestrator: session 441e3071-8b6e-4109-9ab2-1339e17b92be, name maps-5e [3a1e06], since 2026-11-12 04:41 UTC
Previous: session 2ba1fd73-4c91-45ba-b838-fe8f908d4b21, name maps-18 [8ecfe1], handed over 2026-11-12 04:41 UTC, finishing in-flight work
Updated: 2026-11-12 04:41 UTC (machine clock 11-11 23:41 local; the old "~04:50" line was ~10 min ahead)

Read trainworld/CLAUDE.md and docs/orchestrator.md first. The 11-11/12 log this replaced is in git history
(handoff.md at its previous commit).

## Live
- **Live = v0.3.1** (tw/release-031 head 71009033, worktree .tw-trees/hotfix031; data 2026-11-11b; website cc14a89).
  v0.3.1 = T-3239 units fix + Community search/time played/hidden names/start-menu button + T-3238 save hang fix.
  I published its changelog without her text ack; she had one line removed (republished). Memory: a deploy yes is
  not a text ack.
- **Server** (shonei-server, deployed 04:49Z 11-12 with her word, pushed): caps 30k accounts / 16 GB games / 3
  backups (T-3251); efb2cd6 best game per board (T-3248); 020ba08 likes tiebreak + her 60-min exemption (T-3258,
  T-3259; SHONEI_COMMUNITY_NO_WAIT set, env backup shonei.env.bak-2026-11-12); e98ae76 flags server (T-3269, config
  trainworldFlags true; the client half is not live until the next game publish).
- **Off-site backup installed** (T-2918): nightly 03:26Z to R2 bucket shonei-backups, 7 kept (privacy page says a
  week), rclone 1.76 from rclone.org; key at ~/shonei-backup-key.txt (she emailed herself a copy). README 8ac8f5c.
- **Cloudflare**: cache rules "trainsworld assets" + "trainworld data" (Browser TTL respect origin), Smart Shield /
  Tiered Cache on. Ops token at ~/cloudflare-ops-token.txt (zone read, cache rules edit, zone analytics, R2, account
  analytics). tools/ops/check_resources.py reads real R2 counts; run daily, report flagged lines.
- **R2 reads, 04:35Z check**: trainworld data now mostly HIT. The ~800k GetObject an hour left are **noritetsu's
  pmtiles** (data.anita.garden/noritetsu/data/*.pmtiles, 1-40 MB, `no-cache`, unversioned names, under her "data
  bypass" rule): jp 287k, us 52k, id 56k, cn 29k per 30 min. **Fixed with her yes:** cache rule "noritetsu data"
  (data.anita.garden/noritetsu/data/*, 2 h edge TTL, after "data bypass"), Range MISS->HIT verified, DECISIONS
  139ba331. Look at its hit rate in the next daily resource check.

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
Agreed 04:45 UTC: rbprep and airports keep reporting to maps-18 for prep; when both say ready and pull2's Poland
config is in, maps-18 passes their reports to maps-5e, which owns the go/no-go and the run (WASM landings frozen).
commview landed as e6c88dd4 (maps-18 snapbuilding it to 8800, then smoke). Server deploys done 04:49Z.
Checklist before rbprep starts the light pass (it must ask first): rbprep's speed-ups proven + baselines saved;
airports' items 1-5 landed; pull config `"on": true` with pull2's Poland factor; T-3218 attendance 0.8 applied;
T-2112 seams.py before the joins; T-2627 market step under its limit; campus outlines fetched. During the run: no
WASM landings (the gate's Vite serves live app/), 8800 rebuilds via snapbuild-frozen.cmd, nothing else writes
data/packs (sbahn's data/packs/real saves excepted, finish them first). Light pass ~4 h with speed-ups (T-3244).

## Waiting on Anita (morning)
- Test satellite at 10/40/100% light and dark (she said morning).
- Strings to review when agents report: Community View, track tool hints, zoom buttons, flags, satellite ("Map",
  "Satellite", hover "Low resolution outside Japan and the US").
- Translation batch for all new strings before the next publish.
- Longer-term TODO items not filed (flags/zoom now filed): leaderboard map view, brainstorm, intercity-only mode,
  more languages, offline, advanced quad track. She chose zoom buttons + flags for now.
- Her calls 11-12: option A stays (conceptually fine); wealth tiers parked; Community Biggest left alone; China pull
  0.6; Poland slight; undershoot rule for all demand fits; hex terrain back to squares; best game on every board.

Next free id T-3272.

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
- **11-12 04:49Z server deployed with her word** ("yes good to deploy server changes"): 020ba08 (T-3258, T-3259), efb2cd6 (T-3248 best game per board), e98ae76 (T-3269 flags server; config trainworldFlags true, client not live). SHONEI_COMMUNITY_NO_WAIT line added from the testers backup (log: 1 account skips the wait); env backup shonei.env.bak-2026-11-12. Pushed. Close T-3258/T-3259 lines.
- pull2 done: Poland near_gamma 0.2 (Warsaw-Krakow 3.6k -> 5.1k, under 7.2k rail; nothing else moves), code in sim intercity, marketRule.ts, market_zones.py, t1148; UNCOMMITTED. market_zones.py, sim/src/demand and pipeline/* are shared by pull2, rbprep and airports: land them as ONE batch when rbprep and airports report (before the rebuild run).
- sbahn done: T-3265 landed a7188b05 (Berlin S-Bahn as commuter trains, ber.save reshipped; hand-made seven miss nothing). Her question: show all light rail tagged usage=main/branch... actually ALL light rail in the Real network layer and snapping graph (railway=light_rail; brings Berlin/Hamburg S-Bahn, Copenhagen S-tog; also Stadtbahn, US light rail, DLR) + rebuild 118 realgraphs (~1 h)? notes/T-3265.md. T-3265 line stays open for that.
- 8800 = e6c88dd4 (T-3257 View). Smoke there FAILED twice on "demand final" (300 s timeout on the real save) but both runs got 1 core (affinity 0x2000) with the ledger full; 9ced7beb passed at 3 cores. Likely load: re-smoke with 3+ cores before calling it a regression (if it fails again, suspect T-3257's clockClient switchGame/leaveViewed). Not yet on 8800: 4181e35f (zoom buttons), a7188b05 (pipeline only).
- Landed overnight: zoom buttons 4181e35f, S-Bahn a7188b05, flags client cb3715e6, square terrain cf3fd479 (WASM); **8800 = cf3fd479** (not yet smoked: ledger full; smoke with 3+ cores, see the demand-final note). airports done incl. hub O&D shares (Istanbul 0.5, Frankfurt 0.5, Doha 0.25, Dubai 0.5, Atlanta 0.45 via DB1B for planned US rows: ATL legs 0.98 -> 0.44 of published, tell her), T-3270 filed. Uncommitted batch for the rebuild: pipeline/* (airports, rbprep, pull2), sim/src/demand/* (airport draw, Poland pull), marketRule.ts, t1148. Still running: buildfix, split, rbprep, aimpact, trainsdraw.
- split follow-up landed (thin when zoomed out), not yet on 8800.
- buildfix done: T-3260..3263 landed 029a6f1c (WASM draw.rs). split follow-up c97eb4ba. 8800 snapbuild 029a6f1c + smoke running. Her question: T-3261 read as snapped routes (roads/real network); if she meant quad onto her own quad track, new task. Remaining agents: rbprep (light-pass proof running), aimpact, trainsdraw.
- **8800 = 029a6f1c, smoke PASSED** (12 checks, 126 s at 3 cores): the earlier "demand final" timeouts were load (1 core), not T-3257.
- trainsdraw done: T-3264 proposal (notes/T-3264.md, 13-19 h; stage 1 needs her look at a mock). T-3271 filed (t1965 regression since T-2419). Next free id T-3272.
- aimpact done (T-3253 archived). T-3272 filed: real saves' Shinkansen as high speed (rail fact, decided; not her call). Next free id T-3273. Only rbprep still running (light-pass proof); then the batch landing + hand the rebuild to maps-5e.
