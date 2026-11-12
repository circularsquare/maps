# Orchestrator handoff, trainworld

Orchestrator: session 441e3071-8b6e-4109-9ab2-1339e17b92be, name maps-5e [3a1e06], since 2026-11-12 04:41 UTC
Updated: 2026-11-12 ~06:50 UTC (01:50 local; she is asleep; maps-18 wound down at ~06:10 UTC)

Read trainworld/CLAUDE.md and docs/orchestrator.md first. maps-18's 11-11/12 log is in git history (handoff.md at
348ef2c2).

## Live
- **Live = v0.3.1** (tw/release-031 head 71009033, worktree .tw-trees/hotfix031; data 2026-11-11b; website cc14a89).
  A deploy yes is not a text ack: changelog/player text waits for "text acknowledged".
- **Server** (shonei-server, deployed 04:49Z 11-12 with her word, pushed): caps 30k accounts / 16 GB games / 3
  backups (T-3251); efb2cd6 best game per board (T-3248); 020ba08 likes tiebreak + her 60-min exemption (T-3258,
  T-3259; SHONEI_COMMUNITY_NO_WAIT set, env backup shonei.env.bak-2026-11-12); e98ae76 flags server (T-3269, config
  trainworldFlags true; the game's half shows from the next publish).
- **Off-site backup** (T-2918): nightly 03:26Z to R2 bucket shonei-backups, 7 kept; key at ~/shonei-backup-key.txt.
- **Cloudflare**: cache rules "trainsworld assets", "trainworld data", and (11-12, her yes) "noritetsu data"
  (data.anita.garden/noritetsu/data/*, 2 h edge TTL, after "data bypass"; DECISIONS 139ba331). Ops token at
  ~/cloudflare-ops-token.txt. **Daily**: tools/ops/check_resources.py; look at the noritetsu rule's hit rate in the next one.

## 8800 and master
- 8800 = dd7b6fab (rebuild batch), smoke PASSED (12 checks, 167 s).
- Master: ef16dd6f (T-3218 attendance 0.8 in sim, WASM rebuilt in the main tree; market_pull.json "on": true). Not
  on 8800; snapbuild after the light pass.

## World rebuild light pass (her go: overnight 11-12, 10 cores); maps-5e owns it
Checklist: (1) campus band fetch pid 45568 (log artifacts/trainworld/t3244/campus_fetch.log, 21/40 at 01:18 local,
overpass 502/504s; failed bands: rerun `world_run.py step campus_outlines fetch`): WAITING; (2) pull "on": true:
done ef16dd6f; (3) T-3218 0.8: done ef16dd6f; (4) no sim/ landings during the run (hsrsave works in a worktree).
T-2112 seams not a step; T-2627 done.
Command (maps root, detached): `venv\Scripts\python.exe -I trainworld\pipeline\world_run.py step light_pass run --jobs 10`
(runbook docs/world-run.md "The light pass"; ~2-2.5 h; logs artifacts/trainworld/light_pass/<batch>/; failed packs:
`--from through --ids <failed.txt>`, then `--from cross`). Baselines artifacts/trainworld/t3244/baseline/ (gate
baseline: 346 packs FAIL today, 345 joint_borders; report only new FAILs).
After it: campus figures (keep T-3218?), UBC (T-3213/3214), Atlanta legs 0.98 -> 0.44 of published (DB1B O&D share;
tell her); close T-3240/3244/2896/3214/3216/3226/3218 as their checks pass; snapbuild 8800; smoke.
Side effects already there: rbprep's sample appended to data/raw/tier0/wikidata_parents.json; staged campus parts
for van/hfd/mex/rio.

## Running agents (report to maps-5e)
- hsrsave a3bf5409c401bf570 (Sonnet, worktree .tw-trees/hsrsave, Vite 6150, 3 cores): T-3272 real saves mark
  high-speed lines; writes rebuilt saves to scratch only; I copy them into data/ after the light pass, then land.

## Waiting on Anita (morning)
- Test satellite at 10/40/100% light and dark.
- Strings to review: Community View, track tool hints (incl. "Hold Ctrl to draw straight"), zoom buttons, flags,
  satellite ("Map", "Satellite", hover "Low resolution outside Japan and the US"). Collect old -> new from
  notes/T-3257.md, T-3260.md, T-3268.md, T-3269.md, T-3249.md for the morning message.
- Translation batch for all new strings before the next publish.
- T-3264 trains drawn through each other: stage 1 needs her look at a mock (notes/T-3264.md, 13-19 h total).
- T-3265 her question: all light rail (railway=light_rail) in the Real network layer and snapping, plus rebuilding
  118 realgraphs (~1 h)? notes/T-3265.md.
- T-3261: she may have meant quad onto her own quad track (built: quad on snapped routes); new task if so.
- Flags extra list (notes/T-3269.md).
- Longer-term TODO items not filed: leaderboard map view, brainstorm, intercity-only mode, more languages, offline,
  advanced quad track.

Next free id T-3273.

## Her calls 11-10/12 (for the record; detail in DECISIONS and ARCHIVE)
11-12: option A stays; wealth tiers parked; Community Biggest left alone; China pull 0.6; Poland slight; undershoot
rule for all demand fits; hex terrain back to squares; best game on every board; zoom buttons + flags now.
11-10/11: single track two-way by default, margin 240 s; double track lanes in one piece; flyover ramps 4%; train
count cap = running + 1; HSR and tram quad yes (tram quad no street running); buses and trams on any road; snap to
roads; text copied word for word from her message needs no review; rebuild speedups 1-3 yes; R2 purges via the
guarded script; 4-6 min rebuilds "not so bad"; Privacy · Terms in the signed-in panel.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- Cores: ledger cap 14 (`python cores.py list`).
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
