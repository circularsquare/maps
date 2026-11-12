# Orchestrator handoff, trainworld

Orchestrator: session 441e3071-8b6e-4109-9ab2-1339e17b92be, name maps-5e [3a1e06], since 2026-11-12 04:41 UTC
Updated: 2026-11-12 11:15 UTC (06:15 local; she is asleep)

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
- **8800 = 6a325b9c** (the light pass's data), smoke PASSED against 8800 (12 checks, 113 s at 4 cores; real save
  12.58M rail trips a day).
- Master 6a325b9c, pushed 06:25 local.

## World rebuild light pass: DONE (her go, overnight 11-12)
Batch 20261112-025728, 3,919 tier 0 packs, ~3 h wall at 10 cores (T-3244 archived with times). Fixed on the way:
an invalid boundary in campus split, an empty university column in tier0, appended logs counted as failures, freeze
deadlocked on the run's own packs lock (now `--locked`). Gate: 346 FAIL as before + 1 new (jiw university_rate 0.36,
T-3273); load: all 3,983 load, 3,481 with other legs. Frozen as data/frozen/20261112-060723. Backup of the old packs
in data/work/pre-T3244/ (delete once the new state is checked clean, e.g. after T-3273's rerun and a smoke).
To tell her: Atlanta's airport legs 0.98 -> 0.44 of published (DB1B O&D share).

## T-3273 rerun: DONE 08:23 local, committed 440a601f (archived); **8800 = 440a601f, smoke PASSED** (12 checks, 117 s)
parentsplit resumed on T-3274 (Taiwan MoE tertiary figure, my call) + T-3275 (freed students onto building-sized
sites): builds code only, hands back a pack list + one rerun command. Next free id T-3278 (T-3276/3277 to it).
Old text, for the record:
parentsplit (Opus, a4fc96c86a876dea2, done) built pipeline/campus_pools.py (world table data/raw/tier0/campus/pools.json),
tier0.university_sizes on it, wikidata_prefill builds it, open universities and university farms/forests out; all
uncommitted (notes/T-3273.md). Rerun: `world_run.py step light_pass run --jobs 10 --from campus --ids
artifacts/trainworld/t3273/packs_changed.txt` (828 packs), runner pid 42656, stdout artifacts/trainworld/t3273/light_pass.log,
~1.5 h. Expected: Galveston 94k -> 0, Kennesaw Marietta 99.9k -> 7.1k, IU Indy 79k -> 14k, jiw 26k -> 6.5k (still
0.090, just over the band: T-3274 Taiwan national tertiary figure from its MoE, my call, not hers: the same rule
every World Bank country already has; resume parentsplit for it with T-3275 and one more Taiwan + affected rerun); T-3275 filed (freed students land on
area-sized single buildings in scaled packs: ina "Lecture Hall" 7.3k). After: commit, snapbuild 8800, smoke, then
delete data/work/pre-T3244/.

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

Next free id T-3278.

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
