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
## Running agents (report to maps-5e), started 11-12 morning
- satfade a5f4536449d4ca90a (Sonnet): T-3278 DONE, uncommitted (basemap.ts, docs/ui-map.md, notes/T-3249.md):
  imagery opacity = slider, drawn map underneath at its 0% look; couldn't reproduce her solid grey in headless
  Chrome. Commit with the next batch; she checks on 8800 after the snapbuild; archive after her look.
- flags2 a97c6cd861102a368 (Opus): T-3279 DONE + archived, uncommitted in maps (flags.ts/json, Flag.tsx, flag.css,
  tools/flags.mjs, 309 flag files, Leaderboard/Community/account edits, checks/t3279-flags.mjs). shonei-server
  488e09a + e878c0f (48 new codes, trainworldFlagSet 2) committed, not deployed; until then the picker hides the new
  codes. Strings to her: flagTip -> "Flag shown after your name on the leaderboards and in Community"; "New York City".
  At the next publish: website about-page credit line (flag-icons, circle-flags MIT; prefecture flags Commons PD).
- jpattract a970ecbf4714f9168 (Opus): T-3280 stage 1 DONE (notes/T-3280.md): resort towns missing from the
  attraction list + attraction trips local only (halve per 10 km, no zone crossing); Hakone ~10x under, Atami ~5x.
  Options A (recommended: 60-100 Japanese resort municipalities with prefecture visitor counts added to Japan's
  long-trip destination weights, beta 2-4 fitted under rail counts, ~6 h, market-only pass over Japan) / B / C / D.
  Asked her; resume jpattract to build on her yes. Ids T-3283/3284 unused.
- lightrail a0244723dad581e04 (Sonnet, 6154): T-3285 light_rail in the real network layer + snapping, rebuild 118
  realgraphs + Berlin (her yes); writes data/packs only after the light pass ends; closes T-3265.
- stations a4afc595a574ff195 (Sonnet): T-3286 + T-3288 DONE + archived in worktree .tw-trees/stations (WASM).
  Land after the light pass ends; main tree's docs/ui-map.md also has satfade's edit: commit main-tree work first.
  Her check: map labels of a built complex (agent judged only the inspector list).
- gaps ac5cdaf84781d40c7 (Opus, read-only): T-3287 design for her: gap regions (Agder, Baltics, eastern Ontario,
  North Korea) and a contiguity rule, dry count, cost; review image t3287-gaps.png. Ids T-3293/3294.
- transfer a03f13c66621b0720 (Opus, worktree .tw-trees/transfer, 6156): T-3295 trips won't change between two
  stops of the same line at a near self-intersection, even with a passage (her TODO verbatim). Ids T-3298/3299.
- dbltrack a24dfb160b1f47d8c (Sonnet, worktree .tw-trees/dbltrack, 6157): T-3296 rebuild single into double track.
  Ids T-3300/3301.
- lhtscope a9bf2515b4ef71c76 (Opus, read-only, 6158): T-3297 how hard a per-save left-hand traffic flip is; notes +
  staged estimate for her.
- Worktrees made from 32ac89b1: their registry/market-rows must be copied from the main tree before pages (told them).
- Her Q: is the high-speed fare fix live? Answer: v0.3.1 has d5911dec (fares in the start country's dollars, long
  trips take HSR again); option A eb23b84b and jpn7's Shinkansen (T-3272) are 8800 only, wait for the next publish.
- Queued: T-3289 freight/all rail as a three-way real network setting (none, passenger, all) -> lightrail after T-3285.
- shonei-server DEPLOYED 15:01Z 11-12 with her word, pushed (e878c0f): caps 45k accounts / 22.5k password / warn
  30k, browsers 400k; place flags (trainworldFlagSet 2 live). Her calls 11-12: no Resend Pro; T-3276 left as is;
  T-3280 option A -> jpattract resumed to build it (hands back a Japan market-only rerun).
- Mine, uncommitted: her string edits (View a copy; "You can save it to keep it."; HSR notice shortened) and the
  cursor tag's Ctrl line removed (tools.ts, en.json, context.json, DECISIONS). Commit with the light pass batch.

## T-3274 + T-3275 rerun: RUNNING since 08:40 local
parentsplit built both (uncommitted: tier0.py NATIONAL_TERTIARY TW 0.0468 from MoE; site_inside: a building inside a
campus outline joins it or is left out). Rerun `--from through --ids artifacts/trainworld/t3274/packs_changed.txt`
(1,041 packs), pid 27492, stdout artifacts/trainworld/t3274/light_pass.log, ~1.5 h. Expected: jiw 0.036, ina
"Lecture Hall" 7,318 -> 33. After: commit, archive T-3274/3275, snapbuild, smoke, push, delete data/work/pre-T3244/.
T-3276 filed (Georgia Gwinnett 25k: area-sized campuses in scaled packs; a lower cap moves 10-24% of students to
near home worldwide, a demand-wide shift: asked her, I recommended leaving it). Next free id T-3302.
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
