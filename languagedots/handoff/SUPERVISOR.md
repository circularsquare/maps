# Supervisor handoff — 2026-10-07 (project set aside)

Session `da1b1b09`. Anita: "its in a very good state i think", leaving it for a while after this
publish. Everything below the line is the 2026-10-06 handoff, still true unless said here.

- **Deployed 2026-10-07** (R2 data + index.html copied to the website repo; she commits/pushes).
  Archive 513 MB (tiles now stored in position order; ties drawn by markHash in the viewer).
- **Viewer added:** colour picker on legend dots (religiondots' popover; `copyOverrides()` in the
  console gives fragment lines), language search under the legend, double-click a tooltip language
  to select it, round Size stops, Auto needs 65% of the centre-weighted visible land (narrow
  countries need a deep zoom: fine by her), world copies off while the world fits on screen, pies
  read the dots' decode cache (~4x faster).
- **Data added:** ci foreign residents, ss's three hatched states, bd Sylheti/Chittagonian/Rohingya,
  India's Barak Valley and North Tripura as Sylheti, cd city Swahili/Lingala, ru settlement
  placement, pk tehsils, om/sn/ml placement, qa/ae expat pools by sex. Moldovan under Romanian,
  Dari and Tajik under Persian (regroup.txt). One line each in runlog.md.
- **Next, if picked up:** followups.md "weakest places, queued" is the list; "speed and size
  catalog" has what is still open there (archive per Aggregation step, side-file trims, caching).
- **2026-10-08, published:** Draw as "none"; long-press merge box in pies mode; tooltip place
  line (`placeLine`: nearest basemap place by class and rank, else Natural Earth province from
  `admin1_shapes.geojson` built by `tools/make_admin1.py`, then the country); RTL text plugin and
  a `name_en` guard (Türkiye read "T") in both languagedots and religiondots. The same two label
  fixes were added to the other MapLibre maps (flights, cityhistory, citybrowser, noritetsu,
  neighborhoods, chinaethnicity, ancestrydots, traincounts, the riders maps) but NOT published.
- **Her open calls:** Italian (and French d'oïl) regional languages under their roof language: she
  was unsure; left ungrouped. qa.csv predates the latest home mixes (re-running qa_build.py
  changes counts).

---

# Supervisor handoff — 2026-10-06 (afternoon)

Written by the ld-super session `5d7dac7e` as it handed back. Read this, then `runlog.md` (one
line per result, newest last; the 2026-10-06 lines cover this session), `followups.md`,
`ask/OPEN.md` and `python tools/claim.py`. The 2026-10-05 handoff is fully applied and superseded.

## State at handoff

- **All 217 countries drawn**, queue empty, nothing claimed, nothing waiting for the build tail
  (last tail 2026-10-06 ~15:00, ok). Brunei drawn today from race by district + estimates.
- **Every country has `parts`** (the per-source Data display) and shortened `how`/`grain`/`gap`/
  `note_public`. Cut detail lives in `sources/<cc>.md` (13 s-z countries recovered from a
  transcript after the sweep skipped that check).
- **The tree is regrouped by a move list**: `taxonomy/regroup.txt`, applied in build.py,
  countries.py `load_one` and audit_groups.py; data files keep old ids. See `taxonomy/GROUPING.md`.
  New Bantu/Austronesian languages need a regroup.txt line. Arabic is a group of 21 with an
  "Arabic (variety not given)" leaf; ~60 standard languages are groups over their Glottolog
  dialects (German includes Swiss German).
- `tools/build_tail.py` now also runs `not_drawn.py` (the hatching). `tools/audit_groups.py`
  lists named people on group nodes; run it after any tree change.

## Anita's rulings today

- Asks needing a download carry the URL and target folder at the top (AGENT_BRIEF §7).
- Nigerian and Cameroonian Pidgin drawn from estimates; Isan (and Northern/Southern Thai) from
  WVS 7; Low German at "speaks very well"; Swiss German its own language in ch; Bavarian,
  Alemannic etc. stay German in de/at; dialects grouped under their main language.
- Neighbour-pull softening: Philippines only, weakened (big lowland languages only, fades in the
  mountains). China uses atlas polygons and, now, inter-province migrants by home province.
- chinaethnicity is a dead project: leave its estimated county totals (Shenzhen 10.5M vs 17.6M).
  Open question to her: correct cn's county totals inside languagedots from helper1m's panel.
- Auto country selector: entry as built, release as religiondots (Bangladesh hard to auto-pick is fine).

## China, later the same afternoon (sources/cn.md §8-§11)

- County totals rescaled to the 2020 census in chinaethnicity's 15 estimated provinces
  (`sources/cn_totals.py`; chinaethnicity itself untouched, Anita: dead project).
- Inter-province migrants on their home province's dialect mix (`cn_migrants.py`, 119.4M).
- Putonghua shift from CLDS 2016 after-work language by prefecture x migrant status, ten southern
  provinces, 43.6M moved (`cn_putonghua.py`; Anita approved aggregate tabulation, non-commercial);
  checked against WVS 2018 home language.
- MCPDict dialect points (MIT): capped shares for groups present but missing from a county, and
  within-county placement toward each group's points (`cn_mcpdict.py`); carved districts take the
  1987 polygons (Pingshan, Guangming now Hakka).
- Viewer: about panel open on load and closes on outside click, Anita's text; religiondots' phone
  fixes ported; DOT_KM2 3 -> 3.9; no "language not named" anywhere.

## Paused 2026-10-06 late (Anita went to sleep)

- Deployed to R2 (languagedots 572 MB archive, religiondots) and **live at anita.garden**
  (Anita pushed the website repo 2026-10-07; working). Later updates: rerun `tools/deploy.py`,
  then she commits and pushes.
- Since that copy, index.html changed locally: ring opacity floor 0.32 (copied), "Mother tongues"
  heading removed, tile-blanking fix (pies built per tile with parent/child stand-ins) and phone
  info panel closed by default. All of it is now in the website copy (deploy.py rerun 2026-10-07,
  data unchanged). Left: check whether religiondots' viewer has the same blanking gap.

## Publishing

- `tools/deploy.py` (`--dry-run`, `--verify`) and COMMANDS.txt DEPLOY are ready; nothing uploaded.
  Upload only on Anita's explicit go. (A 2026-10-06 rule to redeploy religiondots with every
  languagedots deploy was dropped by Anita on 2026-10-08: deploy each map only when it changed.)
- `country_shapes.geojson` must ship with the data (`tools/make_shapes.py` rebuilds it).

## Open

- Brahui in Afghanistan at 200k (southern Helmand 64% Brahui): asked Anita whether to halve.
- Guangxi far off its gazetteer (Hakka 3.4M vs 7.0M, Pinghua 2.5M vs 5.0M, Yue 20.9M vs 15.1M):
  asked Anita whether to rake fully. Hainan Putonghua share (10% vs WVS 52%): asked.
- cn Luhe (Shanwei) mis-filed as Yue in cn_dialect.py (followups).
- New group colours (Guthrie zones, Polynesian, Micronesian, Arabic group) generated, unreviewed.
- followups.md: India's layer crosses the LoC near Poonch; Karabakh resettlers counted twice in az;
  nl Belgium-born record contradiction; si Bosnian/Serbian identical column; Bangkok Isan ~1%;
  cn Dai/Tibetan and dz Amazigh could be split by place; ask 020 (South Sudan) still postponed.

## Things that bit today

- Sweep agents fanned out to nested helpers; their reports do arrive, but check `parts=` on disk.
- An agent shortening text must check each cut fact is in `sources/<cc>.md` first; country files
  are untracked and have no backups (transcripts under ~/.claude/projects/.../subagents/ do).
- The permission check refuses agents' edits to shared taxonomy files unless Anita asked for that
  change in her own words; do not run a refused step for them, surface it.
- A build tail started while agents scatter can miss their dots; queue a second tail behind it
  (`build_tail.py --id <sid>-2` waits for the lock).
