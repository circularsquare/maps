# RINF lines in pieces: state at hand-off (2026-10-05)

Agent: the RINF-pieces investigation (Anita's "really fragmented train in austria", the
Tauernbahn). No real rebuild was run and nothing in dist/ was written. Trials only, in the
agent's scratchpad. The dev code and tools are copied to `handoff_notes/rinf_pieces/`.

## The Tauern Railway: why it was in four pieces

RINF itself lacks three stretches of ÖBB route 222 01. Our fetch (`data/raw/rinf/at`,
2026-09-30) has 26 sections of 22201 in four connected pieces (`rinf_ids`
`22201#1..#4`). No section under any line id touches:

- Mühldorf-Möllbrücke (AT02126) to Pusarnitz-Süd (AT05803). Pusarnitz itself is not a RINF point.
- Markt Paternion (AT02132) to Paternion-Feistritz (AT02133).
- Gummern (AT02135) to Abzw Gu 2 (AT90252), towards Villach.

The cause is not drop_unridden_sections, rejected traces, station placement or the GTFS check.
OSM's route=tracks relation 3292769 ("222 01", 828 ways) covers the whole line, holes included.

Not checked: whether RINF has these sections without `era:inCountry`, which Q_SECTIONS requires.
Checking needs a few SPARQL rows (sections whose opStart/opEnd is AT02126), which counts as a
new download, so it was not run. Worth asking first. A fetch-side fix would be cleaner than
the build-side fill below. 59 of Austria's 160 RINF ids are in several pieces, which looks
systematic.

## Cross-country measurement (baseline = current code, matches shipped dist)

Every RINF-read country: **602 of 5,642 register lines in more than one piece; 80,412 km on
them; 34,667 km outside each line's biggest piece.** Causes (a line can have several):

| cause | lines | where |
|---|---|---|
| RINF hole: the line's own RINF sections do not connect | 130 | at 45, de 42, cz 17, be 7, nl 5, hu 3, sk 3, bg 3, pl 2, ro 2, hr 1 |
| reader: sections left out by rinf.build (rejected/untraceable trace, unplaced point) | 467 | ru 303, ua 62, in 19, kz 15, de 9, it 9, pl 9, by 6, cz 5 ... |
| drop-gap: build_model dropped a junction-ended section between two pieces | 11 | it 5, pt 1 (Linha do Sul Pinheiro - Grândola Norte 36.9 km), sk 1, ro 1, uz 1, ru 2 (0 km) |

Russia and the other tariff-guide readers are almost all one bug. ru_register gives each
"clone" junction (`<esr>@<section>`, used so a long stretch answers to OSM's routes) a 0 km
link to its stop. build_model drops that link as unridden, so the line breaks at every clone
whose stretch was kept. Лена-Восточная — Хани came out in 14 pieces. ru_sources.md says
"both vanish in the build", but nothing rejoins the stretch to its stop.

The other "reader" cases are mostly track missing from the OSM extract ("no path": Poland's
108 and 201, India's breaks, Russia's Карталы - Никель) or RINF lengths no trace can meet.

The 15 worst at baseline are all ru apart from kz Бейнеу — Жанаозен (Лена-Восточная — Хани
848 km outside, Сосногорск — Воркута 544, Куэнга — Бамовская 534...). Full table:
`handoff_notes/rinf_pieces/causes_base.txt`.

## The fix, in three parts (dev copy: `handoff_notes/rinf_pieces/rinf_dev.py`, diff vs repo `rinf_dev.diff`)

**A. `fill_holes` (in rinf.build, gated by COUNTRY `fill_holes: True`).** After a line's
sections are built, loose ends of different pieces are joined over the line's own OSM track.
A join is the shortest trace preferring the ways of the line's numbered OSM relation (pass 2's
`own`). It must be at most 25 km and at most 1.5x the crow-fly distance + 1 km, at least 90%
on the line's own relation, and at most 30% on its other sections. The nearest pair is joined
first, and the step repeats until nothing joins. Filled sections have chain = traced km, so
check_model stays neutral. Stations an OSM train route stops at along a fill become stops
(split_at_osm_stops), e.g. Pusarnitz. A line with no numbered relation is never filled.

**B. Link folding (new `rinf.split_pieces`, the build_model hook, runs after the drop).**
rinf.build records each 0 km register section between a stop and a junction within 50 m
(`GROUPS[lid]["links"]`). After build_model has judged the stretches, the junction is renamed
to its stop in sections, geometry keys, highspeed_sections/chain, display and
`state["sec_ways"]`. This only acts where such links exist, so it is a clearly scoped rule
with no COUNTRY key.

**C. Bridges (`_bridge`, gated by COUNTRY `bridge_pieces: True` or a dict of pieces.Rules
settings).** pieces.bridge_gaps over the passenger track graph. A way's name is the register
line holding it (reg_ways); routed track no register line holds is in the graph unnamed. A
bridge's km is added to km_official. Nothing is split. **Written but never run**: the
intended targets are pt (Linha do Sul), it (5 drop-gaps, likely freight bypasses round Udine,
Foggia, Palermo), sk, ro. pieces.py itself is unchanged.

The dev copy also has two trial-only items to remove before landing:
- `ROOT = Path(r"C:\...\noritetsu")  # DEV COPY` (restore `Path(__file__).resolve().parent`).
- The env switches `RINF_FILL_HOLES` (comma list of cc) and `RINF_FILL_DEBUG` in `fill_on` and
  the "hole left" log. Gate on `conf.get("fill_holes")` only.

## Files changed in the repo

- `rinf.py` (shared): **only a no-behaviour-change stash**. `GROUPS` (per line, the km of each
  connected piece of its RINF sections), `_component_km()`, and `GROUPS.clear()` at the top of
  build(). Everything else is in the dev copy, not landed.
- `pieces.py`, `rinf_countries/*`, every `*_sources.md`: unchanged.
- A stray `noritetsu/base/` folder of trial output was created by mistake and has been moved
  out (nothing left in the project).

## Trial results (fill on in every country trialled, folding everywhere; baseline -> dev)

| cc | register lines in pieces | register km | notes |
|---|---|---|---|
| at | 47 -> 10 | 4,341.6 -> 4,657.9 | 109 gaps filled over 349 km, all 96-100% on the own relation; **Tauernbahn whole**, 100.3 -> 114.5 km, Pusarnitz a stop; km outside biggest piece 1,111 -> 181; 4 unnumbered lines renamed ("first - last" ends moved) |
| de | 50 -> 43 | 31,382.5 -> 31,439.9 | 13 fills (77 km), several over other managers' track at "DB-Grenze" ends (2950's Dissen - Hörne 22.7 km, the Haller Willem; 6663 Adorf - Zwotental 11.5); 4721 Untertürkheim – Nürnberger Str (3.2 km) **gone**, check why; most holes left are routed over other lines (own share 0-60%) or no path |
| ru | 304 -> 36 | 77,913.4 unchanged | 1,145 clone links folded on 324 lines; km outside 26,135 -> 1,642; 1,142 clone junction ids gone (junctions, not stops) |
| ua | 62 -> 29 | 15,734.0 unchanged | folding; 115 clone ids gone |
| cz | 22 -> 11 | 9,157.0 -> 9,187.0 | fills |
| pl | 11 -> 10 | +6.1 km | |
| be | 7 -> 7 | +3.7 km | one line gained a fill |
| in, it, es, se, nl | unchanged | unchanged | |

Not trialled (the run was stopped for this hand-off): hu pt si sk bg ro fi lt lv ee hr gr lu
dk ie by md id nz za rs ba me mk al xk kz uz kg tj tm ge am az xa ma dz tn eg. Of those, by, md,
kz, uz, kg, tm, ge, az (tariff-guide clones) should behave like ru and ua. hu, sk, bg, ro and hr
have RINF holes.

Caveat: foot.json differs base -> dev even in countries whose lines did not change (in, it, es,
nl). Probably dist/regions.json or other shared input changed between the two runs, which
were about an hour apart. Rerun the baseline beside the change before trusting foot diffs.

## Diffs needed in files this agent may not edit

The build_model hook `split_pieces` is found on the *register module*. For `rinf:` countries
(all EU ones, ru, ua, by, md) that is rinf.py, so it runs once rinf.py has it. The wrapper
readers need one line each to expose it (the harness injected it for trials):

```python
# casia_register.py, caucasus_register.py, balkans_register.py, nafrica_register.py,
# in_register.py, id_register.py, nz_register.py, za_register.py (after `import rinf`):
from rinf import split_pieces, LINE_PIECES  # noqa: E402,F401  (build_model's hook)
```

No build_model.py or ownership.py change is needed.

## Next steps

1. Optionally ask Anita to allow the small RINF query for the missing ÖBB sections (inCountry?).
2. Land A and B from `rinf_dev.py` into rinf.py: drop the dev ROOT and env switches, keep the
   GROUPS stash, add a docstring line for `fill_holes` / `bridge_pieces`. Set `"fill_holes":
   True` in rinf_countries/at.py, de.py, cz.py, pl.py, be.py, and in any untrialled country
   whose trial looks right.
3. Trial C (`bridge_pieces`) on pt, it, sk, ro. Look at Germany's 4721 (gone) and at at's 10
   leftovers (rerun with `RINF_FILL_DEBUG=1`; dry run: `python rinf.py --dry <cc>`).
4. Trial every RINF country with tools/ab.py (fill set per country; link folding everywhere).
   For wrapper countries, first add the one-line export above.
5. Rebuild (managing session): at, de, cz, pl, be, ru, ua, plus by, md, kz, uz, kg, tm, ge, az
   and any hole country that trials well. Then build_regions.py.
6. Record in at_sources.md and ru_sources.md (clones now folded), and HISTORY.md.

## Tools (handoff_notes/rinf_pieces/)

`harness.py <cc> <abs out>` is a trial build_model run that records reader / pre-drop /
post-drop sections (`stages.json`). With `HARNESS_DEV=<dir>` it imports rinf.py from that dir.
`driver.py <out> <jobs> cc...` runs it through tools/slot.py. `causes.py <base> [cc]` gives the
cause table and worst lines; `why.py <base> <cc>` the reject log lines; `abdirs.py <base>
<dev> cc...` is ab.py's comparison between two trial folders; `shipped.py dist` measures shipped
data. The harness scripts hard-code the scratchpad and noritetsu paths: fix `ROOT` and
`HERE` before reuse.
