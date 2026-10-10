# West and Central Africa build (ng, cm, ao, ga, cg, sn, gh, bf, cd), 2026-10-08

All nine are built in `dist/data/<cc>` (model, tiles, check_model) by one reader,
`wafrica_register.py` with the lists in `wafrica_lines.py`; each `<cc>_sources.md` has a
"Build (2026-10-08)" section. Nothing below has been applied: the builds need none of it.
No rinf.py hook was needed (the Lagos corridor is handled in the reader: see ng_sources.md).

| cc | register lines | km | running | greyed |
|---|---|---|---|---|
| ng | 8 (+ LAMATA Blue Line as an OSM line, 11.8 km) | 808 | 6 / 764 | 2 / 44 (Abuja metro) |
| cm | 3 | 979 | 3 / 979 | 0 |
| ao | 7 | 2,529 | 6 / 2,475 | 1 / 54 (Zenza – Dondo) |
| ga | 1 | 646 | 1 / 646 | 0 |
| cg | 1 | 508 | 1 / 508 | 0 |
| sn | 1 | 55 | 1 / 55 | 0 |
| gh | 3 | 128 | 2 / 108 | 1 / 20 (Adome – Mpakadan) |
| bf | 2 | 495 | 1 / 349 | 1 / 145 (Bobo – Niangoloko) |
| cd | 2 | 377 | 1 / 20 | 1 / 356 (Kinshasa – Matadi, to Kenge) |

## 1. tools/rebuild.py: REGISTER entries

```diff
     "al": "balkans_register:data/raw/rinf/al", "xk": "balkans_register:data/raw/rinf/xk",
+    # West and Central Africa: one reader writes rinf.py's inputs (wafrica_register.py,
+    # lists in wafrica_lines.py). After an extract (with --station-areas): `--clip <cc>`
+    # (also joins track gaps and splits gauge breaks) then `--fill <cc>`.
+    "ng": "wafrica_register:data/raw/rinf/ng", "cm": "wafrica_register:data/raw/rinf/cm",
+    "ao": "wafrica_register:data/raw/rinf/ao", "ga": "wafrica_register:data/raw/rinf/ga",
+    "cg": "wafrica_register:data/raw/rinf/cg", "sn": "wafrica_register:data/raw/rinf/sn",
+    "gh": "wafrica_register:data/raw/rinf/gh", "bf": "wafrica_register:data/raw/rinf/bf",
+    "cd": "wafrica_register:data/raw/rinf/cd",
 }
```

Without it, rebuild.py's default (`rinf:data/raw/rinf/<cc>`) would run rinf.build on the last
conversion without re-converting; it would still work, but the reader's own steps (operator,
colour and kind from the list, served sections, twin stations) would be skipped. Each model
takes under 30 s, tiles a few seconds: MINUTES `1` for each if wanted.

## 2. borders.py: nothing

No passenger train crosses a border in the region (wafrica_survey.md; bf's line stops at
Bobo-Dioulasso, the greyed Bobo – Niangoloko ends at Niangoloko, short of Côte d'Ivoire).
No line was built to a border point.

## 3. tools/build_regions.py, then neighbours

Run it to put the nine in regions.json. None of their built neighbours (none are built
except through these nine) needs a rebuild: there are no crossings.

## 4. HANDOFF.md "Start here" table row (suggested)

| ng, cm, ao, ga, cg, sn, gh, bf, cd | hand line lists through rinf.py (`wafrica_register.py`, lists in `wafrica_lines.py`; extract with `--station-areas`, then `--clip`, `--fill`) | 8, 3, 7, 1, 1, 1, 3, 2, 2 | 808, 979, 2,529, 646, 508, 55, 128, 495, 377 | `<cc>_sources.md`, `wafrica_survey.md` |

## Files the agent wrote

New: `wafrica_register.py`, `wafrica_lines.py`, `rinf_countries/{ng,cm,ao,ga,cg,sn,gh,bf,cd}.py`,
`data/raw/rinf/<cc>/{sections,points,names}.json`, `data/proc/<cc>/`, `dist/data/<cc>/`,
`dist/data/<cc>.pmtiles`, this note. Edited: `check_model.py` (REGISTER for ng, ga, cg, sn,
bf, cm, ao, cd; gh has no published per-line figure, only a comment), the nine
`<cc>_sources.md`, `wafrica_survey.md`. No rules/<cc>.py and no colours/<cc>.csv (the
three line colours set are in the lists, marked picked in ng_sources.md). All nine .pbf
extracts are deleted.

`--clip` repairs in data/proc/<cc> only: synthetic link ways (ids from -8.2e12), copied
nodes where two gauges met (ids from -8.3e12, coords.npz rewritten), and `--fill` halts (ids
from -8.4e12, tagged `source=wafrica_register timetable`; nafrica uses -8.0e12, pk -8.1e12).
