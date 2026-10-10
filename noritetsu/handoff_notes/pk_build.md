# Pakistan build (pk), 2026-10-08: shared-file changes for the managing session

pk is built in `dist/data/pk` (model, tiles, check_model passing; pk_sources.md "Build").
Nothing below has been applied. The pk build itself needs none of it; (1) is for batch
rebuilds, (2) for the Taftan crossing, (3) to put pk on the map.

## 1. tools/rebuild.py: REGISTER entry

```diff
     "za": "za_register:data/raw/rinf/za",
+    # Pakistan: a hand list through rinf.py (pk_register.py, pk_lines.py). After an extract
+    # (with --station-areas): `python pk_register.py --fill` (stations OSM lacks, from
+    # Wikidata, into data/proc/pk/stops.pkl).
+    "pk": "pk_register:data/raw/pk",
```

and in MINUTES `"pk": 1` (model 40 s, tiles 7 s).

## 2. borders.py EXTRA: Koh-i-Taftan - Mirjaveh

```diff
     ("eXARUPSOU", 40.008443, 43.393733, ["ru", "xa"]),
+    # Koh-i-Taftan - Mirjaveh (Pakistan - Iran): where OSM's track (way 1457452999) crosses
+    # OSM's boundary (way 440799384), Overpass 2026-10-08, 550 m west of Taftan station.
+    # pk_register's Spezand - Koh-i-Taftan (greyed: no passenger train since February 2020)
+    # ends here; ir_register leaves Zahedan - Mirjaveh out, so Iran's side has no line yet.
+    ("eXIRPK1", 61.551797, 28.974554, ["ir", "pk"]),
 ]
```

Gated on the two countries; pk's line already ends at a junction `eXIRPK1` at this point, so
after it lands pk needs only a rebuild for the point's neutral name ("Iran – Pakistan border").
ir: its register has nothing there; a rebuild changes nothing unless ir's OSM half reaches it
(the Zahedan Mixed's OSM route is a named train in ir; ab.py on ir would show).

## 3. tools/build_regions.py, then neighbours

Run it to put pk in regions.json. Neighbours: India (in) has no running crossing (Wagah -
Attari and Khokhrapar - Munabao carried nobody since 2019; pk's Lahore - Wagah and Mirpur Khas
- Zero Point are greyed and end short of the border), Iran (ir) after (2). Afghanistan and
China are not built. No neighbour rebuild is needed for pk's first build.

## 4. Headless check (after build_regions)

Not run: the app loads pk only after build_regions. Lines worth probing with
tools/line_100_probe.js: "Taxila–Havelian" (55 km, 3 stations), "Sher Shah–Kot Adu" (72 km,
chained), "Karachi–Peshawar Line" (1,678 km, 190+ sections).

## Files the agent wrote

New: `pk_register.py`, `pk_lines.py`, `rinf_countries/pk.py`, `rules/pk.py`,
`colours/pk.csv`, `data/raw/pk/{sections,points,names,wikidata_stations}.json`,
`handoff_notes/pk_build.md`. Edited: `check_model.py` (REGISTER["pk"], KNOWN["pk"]),
`pk_sources.md` ("Build (2026-10-08)"). `data/proc/pk/stops.pkl` carries 69 Wikidata stations
(ids <= -8,100,000,000,001, `source=pk_register wikidata`) after `--fill`. The .pbf is deleted.
