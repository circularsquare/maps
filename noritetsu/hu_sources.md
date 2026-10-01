# Hungary register sources (built 2026-09-30)

What the Hungarian build reads, where each piece came from, and what is still wrong with it.
Hungary is built with `rinf.py`; how the reader works is in its docstring, and the per-country
table is `rinf_countries/hu.py`, whose docstring says how the ids become line numbers and names.
Downloads live in `data/raw/rinf/hu/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch hu                                              # RINF + Wikidata, ~20 s
curl -L -o data/raw/hungary-260929.osm.pbf https://download.geofabrik.de/europe/hungary-260929.osm.pbf
$env:OSMIUM_POOL_THREADS = "2"; python extract.py --region hu --pbf data/raw/hungary-260929.osm.pbf   # 40 s; delete the .pbf after
python inspect_region.py --region hu
python build_model.py --region hu --register rinf:data/raw/rinf/hu        # 35 s
python build_tiles.py --region hu                                         # 15 s
python check_model.py --region hu
python rinf.py --dry hu          # the reader alone, with its full log (15 s)
```

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-09-30 (`sections.json`: 1,977 sections; `points.json`: 1,836 points). 273 line ids,
  6,592 km, every section under one infrastructure-manager code, `HU55_IM`.
- **OpenStreetMap**, Geofabrik `hungary-260929.osm.pbf` (data to 2026-09-29), ODbL: track,
  stations, 697 route relations and 359 `route=railway`/`route=tracks` relations. MÁV's
  `route=railway` relations carry the line number as `ref` ("30", "120a", "113 (1)").
- **Wikidata** (`wikidata.json`), CC0: items with a route number (P1671) and P17 Hungary, with
  their Hungarian labels and lengths (P2043). The labels are hu.wikipedia's article titles
  ("Pécs–Mohács-vasútvonal") and give the route in each line's English name. The lengths are
  the articles' infobox figures and are what `check_model.REGISTER["hu"]` compares against.
- **hu.wikipedia**, read directly for lines 1 and 30 to see what the infobox figure covers
  (https://hu.wikipedia.org/wiki/Budapest–Hegyeshalom–Rajka-vasútvonal,
  https://hu.wikipedia.org/wiki/Budapest–Murakeresztúr-vasútvonal).
- **Világgazdaság**, "Vasútvonalak üzemeltetését vette át a MÁV-tól a GYSEV", July 2025
  (https://www.vg.hu/vilaggazdasag-magyar-gazdasag/2025/07/vasutvonal-mav-gysev-szemelyszallitas):
  the list of line sections MÁV handed to GYSEV on 1 July 2025, which is where the operator
  of each line comes from (below).

## What RINF carries in Hungary

The whole MÁV and GYSEV standard-gauge network, including closed lines and freight branches,
all filed under `HU55_IM`. It does not carry the Budapest metro (M1-M4), the HÉV suburban lines
(H5-H9), any tram, the Szeged - Hódmezővásárhely tram-train on its tram track (RINF has only its
2.8 km street section in Hódmezővásárhely, as line 131), the Children's Railway, or any narrow-
gauge forest or economic railway (Szilvásvárad, Lillafüred, Királyrét, Zsuzsi, Balatonfenyves,
Kecskemét...). All of those stay OSM lines: 4 metro lines; 8 light-rail lines (HÉV H5-H9, the
tram-train as "1" and "1A", the Skanzen railway); 53 tram routes; the Sikló funicular; and the
narrow-gauge lines among the 151 OSM train routes.

**Line ids** (`hu.py` docstring): "30", "100/1", "20/2" are MÁV lines 30, 100, 20; a letter
belongs to the number ("5/a" is 5a, "1D" is 1d); 200-299 are connecting curves and 300-499
freight lines, which keep their own number. The rule is certain, with ten fixed exceptions: RINF's
120/1 and 120/2 are public line 120a (Rákos - Újszász - Szolnok), 125/a is the Battonya end of 125,
and GYSEV's letter ids for stubs (15D, 15R, 8G, 8R) fold into their line. Letting OSM's
relations overrule the id did harm: where only a stub of a closed line traced, it lay on a
neighbour's relation and joined it (22 became 25, 64 became 65, and curves 207, 223 and 347
joined line 1).

**Lengths leave out stations.** MÁV's RINF section lengths are the open line between station
limits: line 1 is 154.4 km in RINF, 189.2 traced, 191 published; Szolnok - Szajol is 5.5 km in
RINF for 9.8 of track. GYSEV's lines in the same file (15, 16, 21) match their traces at 1.00.
So the RINF comparison in `check_model` reads a median 1.15 by construction, and the published
figures below are the check. For the same reason `hu.py` sets `tol_abs` 1.0 (rinf.py's default
is 0.3 km): at 0.3, Tatabánya - Bánhida on line 12 (3.34 km traced, 2.61 in RINF), which the S12
runs over, was rejected. At 1.0 one Hatvan yard curve (262c, 0.4 km in RINF, 2.3 traced) is
newly rejected; it is freight and would be dropped as unridden anyway.

**Operators.** RINF gives GYSEV no code of its own, and its "pvh." points (pályavasúti határ,
the boundary between the two managers) are from before July 2025. `hu.py`'s `im_of` names GYSEV
for lines 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 524 and 1d, and the curves
and freight branches OSM gives operator=GYSEV (266, 292, 346, 350-354): its network before 2025
plus the lines it took on 1 July 2025. It also took the Murakeresztúr ends of 30 and 41 and
stubs of 13 and 27, which leave MÁV the larger share of those lines. 21 register lines are GYSEV,
112 MÁV. A section's validity date is no guide: line 8, all GYSEV, has Győr - Csorna dated
2023-03-28 and Csorna - Sopron 1900-01-01.

**Names.** "30-as vasútvonal" in Hungarian, with the suffix vowel harmony gives the number;
"Line 30 (Budapest–Murakeresztúr)" in English, the route from the Hungarian Wikidata label. The
English labels were not used: several are wrong (65's reads "Villány–Magyarbóly", which is 66;
127's "Oradea–Kótpuszta"). Lines with no Wikidata item are "Line 400", "Line 264j". Line 26 is
"Tapolca–Ukk" because Wikidata's only item numbered 26 is that part; the whole line is
Balatonszentgyörgy - Tapolca - Ukk (Wikidata files it as "26B"). Line 1d is "Pozsony–Hegyeshalom",
the item for the Bratislava line whose Hungarian part it is.

No `colours/hu.csv`: MÁV and GYSEV publish no colours for their numbered lines, and the suburban
S/G/Z colours belong to services, which stay OSM lines with OSM's colours.

## Counts (2026-09-30)

- 133 register lines, 6,721 km (21 GYSEV, 112 MÁV). 44 more RINF lines were left with nothing:
  connecting curves and freight branches whose junction-ended sections no OSM passenger route runs
  over (74 sections, 211 km dropped; 56 junction-ended sections, 162 km, kept because routes run
  over them).
- 350 lines in all with the OSM ones, 1,953 stations, 17,324 route-km; 9 of them named trains
  (IC 929 Savaria, IC Göcsej, IC Kék Hullám, EC Hornád, ICE 90, and four EuroCity/EuroNight).
- 1,679 RINF passenger-typed points, of which 1,340 are an OSM station (1,313 distinct), 11 by
  distance alone. Three of those 11 may be wrong and are worth a look: "Macs-Ipari Park" onto
  Látókép (37 m), "Bodajk felső" onto Csókakő (184 m), "Székesfehérvár-Repülőtér" onto Maroshegy
  (172 m). The other 339 are junctions, yards and halts OSM no longer maps (Harka, Pereszteg,
  Felsőgalla, Bösztör, Kilimán...), which become junction ends.
- `build_tiles` left out 209 ways on track no line touches (126 narrow gauge, 82 rail, 1 tram).

## Check

`python check_model.py --region hu`, against the hu.wikipedia infobox lengths (via Wikidata),
30 lines. 27 are within 2% (0.98-1.01): 100, 80, 20, 140, 29, 60, 108, 17, 41, 25, 35, 154, 16, 5,
147, 42, 102, 10, 75, 70, 116, 111, 23, 14, 47, 12, and 1 (0.99, though its extent differs at
both ends; see the note in REGISTER). The other three differ in extent, not in the build:

- **30** (1.05): the infobox's 221 km is Budapest-Déli to about Nagykanizsa. The article's own
  station table has Székesfehérvár at km 66.9, and the build has it at 66.7.
- **8** (1.05): the numbered line runs on 5.4 km past Sopron to the Austrian border.
- **120a** (0.92): the article counts from Budapest-Keleti; MÁV's line starts at Rákos, about
  8 km out on line 100.

Against RINF's own lengths the median is 1.15 (MÁV's leave out station track, above). The few
below 1.00: 131 (0.85, the tram-train street section, partly on tram track), 205 (0.92), and 8
(0.94), where RINF gives Veszkény - Kapuvár 9.9 km for 3.9 of track.

## Still off, and why

- **Closed lines are drawn where OSM still has their track and stations.** Nothing in a RINF
  build says which lines have passenger trains, and `not_running.py` only greys sections with no
  drawn track. Lines closed to passengers in 2007-2009 whose track OSM keeps as `railway=rail`
  (parts of 27, 37, 62, 117, 151, 152, 13, 64) are register lines like any other. Where OSM
  maps the track as disused or removed (most of 13, 64, 106, 107, 112, 126, 129, 76, 95, 22), the
  125 sections are "unplaced" and left out. A MÁV timetable feed would settle which is which.
- **OSM has few passenger route relations south and east of Budapest**: line 150
  (Budapest - Kelebia) has 2 of 23 stops on any OSM route, 153, 103 and 114 similar. The
  register lines are complete; only operating patterns and junction-ended sections depend on
  routes there. OSM's "S150" relation is Komárom - Székesfehérvár, not line 150.
- **Missing sections on open lines**, 2-3 km each: Röszke - Serbian border on 136 (no OSM track
  path to the border), and junction-ended stubs dropped as unridden on 15 (Harka - the border
  towards Deutschkreutz) and 8 (Győr's curves).
- "916 Répce IC" stays a line rather than a named train (its name starts with the train number).
- 20% of RINF's passenger-typed points (339 of 1,679) are not OSM stations. Most are closed
  halts, and the few that looked live (Harka, Pereszteg, Felsőgalla) have no stop of any kind in
  OSM.

## Shared-file changes made for Hungary (by the managing session)

- `rinf.py`: `NOT_MAINLINE` leaves out trolleybus items (Hungarian Wikidata numbers Debrecen and
  Budapest trolleybus lines 3, 5, 9, 10... like MÁV lines); an optional `im_of(section)` per
  country for the operator; an optional `tol_abs`. `--fetch` was also fixed (a local variable
  shadowed `country()`).
- `build_model.py`: a `hu` branch in `looks_like_service` (`HU_TRAIN`: IC, EC, EN, ICE, RJ and
  EuroCity/EuroNight names, with or without a leading "Train ").
