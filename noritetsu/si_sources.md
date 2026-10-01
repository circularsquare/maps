# Slovenia register sources (built 2026-10-01)

What the Slovenian build reads, where each piece came from, and what is still wrong with it.
Slovenia is built with `rinf.py`; how the reader works is in its docstring, and its
per-country entry is `rinf_countries/si.py`. Downloads live in `data/raw/rinf/si/`
(gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch si                                              # RINF + Wikidata, ~10 s
curl -L -o data/raw/si-260930.osm.pbf https://download.geofabrik.de/europe/slovenia-260930.osm.pbf
$env:OSMIUM_POOL_THREADS=1; python extract.py --region si --pbf data/raw/si-260930.osm.pbf   # 25 s; delete the .pbf after
python inspect_region.py --region si
python build_model.py --region si --register rinf:data/raw/rinf/si     # 10 s
python build_tiles.py --region si                                      # 5 s
python check_model.py --region si
python rinf.py --dry si          # the reader alone, with its full log
```

The dated file came from https://download.geofabrik.de/europe/slovenia.html (313 MB). The
extract must be made with an `extract.py` that keeps construction track under passenger
routes (2026-10-01, see "Line 50 at Preserje" below); an older one loses 6 km of line 50.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01 (`sections.json`: 319 sections, one version each, all valid 2025-01-27 to
  2099-12-31; `points.json`: 316 points). 28 line ids, about 1,206 km, all under
  SŽ-Infrastruktura (`0079_IM`). Every point has a coordinate.
- **OpenStreetMap**, Geofabrik `slovenia-260930.osm.pbf` (data to 2026-09-30), ODbL: track,
  stations, 43 passenger route relations, 73 `route=railway` relations carrying SŽ's line
  numbers as `ref`.
- **Wikidata** (`wikidata.json`), CC0: 42 rows, items with a route number (P1671) and P17
  Slovenia, Slovenian labels only (see `si.py` for why). Used only for the "built as no line"
  log line; names come from `si.py`'s `ROUTES`.
- **sl.wikipedia**, raw wikitext via the API, retrieved 2026-10-01: "Seznam železniških prog
  v Sloveniji" and every article in "Kategorija:Železniške proge v Sloveniji". The article
  titles give the route part of each line name; the infobox "dolžina" (length) and "oznaka"
  (chainage, e.g. 565,9-682,5 for Ljubljana-Sežana) give the published lengths in
  `check_model.REGISTER["si"]`. One infobox (Prvačina-Ajdovščina) cites SŽ-Infrastruktura's
  network statement, Program omrežja 2025; the rest cite nothing, but their chainage and
  RINF's section lengths agree to within 1% on every line except 70 and 82 (below), so
  they are SŽ's figures in effect. The network statement itself was not read.

## How `si.py` reads the ids

RINF's line id is SŽ's own line number, "10", "20", "50", "62". Wikidata's P1671 uses the
same numbers, and so do OSM's `route=railway` relations (25 of 26 ids built confirmed by a
relation of that number, the 26th has none near it). The id is the number
(`rule_certain`).

Riders and sl.wikipedia name a line by its ends ("Železniška proga Ljubljana–Sežana–d. m.",
d. m. = državna meja, the state border). The name is number then route, **"Proga 50
Ljubljana–Sežana"**, English **"Line 50 (Ljubljana–Sežana)"**. The route is a fixed table,
`ROUTES`, from the sl.wikipedia article title for each number, with "Železniška proga" and a
trailing "–d. m." taken off (kept where it would leave one place: "Proga 43 Lendava–d. m.").
Numbers with no article of their own take Wikidata's Slovenian label (60 Divača–Prešnica, 61
Prešnica–Podgorje, 71). Wikidata's labels were not used directly: several numbers carry a
second, historic item (50 is also the whole Südbahn to Trieste, 70 the Bohinj Railway, 64
Pivka-Rijeka).

Other settings, each measured:

- `skip_line`: 11-13, the freight tracks Ljubljana Zalog - Ljubljana (P3-P5). Once Ljubljana
  Zalog is a stop, 12 and 13 ran stop to stop (never questioned by build_model) and traced 8
  km for RINF's 3.8 and 3.5, over line 10's track.
- `name_m` 1500: Frankovci, Radeče and Vidina have their RINF coordinate 1.1-1.3 km from the
  OSM halt of the same name, so at 1,000 m they became junctions and fell out of lines 44, 10
  and 32 as stops. With them matched, the neighbouring sections show RINF's misplacement
  directly (Zidani Most - Radeče 1.93 km in RINF, 3.06 of track; Radeče - Loka 3.10, 1.83),
  and are kept because they run on the line's own relation.
- `stop_names`: Kamnik Graben (type 50, depot; the terminus of line 21), Ajdovščina (50; the
  terminus of line 72), Ljubljana Vižmarje (60; mid line 20) and Ljubljana Zalog (100, shunting
  yard; on line 10) are passenger stations RINF types otherwise. Without this, line 21 ended
  at Kamnik mesto and the other three were not stops.

## Line 50 at Preserje

OSM maps both tracks of line 50 at Preserje as `railway=construction` (the station is being
rebuilt), yet the IC Koper - Maribor, Koper - Ljubljana and Opčine - Ljubljana route
relations run over one of them. With only `railway=rail` track, rinf.py found no path
Preserje - Borovnica and line 50 lost 6.0 km (0.94 of published). `extract.py` now keeps a
construction way when a passenger route lists it (9 of 117 such ways in Slovenia, 4 km), and
line 50 builds at 1.00.

## What is in the register and what stays OSM

Kept: **22 register lines, 1,157 km**, all SŽ-Infrastruktura. 290 stations in the region,
283 section ends on register lines.

Left out, rightly:
- **42 Ljutomer - Gornja Radgona** (22.6 km): freight only (sl.wikipedia marks it
  "tovorna"); no OSM passenger route, dropped as unridden.
- **71 Šempeter pri Gorici - Vrtojba** and **73 Kreplje - Repentabor**: freight links to
  Italy. 73's points are unplaced (no coordinate in reach of track).
- **35 Maribor Tezno - Maribor Studenci** (1 km curve): no passenger route.
- Border stubs with no OSM route over them: Šentilj - border (2.3 km, line 30; trains to Spielfeld-Straß
  run there, but OSM has no route relation for them), Rosalnice - Metlika border
  (0.6 km, line 80), Koper - Koper tovorna (the port freight yard, line 62), Imeno - border
  (line 33, traced 0.35 km for RINF's 1.52 and rejected).

Not in RINF and not mapped as lines: Kranj - Naklo (22), Dravograd - Otiški Vrh (36),
Novo mesto - Straža (83), all freight.

Stay OSM lines (12, of which 2 named trains): R Jesenice - Sežana, the Maribor - Bleiburg
regional (ÖBB), the SŽ cross-border trains to Croatia (Vlak 78 Celje/Rogatec - Đurmanec,
Vlak 18 Opčine - Rijeka, Vlak 1272 Istra Divača - Pula), HŽPP's stubs into Slovenia (Vlak
73 to Lendava, Vlak 80/21 Dobova), a 1 km "Novo mesto - Metlika" (an OSM route with only two
stops mapped), the Ljubljana castle funicular, and the named trains EC 212/213 (merged as
"EC") and EuroNight Lisinski. `si` is in `build_model.EU_TRAIN_REGIONS` (2026-10-01), which
flags exactly those two.

**OSM's passenger routes are thin here**: 43 route relations, and 9 of them have no stop
roles (IC Koper - Maribor, Koper - Ljubljana, Opčine - Ljubljana, Nova Gorica - Ajdovščina,
Ljubljana - Kočevje, Ljubljana - Novo mesto, GYSEV's Hodoš - Zalaegerszeg...), so they never
become lines; their ways still count as ridden for `drop_unridden_sections`. Kamnik,
Celje - Velenje, Ljubljana - Maribor, Pragersko - Ormož and Sevnica - Trebnje have no route
relation at all. Only 59% of main-line track km is under any route (`inspect_region`). That
is why the border stubs above drop, and why riding a register line is the only way to
record most Slovenian trips.

No `colours/si.csv`: SŽ publishes no line colours, and no OSM route relation carries one.

## Counts (2026-10-01)

- 22 register lines, 1,157 km (1,184 km of RINF length on the 25 lines rinf.py built).
- 34 lines in all: 22 register, 9 OSM train lines, 2 named trains (EC, EuroNight
  Lisinski), 1 funicular. 290 stations, 1,870 route-km. Tiles 0.4 MB.
- 276 RINF passenger-typed points, 274 an OSM station (273 distinct), none by distance
  alone. No OSM station: Cvetkovci, Dankovci (line 41; not in OSM), Planina (line 50; OSM
  has only an untyped stop position), Verd (OSM: yard), Rižana and Koper tovorna (freight).

## Check

`python check_model.py --region si`: against RINF's own section lengths, 22 lines of 2 km
or more, median 0.997, one off by more than 5% (82, RINF short). Against sl.wikipedia, 19
lines:

| line | built | published | ratio |
|---|---|---|---|
| Proga 70 Jesenice–Sežana | 129.4 | 129.8 | 1.00 |
| Proga 80 Ljubljana–Metlika | 123.4 | 124.4 | 0.99 |
| Proga 50 Ljubljana–Sežana | 116.2 | 116.6 | 1.00 |
| Proga 10 Ljubljana–Dobova | 114.4 | 114.8 | 1.00 |
| Proga 30 Zidani Most–Šentilj | 105.5 | 108.3 | 0.97 |
| Proga 34 Maribor–Prevalje | 82.9 | 82.1 | 1.01 |
| Proga 20 Ljubljana–Jesenice | 70.6 | 70.4 | 1.00 |
| Proga 41 Ormož–Hodoš | 68.8 | 69.2 | 0.99 |
| Proga 82 Grosuplje–Kočevje | 48.3 | 49.1 | 0.98 |
| Proga 31 Celje–Velenje | 37.4 | 38.0 | 0.99 |
| Proga 32 Grobelno–Rogatec | 36.7 | 36.5 | 1.00 |
| Proga 81 Sevnica–Trebnje | 31.1 | 31.4 | 0.99 |
| Proga 64 Pivka–Ilirska Bistrica | 24.4 | 24.4 | 1.00 |
| Proga 21 Ljubljana Šiška–Kamnik Graben | 23.4 | 23.0 | 1.02 |
| Proga 72 Prvačina–Ajdovščina | 15.0 | 14.8 | 1.01 |
| Proga 61 Prešnica–Podgorje | 14.7 | 14.7 | 1.00 |
| Proga 62 Prešnica–Koper | 28.9 | 31.5 | 0.92 |
| Proga 33 Stranje–Imeno | 12.8 | 14.2 | 0.90 |
| Proga 43 Lendava–d. m. | 4.6 | 5.2 | 0.88 |

- **62** (0.92): the published 31.5 runs into Koper's freight yard (Koper tovorna, 2.6 km),
  which the build leaves out (trace rejected, no passenger route).
- **33** (0.90): Imeno - border (RINF 1.52 km) is rejected; the trace found 0.35 km. No
  passenger train crosses there.
- **43** (0.88): RINF's own length is 4.6; the article's 5.2 is chainage 22.6-17.4, which
  looks to start further back.
- **30** (0.97): Šentilj - border (2.3 km) has no OSM route and is dropped.
- Lines 40 and 44 are one article (Pragersko-Središče, 51.9 km; built 40.2 + 11.5 = 51.7)
  and are not listed.

## Still off, and why

- **RINF's lengths are misallocated or short** in places, kept because the trace runs on the
  line's own OSM relation: Plave - Solkan on 70 (RINF 6.7 km, track 10.7; RINF's line 70 is
  124.8 against 129.8 published, the build gets 129.4), Velike Lašče - Ortnek on 82 (3.1
  against 7.0; RINF 44.2, published 49.1, built 48.3), and the Radeče, Frankovci and Vidina
  pairs where RINF misplaces the halt.
- **Border stubs with trains but no OSM route** are dropped (Šentilj - Spielfeld above).
- Planina and Verd on line 50, Cvetkovci and Dankovci on 41: not stops, since OSM has no
  station for them. If any still has trains, OSM needs a station node there.
- Lines have no colour.
