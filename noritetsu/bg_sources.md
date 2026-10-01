# Bulgaria register sources (built 2026-10-01)

What the Bulgarian build reads, where each piece came from, and what is still wrong with it.
Bulgaria is built with `rinf.py`; how that reader works is in its docstring, and the
per-country entry is `rinf_countries/bg.py`. Downloads live in `data/raw/rinf/bg/` (gitignored).
Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch bg                                              # RINF + Wikidata, ~10 s
curl -L -o data/raw/bulgaria-260930.osm.pbf https://download.geofabrik.de/europe/bulgaria-260930.osm.pbf   # 174 MB
$env:OSMIUM_POOL_THREADS=1; python extract.py --region bg --pbf data/raw/bulgaria-260930.osm.pbf --station-areas   # 30 s; delete the .pbf after
python inspect_region.py --region bg
python build_model.py --region bg --register rinf:data/raw/rinf/bg     # 12 s
python build_tiles.py --region bg                                      # 8 s
python check_model.py --region bg
python rinf.py --dry bg          # the reader alone, with its full log (every rejection)
```

`--station-areas` matters: Bulgarian OSM maps 29 stations only as a building or area with
`railway=station` and no node (Чирпан, Калофер, Клисура, Първомай, Силистра, Плевен Запад,
Берковица, Бяла, Дряново...). Without it 15 RINF stations become junctions and the ends of lines
91 (Дулово - Силистра) and 71 (Монтана - Берковица) are dropped as unridden.

The dated file came from https://download.geofabrik.de/europe/bulgaria.html (data to
2026-09-30).

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01 (`sections.json`: 350 sections, 54 line ids, 3,766 km; `points.json`: 320 points,
  every one with a coordinate). Only NRIC (НКЖИ, `0052_IM`) is registered.
- **OpenStreetMap**, Geofabrik `bulgaria-260930.osm.pbf`, ODbL: track, stations, and only 16
  `route=train` relations (plus 8 metro and 34 tram routes); 56 `route=railway` relations,
  NRIC's carrying the line number as `ref` ("2", "82.1") and a name "Железопътна линия 2:
  София - ... - Варна".
- **Wikidata** (`wikidata.json`), CC0: 41 route numbers (P1671) with Bulgarian and English
  labels. Several items carry a wrong P1671 of 1 (the Cherven Bryag - Oryahovo, Yasen -
  Cherkvitsa and Septemvri - Dobrinishte items); `rinf.wikidata_item`'s length check keeps them
  off line 1. Line 24's item has line 23's length (43 km), so line 24 gets no English route.
- **bg.wikipedia**, the line articles via the MediaWiki API, retrieved 2026-10-01
  (`bgwiki_lines.json`, `bgwiki_mainlines.json`): the infobox "дължина" or the length in the
  text. These are the figures in `check_model.REGISTER["bg"]`. Most are uncited.
- **NRIC Network Statement 2026-2027** (`nric_network_statement_2026.pdf`,
  https://www.rail-infra.bg/upload/7621/Network+Statemen+2026-2027_v.00_14122025.pdf): network
  totals and border crossings only, no per-line lengths; its line map is a separate annex.
  NRIC's server presents a certificate chain Git Bash's curl rejects; Windows' `curl.exe` works.

## Line numbers

RINF's ids are NRIC's numbers with lettered parts; `bg.py`'s FIXED reads them back:

| RINF id | line | what it is |
|---|---|---|
| 1A, 1B | 1 | Sofia - Svilengrad, Sofia - Kalotina Zapad |
| 1A1 | 1 (from OSM's line 1 relation) | Svilengrad - Greek border |
| 3A, 3B | 3 | Iliyantsi - Zimnitsa, Karnobat - Varna feribotna (line 8 between) |
| 4A, 4B, 4C, 4A2-4A4 | 4 | Ruse - Stara Zagora, Dimitrovgrad - Mihaylovo, Dimitrovgrad - Podkova, Ruse junction pieces |
| 6A, 6B | 6 | Voluyak - Pernik Razp., Radomir - Gyueshevo (line 5 between) |
| 8A | 8 | second Burgas approach |
| 51A / 51B | 51 / 52 | Dupnitsa - Bobov dol / General Todorov - Petrich |
| 821, 291, 331 | 82.1, 29.1, 33.1 | dotted numbers written without the dot |
| 701, 7A2 | none | Vidin - Danube Bridge 2 links (see bg.py; built, then dropped as unridden) |

Every other id is the number itself. Names are NRIC's form, "Железопътна линия 2"; English "Line
2", plus Wikidata's English label where it passes the length check ("Line 2 (Sofia–Varna
railway)").

No `colours/bg.csv`: NRIC's numbered lines have no colours.

## Stations

RINF lists stations (гари) and not halts (спирки): 265 passenger points for about 650 places
trains call at. Three things make the station list usable:

- `stop_names` in bg.py: 29 points RINF types as junction (80), yard (100) or switch (120) that
  are passenger stations: Мездра, Шумен, Карнобат, Каспичан, Левски, Радомир, Волуяк, София
  Север, Русе Разпределителна... Each checked against an OSM station or halt within 100 m.
- `osm_stops: "all"` (rinf.py, added for Bulgaria and the Baltics): every OSM rail station
  lying on a traced stop-to-stop section becomes a stop of it. "All" rather than "stopped at by
  an OSM route", because Bulgaria has only 16 route relations. 728 candidates; 682 stations are
  on register lines in the build.
- rinf.py's `CYRILLIC_FOLD` (added for Bulgaria): RINF names are an old Latin transliteration
  in capitals (KAZANLAK, TRJAVNA, KJUSTENDIL), OSM's are Cyrillic. Folding Cyrillic lets 236 of
  288 RINF stops match by name; 52 match by distance alone (all within 100 m, listed in the
  build log). Before the fold every Cyrillic stop position counted as a separate station, which
  put 142 same-name pairs on register lines (Атолово three times).

6 RINF passenger points match no OSM station: Горна Оряховица разпределителна, Пост 1, РП
Биримирци, РП Капитановци, Русе Север, Русе Запад. They are junction ends.

Freight terminals (type 40: Видин товарна, Станянци, Бобов дол, Пловдив разпределителна) are
not stops. `osm_stops` still adds Видин товарна, Товарна гара Бургас, Бургас разпределителна and
Вагоно-Ремонтно Депо as stops where OSM tags them `railway=station`/`halt` on a line's track.

## Counts (2026-10-01)

- 30 register lines, 3,576 km, after build_model dropped 14 junction-ended sections (102 km)
  no OSM passenger route runs over: the freight branches 11, 12, 15 and 51, curves 29 and 38,
  41 to Lyaskovets, the Vidin bridge links, border stubs at Kalotina and Kulata, and Trakia -
  Plovdiv Razpredelitelna on line 8 (below). 9 RINF lines were left with nothing.
- 58 lines in all: those 30, 4 Sofia Metro lines, 17 Sofia tram lines, 7 OSM train routes, of
  which INT 12503 Istanbul - Bucharest is a named train (`EU_TRAIN` with `INT`, bg added
  2026-10-01). 880 stations, 682 on a register line. bg.pmtiles 0.9 MB.

## Check

`python check_model.py --region bg`: against RINF's own section lengths, 30 lines, median 0.998
(61 at 1.06 and 32 at 1.08, both under 18 km). Against bg.wikipedia / Wikidata, 18 lines, 15
within 3%. The others:

- **1** (1.05): the built line also has Svilengrad - Turkish border (18.9 km) and Svilengrad -
  Greek border (3.9), which OSM's Istanbul and Pythio trains run over, and Sofia - Poduyane
  (3.2); it lacks Ihtiman - Verinsko (8.4 km), see below.
- **4** (1.06): the figure is Ruse - Stara Zagora plus Mihaylovo - Podkova; the build adds the
  Ruse junction pieces and the Giurgiu border stub (about 17 km), and Ivanovo - Dve Mogili traces
  15.6 km where RINF says 9.7 (the halts between are 14.2 km apart as the crow flies, so RINF is
  wrong there).
- **28** (0.95): Kardam - Romanian border has no OSM path, Razdelna - RP Razdelna is unridden.
- **52** (0.94): RINF's own is 8.9 km against the article's 9.575.

Left out of REGISTER: freight lines 11, 12, 15, 51; line 86 (Burgas - Sarafovo, razed in OSM);
3, 7 and 24, which have no single clean published figure (7's article says "just under 192 km";
24's infobox repeats line 23's 43.005).

## Still off, and why

- **Line 1 has an 8.4 km gap, Ihtiman - Verinsko.** The Sofia - Plovdiv modernisation is
  rebuilding it; OSM tags the new alignment `railway=construction` and the old one `razed`, and no
  OSM route relation runs there, so extract.py's construction-track rule (which keeps
  construction track a passenger route uses) does not apply. It comes back when OSM retags it.
- **Line 8 lacks Trakia - Plovdiv Razpredelitelna (5.5 km)**, the track every Plovdiv - Stara
  Zagora train uses. RINF ends line 8 at the Plovdiv Razpredelitelna freight yard, which has no
  OSM station within 230 m, so the section ends at a junction, and no OSM train route runs over it.
  Needs a shared change (reported to main).
- **Septemvri - Dobrinishte (NRIC line 16, 760 mm, 125 km) is missing entirely.** It is not in
  RINF, and OSM has only its `route=railway` relation (ref 16), no `route=train`, so nothing
  makes it a line and build_tiles leaves out its 176 narrow_gauge ways as untouched track. The
  same holds for every other Bulgarian line with no OSM route relation and no RINF entry
  (Червен бряг - Оряхово, Червен бряг - Златна Панега, Пазарджик - Варвара, all closed or
  narrow gauge anyway).
- **Strip-diagram order for split lines**: `display` comes from n02.walk_order, which only walks
  the first connected piece, so lines 3 (two pieces), 4, 6 and 1 (the Ihtiman gap) list fewer
  display stations than they have (reported to main).
- **Same-name pairs left**: 5 stations mapped as both a `railway=station` node and a
  `railway=halt` node of one name (Бойка, Кадиево, Стамо Костов, Синьо бърдо, Брусен) appear
  twice on their line, a few metres apart. Челопеч appears twice 843 m apart (station and halt).
- **Karlovo - Botev** (line 3) traces 8.2 km for RINF's 5.4, and **Karlukovo - Cherven Bryag**
  (line 2) 11.7 for 15.1; both kept because the trace lies on the line's own OSM relation, and the
  line totals agree with RINF.
