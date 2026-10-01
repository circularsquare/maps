# Poland register sources (built 2026-09-30)

What the Polish build reads, where each piece came from, and what is still wrong with it.
Poland is built with `rinf.py`; how that reader works is in its docstring, and the per-country
entry is `rinf_countries/pl.py`. Downloads live in `data/raw/rinf/pl/` (gitignored). Nothing
needed a login or a key.

## Run

```powershell
python rinf.py --fetch pl                                             # RINF + Wikidata, ~20 s
curl -L -o data/raw/poland-260929.osm.pbf https://download.geofabrik.de/europe/poland-260929.osm.pbf   # 2.1 GB
$env:OSMIUM_POOL_THREADS=2; python extract.py --region pl --pbf data/raw/poland-260929.osm.pbf        # 4 min; delete the .pbf after
python inspect_region.py --region pl
python build_model.py --region pl --register rinf:data/raw/rinf/pl       # 3 min
python build_tiles.py --region pl                                        # 1 min
python check_model.py --region pl
python rinf.py --dry pl          # the reader alone, with its full log (every rejection)
```

The dated file came from https://download.geofabrik.de/europe/poland.html (data to 2026-09-29).

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-09-30 (`sections.json`: 4,968 sections, 638 line ids, 19,883 km; `points.json`: 4,228
  points, every one with a coordinate). Only PKP Polskie Linie Kolejowe (`0051_IM`) is
  registered.
- **OpenStreetMap**, Geofabrik `poland-260929.osm.pbf`, ODbL: track, stations, 1,237 passenger
  route relations and 1,039 `route=railway`/`route=tracks` relations. PLK's own relations carry
  the line number as a plain `ref` ("91"); the extract also holds Czech (SŽ), German (DB
  InfraGO), Belarusian and Ukrainian border relations and a few industrial lines (JSK numbers
  its own lines 21-24, clashing with PLK's), none of which can rename a line because the RINF
  id is trusted outright.
- **Wikidata** (`wikidata.json`), CC0: 600 items with a P1671 route number and P17 Poland,
  Polish labels only (see Names).
- **pl.wikipedia**, the infobox "długość" of each "Linia kolejowa nr N" article, via the
  MediaWiki API, retrieved 2026-09-30. The articles cite PLK's own line list, **Id-12 (D-29)
  "Wykaz linii"**, editions 2020 to 2026 (the 2026 edition is dated 2026-02-06). These are the
  figures in `check_model.REGISTER["pl"]`. The PDF itself could not be fetched: both PLK URLs
  the articles cite (`.../Instrukcje/Wydruk/Id/Id-12__D-29__Wykaz_Linii.pdf` and the older
  `.../Wydruk/Id-12__D29__Wykaz_linii.pdf`) return 404 as of 2026-09-30, PLK's instructions
  page no longer links it, and the Wayback Machine holds only the 2016, 2019 and 2020 editions.
  PLK's server also presents a certificate chain that Git Bash's curl rejects; Windows'
  `curl.exe` gets through.

## Names

RINF's id is `PL` + the IM code `0051` + PLK's three-digit line number: `PL0051001` is line 1,
`PL0051573` a connecting curve. Every one of the 638 ids has that form, and the number is what
riders, PLK and pl.wikipedia use, so `pl.py` reads it straight off the id (`rule_certain`).
All 646 RINF line pieces got their number that way: 463 confirmed by an OSM relation of that
number lying on the traced line, 58 with no relation near, 125 by the rule alone.

Names are PLK's form, `Linia kolejowa nr 91`; English names `Line 91`. Wikidata's English
labels are not used: of the 74 lines that had one, several named a historic railway the modern
line only partly follows ("Prussian Eastern Railway" on 203 Tczew - Kostrzyn, "Berlin-Wrocław
railway" on 273 Wrocław - Szczecin), called a running line "Former" (234), or were boilerplate
("railway line nr 52 (Poland)"), and rinf.py would have used them as the whole English name.
The loss is a handful of good ones, "Central Rail Line" for 4 and "Polish Coal Trunk-Line" for
131.

No `colours/pl.csv`: PLK's numbered lines have no colours. Service colours (SKM, KM, KŚ, ŁKA)
belong to the operators' lines, which stay OSM lines with the colours OSM gives them.

## What RINF carries in Poland, and what stays OSM

RINF is PLK alone. These stay OSM lines, as they were:

- **SKM Trójmiasto's own track**, Gdańsk Śródmieście - Rumia (PKP SKM's line 250): RINF has
  only PLK's 0.7 km handover stub at Rumia.
- **PKM** (Pomorska Kolej Metropolitalna, line 248, Gdańsk Wrzeszcz - Osowa): RINF has only
  PLK's stubs at each end. PLK's line 201 on from Osowa to Gdynia is a register line.
- **WKD** (Warszawska Kolej Dojazdowa, lines 47, 48 and 512).
- **PKP LHS** (line 65, broad-gauge freight).
- **The lines DSDiK owns in Lower Silesia** (OSM names it as operator of 308, 310, 317, 318,
  319, 336, 340, 372; none is in RINF), wherever a passenger route runs over them.
- **UBB's stretch to Świnoujście Centrum**: not in RINF, so whatever OSM routes cover it.
- **Warsaw Metro** (M1, M2), every tram network, and the narrow-gauge and heritage railways.

## Counts (2026-09-30)

- 352 register lines, 16,171 km, after build_model dropped the junction-ended sections no OSM
  passenger route runs over (288 sections, 1,559 km). 229 PLK lines were left with nothing,
  the freight lines. 124 of the 352 are numbered 500 and up, PLK's connecting curves and
  station bypasses (388 km).
- 877 lines in all: those 352, 465 OSM lines (Polregio, KM, KŚ, KD, ŁKA, SKM, WKD, Arriva,
  Koleje Wielkopolskie, 213 tram lines, the Warsaw Metro), and 60 PKP Intercity and
  international named trains (flagged as services by `build_model.looks_like_service`'s `pl`
  branch). 5,768 stations, 3,005 of them on a register line. 43,368 route-km without the named
  trains.
- 3,516 RINF passenger-typed points, of which 2,910 are an OSM station (2,880 distinct), none
  by distance alone. The other 606 are stations on lines closed to passengers (Pyrzyce,
  Lipiany, Mrągowo, the Kashubian branches) and freight "stations", and become junction ends.

## Check

`python check_model.py --region pl`: against RINF's own section lengths, 296 lines of 2 km or
more, median 1.004. Against PLK's Id-12 figures (via pl.wikipedia), 30 lines, 29 within 2%
(most 0.99 to 1.00). The other one:

- **38** (0.83): RINF's line stops short of the stretch past Bartoszyce to the border at
  Głomno, which Id-12 counts; RINF's own 202.0 km is what the build has.

RINF agrees with Id-12 to within 1% on almost every main line; line 6 is the exception
(RINF 218.2 against 224.2), and is in REGISTER with that note.

Left out of REGISTER because part of the line is closed to passengers or being rebuilt, which
the build rightly does not draw: 29, 97, 103, 104, 108, 181, 190, 201 (Żukowo - Kościerzyna),
209, 229, 275, 281, 285, 309, 356, 368, 369.

## Still off, and why

- **Lines closed or being rebuilt have no OSM track to trace.** 312 RINF sections are more
  than 1.5 km from any `railway=rail` way, because OSM tags the track `disused`, `abandoned`
  or `construction` (Kłodzko - Kudowa on 309 is `construction` with `opening_date=2026-10-31`;
  Lębork - Łeba on 229, Szamotuły - Międzychód on 368, Kościerzyna - Somonino on 201 are
  disused or under works). They are logged as `untraceable` and `rejected`, and they come back
  by themselves once OSM retags the track and the extract is re-run.
- **Two lines whose own track is gone were traced over other lines** until `rinf.py` gained a
  guard (2026-09-30, `OWN_DIRECT`): 223 Czerwonka - Ełk, disused through Mrągowo and
  Mikołajki, came out as 135 km over Korsze and Giżycko (lines 353 and 38), 1% of it on its
  own relation's ways, which was within 15% of RINF's 121 km. The same happened to 145
  Chorzów Stary - Radzionków. Both are now rejected.
- **PLK's short connecting curves read 1.5 to 4.7 times their RINF length** (89 lines off by
  more than 5%, nearly all numbered 500 and up and under 5 km). RINF puts a station's point at
  the station centre, while the curve's chainage starts at the station throat: Łowicz Główny
  to "Łowicz Główny PZS R12" is 3.0 km as the crow flies but 0.75 km in RINF. The trace runs
  from the station along the main line to the curve, so these register lines lie partly on
  their main line's track. `rinf.py`'s "traces agree" rule keeps them. Longer lines off for
  the same reason: 100 (Kraków Główny - Mydlniki, 1.53), 67 (Lublin - Świdnik, 1.25), 78
  (1.18), 160 (1.13), 13 (1.07), 15 (1.05).
- **The Warsaw airport spur (440) is left out.** Its west end is RINF's freight station
  "Warszawa Okęcie" at the junction; the reader puts it at the Okęcie passenger halt, south of
  the junction, and the trace to Lotnisko Chopina comes out 3.4 km against RINF's 1.9. Airport
  trains come from Służewiec and never make that move, and the SKM and KM airport routes stay
  as OSM lines, so only the register's 1.9 km is missing.
- **159 sections are kept although their length disagrees with RINF's** (`length off` in the
  log), because the trace runs on the line's own OSM relation or two independent traces agree.
  The largest main-line one: line 24, Piotrków Trybunalski - Rogowiec, 32.2 km in RINF and
  37.5 km traced.
