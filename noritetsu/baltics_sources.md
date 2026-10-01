# Baltic register sources (built 2026-10-01): Lithuania, Latvia, Estonia

What the three Baltic builds read, where each piece came from, and what is still wrong with
them. All three are built with `rinf.py`; how the reader works is in its docstring, and the
per-country entries are `rinf_countries/lt.py`, `lv.py` and `ee.py`, whose docstrings say how
each reads its RINF ids. Downloads live in `data/raw/rinf/<cc>/` (gitignored). Nothing needed a
login or a key.

## Run

```powershell
python rinf.py --fetch lt          # and lv, ee; RINF + Wikidata, a few seconds each
python rinf.py --fetch-wikidata lv # Latvian labels only (see lv.py)
curl -L -o data/raw/lithuania-260930.osm.pbf https://download.geofabrik.de/europe/lithuania-260930.osm.pbf
curl -L -o data/raw/latvia-260930.osm.pbf https://download.geofabrik.de/europe/latvia-260930.osm.pbf
curl -L -o data/raw/estonia-260930.osm.pbf https://download.geofabrik.de/europe/estonia-260930.osm.pbf
$env:OSMIUM_POOL_THREADS=1; python extract.py --region lt --pbf data/raw/lithuania-260930.osm.pbf   # 15 s each
python build_model.py --region lt --register rinf:data/raw/rinf/lt   # 4 s; lv 9 s, ee 3 s
python build_tiles.py --region lt                                    # 2-4 s
python check_model.py --region lt
python rinf.py --dry lt            # the reader alone, with its full log
```

The `.pbf` files were deleted after the checks below; `data/proc/<cc>/` keeps what was
extracted.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01. Lithuania 211 sections / 187 points / 29 line ids / 1,832 km, all LTG Infra
  (`1224_IM`). Latvia 40 / 36 / 20 / 1,505 km, all Latvijas dzelzceļš (`0025_IM`). Estonia 106 /
  101 / 31 / 1,011 km: Eesti Raudtee (`0001_IM`), Edelaraudtee Infrastruktuur (`0002_IM`:
  Tallinn - Lelle - Pärnu, Lelle - Viljandi, Liiva - Ülemiste) and four industrial managers
  (`0007`-`0010`, the Ida-Viru oil-shale railways). Every point in all three has a coordinate;
  Lithuania's are on `geo:hasGeometry` written `POINT (+22.69 56.19)`, which rinf.py reads since
  its WKT pattern took the `+` (2026-10-01). The "Lithuania has no coordinates" in
  multi_sources.md was that sign.
- **OpenStreetMap**, Geofabrik `<country>-260930.osm.pbf` (data to 2026-09-30), ODbL: track,
  stations, train routes (Lithuania 14 Lithuanian ones, Latvia 56, Estonia 38, nearly all
  Elron's), trams.
- **Wikidata**: Latvia's line items (LDz numbers 01-36 in P1671, lengths in P2043); Lithuania's
  P1671 values are section-pair codes ("524-525") that join nothing, so `lt.py` fetches none;
  Estonia has one (R14, a timetable route).
- **Passenger service**, to decide which lines with stations at both ends carry no train:
  Vivi's timetable page (https://www.vivi.lv/lv/informacija-pasazieriem/) and an irliepaja.lv /
  nra.lv report of the Rīga - Liepāja trains; lt.wikibooks "Traukinių tvarkaraščiai/2025" and a
  search of ltglink.lt for LTG Link (ltglink.lt itself answers 403 to scripts:
  https://ltglink.lt/nauji-tvarkarasciai); en.wikipedia "Elron (rail transit)" for Elron.
- **Published lengths** for `check_model.REGISTER`: lt.wikipedia "Lietuvos geležinkelių
  transportas" (LTG Infra's line list with km), Wikidata P2043 for Latvia, et.wikipedia "Eesti
  raudteetransport" (the line list with km), raw wikitext, retrieved 2026-10-01.

## How each country's ids read

- **Lithuania**: the id is the line's name in ASCII, "Kyviskes-Vilnius-Kaisiadorys-Kaunas-KazluRuda".
  No public numbers exist, so lines have no ref and are named from the id with the letters put
  back (`NAMES`, via `id_name`). Point names are ASCII too ("KazluRuda"); `rinf.norm` folds them
  onto OSM's "Kazlų Rūda".
- **Latvia**: the id is the codes of the section's two end points ("LV2060100301" is Rīga
  Pasažieru LV106010 to Jelgava LV103010), not LDz's line number, so `FIXED` maps each of the 20
  ids to LDz's 01-36. Names are the line's ends, "Rīga–Jelgava"; the number is the ref.
- **Estonia**: the id is the name ("Tapa-Tartu"); no numbers; named from the id. Every point is
  "Rakenduspunkt Tapa", stripped by `fix`. `name_m` 1,600 m, because Ülemiste's point is 1.46 km
  from its station (and otherwise fell to the halt Vesse by distance); `tol_abs` 1.0, because
  the same point puts 1.5 km of Ülemiste yard into Balti - Ülemiste. Balti - Tallinn-Väike (3 km,
  Eesti Raudtee's) is folded into the Rapla/Viljandi line with a group key that writes no ref.

## Stops RINF leaves out (`osm_stops`)

Latvia's RINF has only junction stations: Rīga - Aizkraukle was one 82 km section. Estonia's
lists about one Elron stop in three. `osm_stops` (rinf.py, 2026-10-01) cuts each traced section
at OSM stations that an OSM train route stops at and that lie on the trace. Latvia gains 88 stops
(Rīga - Krustpils 2 -> 26 sections, the Jūrmala line 3 -> 24, Ludza and Zilupe, Valmiera),
Estonia 61 (all of Tallinn - Keila, the Rapla, Paldiski, Riisipere and Koidula lines). Lithuania
gains none with it, so it is off there: RINF already lists every route-served stop.

Where OSM has no train route, no stops are added: Latvia's Jelgava - Liepāja (Dobele, Saldus,
Skrunda...) and Daugavpils - Indra (Krāslava...) still run end to end with only their RINF
stations, though Vivi serves both. A station on a parallel line can land on a section that runs
past its platforms (Lilleküla onto the Rapla line, which Rapla trains pass without stopping).

## What is in the register and what stays OSM

**Lithuania** 16 register lines, 1,215 km, 133 stations on them (145 in all, 29 lines in all).
Left out with no passenger train although both ends are stations (`skip_line`): Jonava -
Rizgonys, the Vilnius bypass Kyviškės - Vaidotai - Paneriai, Radviliškis - Pakruojis -
Petrašiūnai, the Šiauliai bypass Šilėnai - Jonaitiškiai, Radviliškis - Tauragė - Pagėgiai,
Rimkai - Draugystė (Klaipėda port). Cut back to their ridden part (`fix`, `KEEP_ONLY`): the old
Minsk line to Vilnius Airport ("Vilnius–Oro uostas"), Radviliškis - Rokiškis to Panevėžys,
Klaipėda - Pagėgiai to Šilutė. Dropped by build_model as unridden: Šeštokai - Alytus, Švenčionėliai
- Utena, Kretinga - Darbėnai, Akmenė - Alkiškiai, Mažeikiai - Reņģe, Vaidotai - Valčiūnai,
Rokai - Jiesia, Mažeikiai - Bugeniai, the border stubs to Belarus and Latvia beyond the trains.
Kept because an OSM route runs over them although it is not a domestic service: Kazlų Rūda -
Kybartai and Kyviškės - Kena carry LTG Link's Kaunas - Kybartai and Vilnius - Kena trains (both
real), and the Kybartai - border stub only Russian Kaliningrad transit trains.

**Rail Baltica**: the standard-gauge Kaunas (Palemonas) - Mockava - Polish border line is in
RINF ("Palemonas-KazluRuda-Sestokai-Mockava-PL", 125 km) and is built, as "Palemonas–Kazlų
Rūda–Šeštokai–Mockava (Rail Baltica)", 124.6 km. The broad-gauge line beside it from Kazlų Rūda
to Mockava is its own register line (64 km). OSM maps the corridor as parallel 1435 and 1520
track (some dual-gauge), and the trace does not filter by gauge, so both lines lie on the same
corridor; riding either credits the other there. OSM's 1435 `route=railway` relation "Rail
Baltica" names nothing, since `lt.py` reads no OSM relation names.

**Latvia** 10 register lines, 923 km, 106 stations on them (318 in all, 52 lines in all).
Left out (`skip_line`), no train per Vivi's route list: 01 Ventspils - Tukums II, 02 Tukums II -
Jelgava, 03 Jelgava - Krustpils, 10 Rēzekne I - Daugavpils, 09 Kārsava - Rēzekne I with the
Rēzekne I - II link. Dropped as unridden: 11 Daugavpils - Kurcums, 12 Eglaine, 21 Glūda - Reņģe,
22 Zasulauks - Bolderāja, 26 the Daugavpils bypass. **Not in RINF at all**, so they stay OSM
lines only: 19 Zemitāni - Skulte (Vivi's Skulte line, busy; OSM has it), and 27 Pļaviņas -
Gulbene (Vivi runs Rīga - Gulbene; OSM has no route, so it is not on the map). The Gulbene -
Alūksne narrow gauge stays an OSM line.

**Estonia** 13 register lines, 721 km, 120 stations on them (164 in all, 38 lines in all).
Left out: the industrial managers' lines and their boundary stubs, the freight bypass Ülemiste -
Blokkpost 4 km, and Valga - Valkapiir (the same 1.9 km to the Latvian border as Valga -
Lugažipiir; the two border points are 5 m apart). Cut back (`fix`, `DROP_AT`): Tallinn - Lelle -
Pärnu at Lelle (Elron ended Pärnu trains in December 2018), named "Tallinn–Rapla–Lelle"; Valga -
Koidula to its Piusa - Koidula end (Elron's Tartu - Koidula trains turn at Piusa; nothing runs
through Võru), named "Piusa–Koidula". Dropped as unridden: Lagedi - Muuga, Liiva - Ülemiste, the
Narva and Koidula border stubs. Riisipere - Turba (reopened 2020) is not in RINF and stays on
Elron's R16 OSM line.

Tallinn's and Rīga's (and Daugavpils') trams stay OSM lines.

No `colours/<cc>.csv`: the managers publish no line colours; Elron's routes keep OSM's colours.

## Check

`check_model` against RINF's own section lengths: Lithuania 16 lines median 0.995, none off by
more than 5%; Latvia 10, median 0.997, one (Rīga–Lugaži, below); Estonia 12, median 0.996, one
(Klooga–Kloogaranna, 3.2 of 3.4 km).

| line | built | published | ratio |
|---|---|---|---|
| **lt** Naujoji Vilnia–Turmantas | 138.1 | 139.0 | 0.99 |
| Šiauliai–Joniškis | 59.2 | 60.0 | 0.99 |
| Palemonas–Kazlų Rūda–Šeštokai–Mockava (Rail Baltica) | 124.6 | 120.0 | 1.04 |
| Palemonas–Gaižiūnai–Radviliškis–Šiauliai–Klaipėda | 310.4 | 312.3 | 0.99 |
| Kyviškės–Kena | 18.7 | 19.0 | 0.98 |
| Kazlų Rūda–Kybartai | 50.4 | 57.3 | 0.88 |
| Lentvaris–Varėna–Marcinkonys | 81.1 | 107.0 | 0.76 |
| Senieji Trakai–Trakai | 3.6 | 3.0 | 1.21 |
| **lv** Rīga–Jelgava | 42.6 | 43.0 | 0.99 |
| Jelgava–Liepāja | 179.2 | 180.0 | 1.00 |
| Jelgava–Meitene | 32.8 | 33.0 | 0.99 |
| Rīga–Lugaži | 165.6 | 166.0 | 1.00 |
| Torņakalns–Tukums II | 67.3 | 65.0 | 1.04 |
| Krustpils–Daugavpils | 88.1 | 88.4 | 1.00 |
| Daugavpils–Indra | 68.9 | 76.0 | 0.91 |
| Rēzekne II–Zilupe | 55.2 | 55.0 | 1.00 |
| **ee** Tapa–Tartu | 111.3 | 112.0 | 0.99 |
| Tartu–Valga | 82.6 | 83.0 | 0.99 |
| Tartu–Koidula | 85.4 | 87.0 | 0.98 |
| Tapa–Narva | 131.6 | 133.4 | 0.99 |
| Keila–Paldiski | 20.8 | 21.1 | 0.99 |
| Lelle–Viljandi | 78.4 | 79.2 | 0.99 |
| Keila–Riisipere | 24.5 | 31.0 | 0.79 |
| Klooga–Kloogaranna | 3.2 | 3.0 | 1.08 |

Some published figures are a composite less a piece of RINF's chainage (Tapa–Narva is Tallinn -
Narva 211 less Tallinn - Tapa); `REGISTER`'s notes say which.

- **Lentvaris–Varėna–Marcinkonys** (0.76): the figure runs on to the Belarusian border; RINF's
  line stops at Marcinkonys.
- **Keila–Riisipere** (0.79): the figure is Keila - Turba; Riisipere - Turba is not in RINF.
- **Kazlų Rūda–Kybartai** (0.88): RINF's own Kaunas - border is 87.3 against the article's 94;
  the build matches RINF.
- **Daugavpils–Indra** (0.91): Indra - border (6 km) has no OSM route and is dropped.
- **Senieji Trakai–Trakai**, **Klooga–Kloogaranna**: the figures are whole kilometres.
- **Torņakalns–Tukums II** (1.04): the built line starts at Rīga Pasažieru, 2.3 km before
  Torņakalns.
- **Rail Baltica** (1.04): RINF itself says 125.1.
- **Rīga–Lugaži** against RINF (1.05): RINF's Vangaži - Krievupe (2.8 km for 5.0 of track) and
  Jāņamuiža - Cēsis are short; the trace is on the line's own track and matches the published
  166 km.
- **Jelgava–Liepāja**: RINF puts 58.3 km between Jelgava and Glūda for 15.5 of track, and the
  rest onto Glūda - Liepāja; the line's total is right.

## Still off

- Võru, Antsla and Karula have no OSM station node (only bus stops), and RINF's Lepassaare,
  Nõmmküla and Vaeküla match nothing; none of those lie on a built line now.
- Latvia's Liepāja and Indra lines have no intermediate stops (no OSM route to take them from).
- RINF types LTG's two EMU depots ("El.tr.depas 1/2") as passenger stops, and OSM maps them as
  stations, so they appear as stops on the Turmantas and Kena lines.
- The register lines have no English names (name_en is empty for lt/ee; lv's are "Rīga–Jelgava
  line").
