# Multi-country line sources (surveyed 2026-09-30)

The question: is there a source, or a small set, that gives many countries' passenger lines at
once, so a new country is not a days-long hand-written reader each time? What a country needs,
in order: (1) the line inventory by real name with operator, (2) the stations on each line in
order, ideally with km per section, (3) geometry, which OSM already covers.

All numbers below were measured on 2026-09-30 by the scripts in the session scratchpad (not
kept in the project). Nothing used needed a login.

## The short answer

There is no global line register, but three sources together cover far more than one country
each:

- **ERA RINF** (the EU's Register of Infrastructure) answers (1) and (2) for 27 European
  countries in one open SPARQL endpoint: every mainline as its national line number, split into
  sections of line between operational points, with the length of each section, the type of
  each point (station, passenger stop, junction...), its coordinates and its km on the line.
  Section lengths add up to the published network length in every country checked. It has no
  line names and almost no geometry. It is the Korail 거리표 for most of Europe.
- **OSM infrastructure relations** (`route=railway`, 16,450 worldwide, and `route=tracks`,
  4,032) are line registers of the N02 kind, in the extract we already read: track grouped and
  named by its legal line, often with the national line number in `ref`. Dense in France,
  Germany, China, Poland, Czechia, Hungary and the Balkans; `extract.py` does not keep them yet.
- **Wikidata** has station-to-station adjacency qualified by line (P197 + P81) for a useful
  share of stations everywhere, near-complete for India, strong for China, and it carries the
  line number (P1671) that joins RINF's numbers to names. It is glue and a fallback, not a
  spine: where we know the truth (Japan, Korea, Taiwan, Switzerland) its complete chains cover
  only 40-70% of the register lines.

GTFS aggregators, EuroGlobalMap/EuroRegionalMap and Wikipedia route diagrams were looked at
and are not worth a generic reader; details below.

## ERA RINF

What it gives: (1) national line numbers, no names; (2) yes, sections of line with length,
operational points typed and located with chainage; (3) almost none (2,466 linestrings in the
whole graph; points yes).

Access: SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus` (GraphDB, open,
no key, answers country-wide queries in 1-6 s). Also `https://rinf.data.era.europa.eu/api/v1/sparql/rinf`.
Full dump: Zenodo record 22018826, `ERA-KG.zip`, 676 MB, v10 of 2026-08-19, CC BY 4.0. ERA
states EUPL 1.2 for the service. Vocabulary: `http://data.europa.eu/949/` (`era:`).
bergmannjg/RInfData on GitHub is F# code that queries the same graph (routing, OSM comparison);
it ships no data files.

The model, as found (RINF was restructured in 2025; there is no `NationalRailwayLine` class
any more):

- `era:SectionOfLine`: `era:opStart`, `era:opEnd`, `era:lengthOfSectionOfLine` (km),
  `era:nationalLine` -> `era:LinearPositioningSystem` whose `era:lineId` is the national line
  number, `era:solNature` (regular or link), `era:inCountry`, `era:infrastructureManager`.
- `era:OperationalPoint`: `era:opName`, `era:uopid`, `era:opType` (station 10, small
  station 20, passenger terminal 30, passenger stop 70, junction 80, switch 120, freight 40...),
  and `era:netReference` -> a point with `wgs:lat`/`wgs:long` and `era:hasLrsCoordinate`
  giving line and km ("Railway location of OP Dijon-Ville, km 314.208 on line 830000-1").

Measured per country (sections deduplicated by line and end points; "passenger OPs" are
types 10/20/30/70 that are the end of some section; "one chain" is lines whose sections form a
single connected piece):

| country | lines | one chain | km | passenger OPs | line id looks like |
|---|---|---|---|---|---|
| Germany | 1,492 | 1,429 | 33,406 | 6,574 | 4000 (VzG number) |
| France | 893 | 829 | 27,390 | 2,844 | 830000-1 (RFN code) |
| Poland | 638 | 630 | 19,883 | 3,516 | PL0051003 (line 3) |
| Italy | 327 | 312 | 19,357 | 2,706 | N7, F45-F46 (RFI internal) |
| Spain | 466 | 432 | 15,537 | 2,016 | ESL100100010 (Adif) |
| Sweden | 68 | 54 | 10,800 | 489 | 01 (bandel-like) |
| Romania | 271 | 270 | 10,542 | 2,021 | 300, 100A |
| Czechia | 653 | 650 | 9,685 | 2,751 | 220-00_0401 |
| Slovakia | 133 | 133 | 9,660 | 850 | SK_2601 |
| Hungary | 273 | 271 | 6,592 | 1,679 | 30, 100/1 |
| Finland | 112 | 109 | 5,798 | 210 | 006 |
| Switzerland | 245 | 200 | 5,407 | 986 | 600 (as our Schienennetz) |
| Austria | 160 | 101 | 4,664 | 1,289 | 10501 |
| Belgium | 444 | 444 | 3,971 | 752 | 0360 (line 36) |
| Bulgaria | 54 | 54 | 3,766 | 265 | 2, 3A |
| Norway | none | - | 3,221 | 365 | no line ids |
| Netherlands | 134 | 126 | 3,148 | 408 | Asd-Zp (station codes) |
| Greece | 25 | 23 | 3,021 | 172 | 25.00.00 |
| Portugal | 131 | 130 | 2,492 | 516 | 081 |
| Croatia | none | - | 2,435 | 519 | no line ids |
| Denmark | 340 | 340 | 1,958 | 260 | one id per section, not lines |
| Ireland | 274 | 274 | 1,663 | 267 | SOL1039, one per section |
| Slovenia, Luxembourg, Estonia, Lithuania, Latvia, Liechtenstein | small | | | | |

The km agree with the networks' published lengths: Germany's DB InfraGO about 33,400, France's
RFN about 27,500, Switzerland 5,407 against our Schienennetz build's 5,567. Paris-Marseille
(830000-1) sums to 864 km against a published 862.

Coordinates and km: in every country but those below, every passenger OP has both on its
`netReference` (France 2,844 of 2,844, Germany 13,142 of 13,142 before the version filter).
Austria, Norway, Portugal, Lithuania and Luxembourg OPs have no coordinates on the
`netReference` path (Austria and Norway do have km); a few thousand OPs carry `geo:hasGeometry`
directly, not checked which.

Faults found, all of which a reader has to handle:

- **Germany carries two versions of every section**, labelled "(from 2026-01-01 until
  2026-12-31)" and "(from 2027-01-01)", with different point URIs, so a naive sum is 66,871 km
  and only 69 lines form one chain. Filter on validity (or the label) and it is 33,406 km and
  1,429 of 1,492 lines are one chain.
- **Only the registered infrastructure managers.** Germany is DB InfraGO alone (no NE-Bahnen),
  Poland PKP PLK alone, Spain Adif alone (no FGC, Euskotren, FGV, SFM). Switzerland (17 IMs),
  Italy (about 10: RFI plus regional concessions), Austria, Czechia and Sweden include private
  ones. No metros, trams or light rail anywhere: those stay OSM route relations, as they are now.
- **Passenger typing is operational, not commercial.** Germany's 6,574 station-or-stop points
  include operational Bahnhöfe with no platform (DB has about 5,400 passenger stations).
  France's are tight (2,844, SNCF has about 3,000). Matching to OSM stations, as every reader
  already does, sorts this out; an OP with no OSM station near it is not a stop.
- **Line ids are numbers or codes, never names**, and in Denmark and Ireland each section has
  its own id, so there are no lines to group. Norway and Croatia have no line id at all.
  Italy's ids are RFI's internal codes, not the "Linea Milano-Venezia" names.
- **No section geometry.** Geometry has to come from OSM: shortest path over the rail graph
  between consecutive OP coordinates, checked against the section's own length (which is the
  check that caught every n02.py and kr_register bug).

Wikidata links to RINF only in two countries: P11631 (ERA ID) is on 1,560 Hungarian and 368
Norwegian stations, and single digits elsewhere.

Verdict: build a generic reader. It is what `kr_register.py` does (published station list per
line, km per section, OSM track for geometry), except that European OSM track is not named for
its line, so the geometry step routes between consecutive OPs instead of following named track.

## OSM infrastructure relations (route=railway, route=tracks)

What it gives: (1) yes where mapped, name and often the national line number in `ref`, operator
on about half; (2) no, members are normally track ways only, stations come from proximity;
(3) yes, the relation's own ways.

Access: already in the Geofabrik extracts. Counted worldwide with one Overpass query each
(`rel[route=railway]; out tags center;`, 5.9 MB), countries assigned by the relation's centre
point. Licence ODbL.

| country | route=railway | with ref | route=tracks | with ref | example |
|---|---|---|---|---|---|
| USA | 4,053 | 96 | 7 | 0 | mostly heritage and freight names |
| China | 2,036 | 1,398 | 3 | 2 | ref 0002 京沪线, 0005 京九线 |
| France | 1,898 | 1,438 | 20 | 13 | ref 676 000, Ligne de Carcassonne à Rivesaltes |
| Poland | 864 | 823 | 26 | 17 | Linia kolejowa nr 91 Kraków Główny – Medyka |
| UK | 708 | 311 | 48 | 21 | West of England Line |
| Germany | 705 | 433 | 2,088 | 1,923 | tracks: 4080 Schnellfahrstrecke Mannheim-Stuttgart |
| India | 600 | 14 | 19 | 0 | Chennai Egmore - Chengelpet Mainline |
| Japan | 443 | 101 | 2 | 0 | 高崎線 (Takasaki Line) |
| Canada | 331 | 10 | 2 | 0 | subdivisions, heritage |
| Hungary | 307 | 270 | 2 | 1 | 29: Börgönd–Szabadbattyán–Tapolca |
| Italy | 285 | 63 | 101 | 54 | Treviglio-Cremona |
| Belgium | 276 | 266 | 0 | | L124 (no names) |
| Australia | 243 | 26 | 4 | 0 | Upfield Line |
| Spain | 202 | 125 | 8 | 1 | |
| Romania | 176 | 136 | 70 | 69 | Secția 800 București - Mangalia |
| Switzerland | 163 | 140 | 14 | 11 | KBS 730 Hochrheinbahn |
| Sweden | 147 | 75 | 4 | 2 | Västra stambanan |
| Czechia | 101 | 62 | 581 | 562 | 130 – Ústí nad Labem – Chomutov |
| Slovakia | 100 | 74 | 101 | 101 | |
| Austria | 89 | 68 | 283 | 199 | |
| Norway | 80 | 2 | 12 | 0 | Dovrebanen |
| Portugal | 77 | 59 | 1 | 0 | Linha do Douro |
| Indonesia | 70 | 2 | 0 | | |
| Taiwan | 69 | 10 | 2 | 0 | 屏東線 |
| Denmark | 66 | 18 | 4 | 0 | Gribskovbanen |
| Croatia | 62 | 37 | 14 | 14 | M604 Oštarije – Gospić – Knin – Split |
| Thailand | 46 | 16 | 0 | | สายเหนือ (Northern Line) |
| Bulgaria | 44 | 43 | 0 | | Железопътна линия 2 |
| Korea | 35 | 7 | 121 | 113 | tracks: 101 경부고속선 |
| Serbia | 30 | 25 | 13 | 0 | |
| Netherlands | 0 | | 2 | 1 | |
| Hong Kong | 0 | | 6 | 3 | |

Not measured: how much of each country's passenger track lies inside such a relation. That is
the number that decides whether it can be the geometry register, and it is a per-extract count
of the same kind as `probe_kr_ways.py` (share of main-line km in a `route=railway`/`tracks`
relation). The counts say France, Germany (tracks), Czechia, Poland, Hungary, Belgium, China and
the Balkans are worth that probe; the Netherlands, Norway and Hong Kong are not.

Verdict: build it, as a change to `extract.py` plus a small reader. It is the one source with
line names and geometry together outside Japan and Switzerland, and it is the only multi-country
source for China. It also supplies names for RINF's line numbers wherever its `ref` matches
(France, Poland, Hungary, Romania, Bulgaria, Czechia, Croatia, Belgium, Germany's tracks).

## Wikidata

What it gives: (1) line items with class, operator (P137), length (P2043), OSM relation id
(P402) and route number (P1671); (2) station adjacency qualified by line (P197 + pq:P81), and
in a few countries each station's km on its line (P81 + pq:P6710); (3) station points only.

Access: `https://query.wikidata.org/sparql`, CC0. Country-wide queries run 20-90 s each; one
429 in about 175 queries.

Measured per country. "Lines" is items whose class is a subclass of railway line (Q728937),
with P17 that country; it includes closed railways (Q357685), heritage railways, chords and, in
the UK, 3,357 ELR sections (Q113990375). "Stations" is railway-station items. "Adjacency" is
stations with at least one P197 carrying a P81 line qualifier. "Chained" is lines whose
adjacency pairs form one connected piece of 3 or more stations, which is an upper bound on
complete lines. "km" is station-line pairs with a P6710 km qualifier.

| country | lines | with P402 | with P1671 | stations | adjacency | lines with adjacency | chained | stations on chained | km pairs |
|---|---|---|---|---|---|---|---|---|---|
| Japan | 1,146 | 502 | 402 | 12,314 | 3,855 | 790 | 258 | 2,656 | 20 |
| Korea | 159 | 38 | 58 | 1,750 | 942 | 75 | 46 | 729 | 0 |
| Taiwan | 133 | 22 | 26 | 730 | 333 | 54 | 25 | 295 | 0 |
| Switzerland | 441 | 49 | 177 | 2,034 | 2,266 | 232 | 199 | 2,187 | 60 |
| France | 1,395 | 174 | 858 | 5,958 | 3,484 | 288 | 199 | 3,220 | 10 |
| Germany | 3,776 | 1,100 | 2,739 | 8,884 | 7,451 | 1,223 | 673 | 6,432 | 6,830 (684 lines) |
| UK | 4,756 | 208 | under 70 | 9,707 | 2,323 | 298 | 203 | 2,155 | 349 |
| Netherlands | 512 | 32 | 326 | 1,445 | 682 | 109 | 48 | 585 | 13 (+629 ordinals) |
| Belgium | 392 | 39 | 202 | 1,710 | 431 | 58 | 33 | 436 | 5 |
| Italy | 976 | 249 | 50 | 4,338 | 1,864 | 310 | 149 | 1,624 | 334 |
| Spain | 601 | 103 | 348 | 3,826 | 3,891 | 251 | 201 | 3,641 | 263 |
| Austria | 288 | 38 | 146 | 1,706 | 970 | 151 | 72 | 851 | 865 (47 lines) |
| Poland | 1,308 | 90 | 629 | 5,706 | 1,121 | 194 | 106 | 995 | 0 |
| Czechia | 609 | 260 | 159 | 1,439 | 1,674 | 283 | 105 | 1,086 | 19 |
| Sweden | 665 | 54 | 449 | 1,611 | 736 | 102 | 65 | 602 | 1,629 (130 lines) |
| Norway | 149 | 17 | under 10 | 977 | 1,317 | 102 | 84 | 1,187 | 0 |
| USA | 3,271 | 852 | 110 | 5,652 | 2,277 | 230 | 159 | 2,056 | 162 |
| Canada | 458 | 221 | 24 | 1,173 | 349 | 33 | 24 | 331 | 0 |
| Australia | 581 | 98 | 15 | 2,289 | 652 | 87 | 46 | 503 | 0 |
| India | 566 | 72 | 53 | 10,508 | 7,523 | 338 | 316 | 7,351 | 221 |
| China | 1,489 | 440 | 564 | 14,594 | 9,381 | 986 | 487 | 7,775 | 0 |
| Hong Kong | (in China) | | | | 177 | 26 | 22 | | 0 |
| Singapore | 20 | 12 | 6 | 254 | 210 | 11 | 10 | 189 | 0 |
| Thailand | 52 | 30 | 2 | 744 | 625 | 31 | 18 | 273 | 0 |
| Indonesia | 291 | 31 | 4 | 1,117 | 873 | 83 | 63 | 847 | 78 |

Hong Kong items have P17 = China, so the country filter finds none; the Hong Kong row is
stations located in Hong Kong (P131) instead.

What the numbers say:

- **Against countries we have built**: Wikidata's chained lines are 252 of Japan's 593
  register lines, 42 of Korea's 83, 25 of Taiwan's 36, 172 of Switzerland's 402. So in a
  well-edited country roughly half the lines have a usable station chain. Not a spine.
- **India is the exception.** 7,351 stations sit on 316 chained lines, and Indian Railways has
  about 7,300 stations. The lines are IR's sections ("Mathura–Vadodara Section", 106 stations;
  "Konkan Railway", 69), which is the right unit for a completion tracker. This looks like a
  bulk import and is the best line inventory for India found anywhere.
- **China is strong**: 9,381 stations with line adjacency, 487 chained lines, and they are
  the real named lines (京广铁路 219 stations, 沪昆铁路 169, 京沪高速铁路, metro lines).
  Together with OSM's 1,398 referenced `route=railway` relations it is enough to try China.
- **Germany has km**: 6,830 station-line pairs on 684 lines carry P6710, the station's
  kilometre on that VzG line, and 2,739 lines carry their VzG number in P1671. Sweden (1,629
  pairs on 130 lines) and Austria (865 on 47) likewise.
- **P1671 (route number) is the join to RINF** for names: France 858 lines, Germany 2,739,
  Poland 629, Sweden 449, Spain 348, Netherlands 326, Belgium 202.
- The UK's 4,756 "lines" are mostly 3,346 Network Rail ELRs (P10271), which are engineering
  references, not passenger lines.
- P3858 links a line to its Wikipedia route diagram template (China 457, Japan 464, UK 255).

Verdict: use as glue in every reader (names for RINF numbers, English names, P402 to match OSM
relations) and as the line source for India, and as a check for China. Not a generic reader of
its own except possibly for India.

## Wikipedia route diagrams

What it gives: (1) one diagram per line article, named by the article; (2) yes, ordered rows,
and in de.wikipedia usually the km of every station with closed stations marked by the `e`/`ex`
icon prefixes; (3) no.

Measured: de.wikipedia has 7,731 articles using `Vorlage:BS-header`; en.wikipedia's
`Template:Routemap` has 30,165 transclusions (many are template pages, each diagram usually
lives in its own template); ru 960, ko 908 (Routemap), ja 662 (BS-header alone), pt 592, sv
200, zh 194. The de.wikipedia rows look like
`{{BS3|STR|HST||4,274|[[Bahnhof Mannheim ARENA/Maimarkt|Mannheim-SAP-ARENA/Maimarkt]]}}`:
several track columns per row, parallel lines drawn side by side, km jumps written as their own
rows. No existing parse or dump of them turned up.

Access: `action=raw` per page (fine); the MediaWiki API's `list=embeddedin` answered 429 to most
paging requests from this session even at one request per 1.5 s.

Verdict: a per-line fallback, as zh.wikipedia already is for Alishan. Not a generic reader: the
icon grammar differs between wikis and a diagram mixes the line with its branches.

## GTFS aggregators

What it gives: services with stop sequences (routes are services, not lines); stations as stops
with coordinates; shapes sometimes. No line inventory and no chainage.

- **Mobility Database** catalogue (`https://files.mobilitydatabase.org/feeds_v2.csv`, no login):
  4,548 GTFS feeds in 95 countries. Country-wide rail feeds in it include DELFI (Germany), SNCF,
  NMBS/SNCB, Trenitalia and Trenord, Renfe and FGC, Polregio and the Polish regional operators,
  Amtrak, VIA Rail, Entur (Norway, with SJ and Vy, key needed), Trafiklab (Sweden, key needed),
  and an unofficial Indian Railways feed. None for Korea, China, Hong Kong, Thailand; one for
  Singapore. The Netherlands shows 0 feeds (OVapi exists but is not catalogued).
- **Transitous** (`https://github.com/public-transport/transitous`, feeds listed per region in
  `feeds/*.json`, about 90 countries and regions) republishes its processed feeds for download at
  `https://api.transitous.org/gtfs/` (per-feed zips, e.g. `de_DELFI.gtfs.zip` 325 MB,
  `ch_opentransportdataswiss26.gtfs.zip` 269 MB). That is the one place to fetch many countries'
  timetables in one form.

How far it gets: it tells which stations passenger trains call at and which sections trains
actually run over, which is what `drop_unridden_sections` currently asks of OSM routes. It does
not say which line a section belongs to. Useful later as the "is this ridden" check and for
countries where OSM route relations are thin; not for the register.

## EuroGlobalMap, EuroRegionalMap, GISCO, INSPIRE

- **EuroGlobalMap** (1:1M, EuroGeographics open data): `RAILRDL` has use, TEN-T, gauge,
  electrification, category, speed class; no name or line number. `RAILRDC` stations have names
  but only "important main railway stations". Metros and trams excluded.
- **EuroRegionalMap** (1:250k, spec ERM 2026): `RAILRDL` does have `NAMN1`/`NAMN2` (name) and
  `RCO` (railroad code) and `LEN`, but they may be filled as unknown, lines under 1.6 km are
  dropped, geometry is generalised, no stations in order. Whether the fields are filled per
  country was not checked (it needs the data download from EuroGeographics' open data site,
  which was not tried).
- **GISCO** railway layers derive from EuroGlobalMap. **INSPIRE TN-RA** has a `RailwayLine`
  with `railwayLineCode`, but it is published per country through national portals in varying
  shape; RINF is the better way into the same national numbers.

Verdict: skip all four.

## National registers that pair with these

Single-country, but each is one download and fills what the multi-country sources lack:

- **France, SNCF open data** (`ressources.data.sncf.com`): `lignes-par-statut` has `code_ligne`,
  `lib_ligne` (the line's full name), status, PK at each end and geometry; `liste-des-gares`
  has every station with `code_ligne`, `pk` and a `voyageurs` flag. With RINF's matching
  `830000-1` ids this is a complete French register with names, and it does not need RINF at
  all.
- **Germany, DB InfraGO Streckennetz** (GeoPackage and CSV, CC BY 4.0, via Mobilithek and
  GovData): track centre lines, operational facilities, km markers. The Schienennetz-shaped
  geometry register for DB lines.
- **UK**: Network Rail's track model (VectorLinks by ELR and track id, OGL) is now on the Rail
  Data Marketplace, which needs an account; `openraildata/network-rail-gis` on GitHub was
  archived in August 2024. ELRs are engineering references, so the UK has no passenger line
  register of the kind this project uses; geofurlong.com (CC BY 4.0) maps it.
- **USA, FRA NARN** and **Canada, NRWN** (open.canada.ca, OGL-Canada): track by subdivision
  name, with owner and user; NRWN also has stations (name, type, user) and marker posts.
  Subdivisions are the North American equivalent of a legal line.
- **Australia, Geoscience Australia Foundation Rail Infrastructure** (CC BY 4.0, ArcGIS REST at
  `services.ga.gov.au/gis/rest/services/Foundation_Rail_Infrastructure/MapServer`): a geometry
  register with names, plus stations. Correction 2026-10-02 (Australia agent): the service
  has no `ROUTENAME` / `SECTIONNAME`; each segment carries `name` (the state's line name,
  "MAIN SOUTHERN RAILWAY"), owner, gauge, tracks, length_km and operational_status, with no
  passenger flag and no node topology (`au_sources.md`).
- **Hong Kong, MTR open data**: `https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv`
  (every line except Light Rail, stations in sequence) and `light_rail_routes_and_stops.csv`.
- **India, datameet/railways** (CC0): stations, trains and schedules; distances per train, not
  per line. Wikidata is better for lines.

## Hobby compilations and prior art

- **RailMiles** (railmiles.me) has a UK mileage engine; nothing downloadable.
- **Railway Codes** (railwaycodes.org.uk) lists every ELR with mileages; no open licence.
- **geofurlong.com**: ELR explorer built on Network Rail's track model, CC BY 4.0 samples.
- **viaduct.world** and **Trainlog** log journeys against timetables and OSM, not lines.
- No GitHub repository of multi-country line lists turned up; the closest is RInfData, which
  is code over RINF.

## Recommendation

Two generic readers are worth building, in this order.

1. **A RINF reader** (`rinf_register.py`): one SPARQL pull per country (or the Zenodo dump),
   sections as register sections with their km as `chain`/`km_official`, passenger OPs as
   stations, junction OPs as `junction` ends, geometry by shortest path over OSM rail between
   consecutive OPs and rejected where the path is more than a few percent off the section's
   length. Line names from, in order: a national list where one exists (SNCF), OSM
   `route=railway`/`route=tracks` whose `ref` is the line number, Wikidata P1671, and failing
   those "first OP - last OP". Filter Germany's future-dated sections. This gives most of the
   EU plus Switzerland and Norway their mainline networks; metros, trams and the non-RINF
   private railways stay OSM route relations, as today.
2. **An OSM infrastructure-relation reader**: keep `route=railway` and `route=tracks`
   relations in `extract.py` (ref, name, operator, member ways), and read them as a named-track
   register the way `kr_register.py` reads named ways. It supplies names to the RINF reader and
   is the main source for China. Measure first, per country, the share of passenger track km
   inside such a relation.

Countries next, easiest first:

1. **Hong Kong and Singapore**: MTR's CSV, and Singapore's station codes (which encode line and
   order) with Wikidata's 10 complete chains. Hours each, no generic reader needed.
2. **France**: RINF plus SNCF's two datasets (names, PKs, passenger flag); OSM has 1,438
   referenced `route=railway` relations to cross-check. The first country for the RINF reader,
   because SNCF lets every RINF number be checked against a named line.
3. **Belgium, Netherlands, Austria, Czechia, Poland, Hungary, Portugal, Slovakia, Romania,
   Bulgaria, Slovenia**: the same RINF reader once it works for France. Belgium's names are its
   numbers (L36); Czechia, Poland, Hungary, Romania, Bulgaria and Croatia have OSM relations
   whose names carry the number. The Netherlands' ids are station-code pairs (Asd-Zp), which
   need a name map, and NS lines are not legally named anyway.
4. **Germany**: RINF (1,492 lines, 33,406 km) plus OSM `route=tracks` (1,923 with the VzG
   number) plus Wikidata's station km on 684 lines. Large, but every piece is there; non-DB
   lines from OSM.
5. **Italy, Spain, Sweden**: RINF works but needs a name source (Italy's ids are RFI internal
   codes; Spain's Adif ids hide the line number; Sweden's are track-section numbers), and Spain
   needs FGC, Euskotren, FGV and the metros from OSM.
6. **India**: Wikidata's IR sections (316 complete chains, 7,351 stations) plus OSM track.
   Worth a pilot; nothing else comes close for India.
7. **China**: OSM's 2,036 `route=railway` relations (1,398 with national line codes) plus
   Wikidata's 487 chains. Measure track coverage with the relation probe first.
8. **Australia, USA, Canada**: their own geometry registers (GA's ROUTENAME, NARN and NRWN
   subdivisions). One reader could serve NARN and NRWN, since both are subdivision-named track.
9. **UK, Norway, Denmark, Ireland, Thailand, Indonesia**: no line register with names (UK),
   no line ids in RINF (Norway, Croatia) or ids per section (Denmark, Ireland); Thailand and
   Indonesia have only OSM and partial Wikidata. Later.
