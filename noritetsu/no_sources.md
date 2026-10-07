# Norway register sources (built 2026-10-03)

What the Norwegian build reads, where each piece came from, and what is still wrong with it.
Norway is built with `no_register.py` (its docstring has the method). Downloads live in
`data/raw/no/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python no_register.py --fetch          # Banenettverk GML (11 MB zip, 43 MB unpacked) + Entur's rail lines (8 MB)
python no_register.py --fetch-entur    # Entur's rail lines alone
python no_register.py --dry            # the register half alone, no OSM: lines, km, what is left out (~1 min)
python extract.py --region no --pbf data/raw/norway-latest.osm.pbf     # delete the .pbf after
python tools/slot.py -- python build_model.py --region no --register no_register:data/raw/no
python tools/slot.py -- python build_tiles.py --region no
python check_model.py --region no
```

`tools/rebuild.py` needs `"no": "no_register:data/raw/no"` in its `REGISTER`.

## Sources

- **Banenettverk** (Bane NOR SF, through Geonorge; "Jernbane - Banenettverk"; NLOD 1.0, "no
  conditions apply"): `https://nedlasting.geonorge.no/geonorge/Samferdsel/Banenettverk/GML/Samferdsel_0000_Norge_4258_Banenettverk_GML.zip`
  (11.0 MB, EPSG:4258; the feed says updated 2025-03-06, data.norge.no says 2025-09-01).
  Atom feeds for the other formats: `nedlasting.geonorge.no/geonorge/ATOM-feeds/Banenettverk_AtomFeed{GML,FGDB,SOSI,PostGIS}.xml`.
  It holds Bane NOR's reference line of every banestrekning (`Banelenke`: banenavn, banekortnavn,
  banestatus, baneformål, chainage at both ends), `Stasjonsnode` (844: stations and halts,
  current and closed), `Kilometerpunkt` (4,406) and chain breaks. Every Banelenke is listed
  twice under two ids with one geometry (8,172 records, 4,128 distinct).
- **Entur journey planner** (`https://api.entur.io/journey-planner/v3/graphql`, open, sends
  `ET-Client-Name: noritetsu-hobby-railmap`; Entur's data is NLOD): every rail line in
  Norway's national timetable with its journey patterns' stop places and coordinates,
  `data/raw/no/entur_rail_lines.json`, fetched 2026-10-03: 50 lines (Vy, SJ Norge/SJ Nord,
  Go-Ahead, Flytoget, Flåmsbanen, Vy Tåg, SJ, Snälltåget's "Nattåg"), 2,714 patterns, 483 rail
  stop places. The query is `ENTUR_QUERY` in no_register.py.
- **ERA RINF** (`python rinf.py --fetch no`, `data/raw/rinf/no/`, 2026-10-03; not used by the
  build, see below). `rinf_countries/no.py` exists only so the fetch runs.
- **OpenStreetMap**, Geofabrik `norway-latest.osm.pbf` (2026-10-03), ODbL, extracted by the
  managing session.

## Choosing the register

Three candidates, measured 2026-10-03:

- **ERA RINF**: no line ids, as multi_sources.md said, but each section's `era:nationalLine`
  is labelled with Bane NOR's line ("B05-Nordlandsbanen", B01-B40), and sections carry their
  geometry and Bane NOR's track records. Its 375 sections are 3,234 km in four unconnected
  pieces, though: Støren is in it with no section to it (Dovrebanen, Rørosbanen and
  Nordlandsbanen's Trondheim end are cut apart there), Drammen - Galleberg and Barkåker - Sem
  (Vestfoldbanen), Magnor - border, Kopperå - border, Bjørnfjell - border and half of
  Ofotbanen are missing, and no border point is the end of any section. Per line: Dovrebanen
  371 km of 483, Sørlandsbanen 442 of 549, Bergensbanen 321 of 381, Vestfoldbanen 82 of 129,
  Kongsvingerbanen 83 of 113, Meråkerbanen 54 of 72, Ofotbanen 26 of 43.
- **OSM named track** (kr recipe): `probe_kr_ways.py --region no` on the 2026-10-03 extract:
  99.0% of main and branch rail km (4,321 of 4,366) carry a name, and the names are the
  banestrekninger (Nordlandsbanen 722.8 km, Sørlandsbanen 560.5, Dovrebanen 514.4...). It
  would work, but OSM names some stretches after their tunnel ("Romeriksporten" 30.9 km,
  "Oslotunnelen" 5.2) instead of their line, and Bane NOR's own file is the register, with
  status and chainage.
- **Banenettverk**: picked. Bane NOR's own centre lines, complete (3,931 km of line in use,
  against the 3,962 km of standard-gauge network no.wikipedia's "Jernbane i Norge" gives),
  named, with status and chainage, under NLOD. It has no passenger flag and no halt list of
  its own that says what is served: Entur's timetable supplies both.

## How no_register.py builds the lines

(see the module docstring for the full method)

- Lines are banestrekninger in use (status I, purpose B or P). Connector strekninger fold in
  (`fold`): "<A> spor mot <B>" and "Oslo S mot <B>" are the start of B (Meråkerbanen from Hell,
  Flåmsbana from Myrdal, Gardermobanen from Oslo S); others ("Ofotbanen Katterat") are A's.
- `fill_gaps`: Bane NOR files some of a line's own track under another code (Drammenbanen
  through Asker station runs over "Askerbanen spor mot Drammenbanen" and "... mot
  Spikkestadbanen" links); paths of up to 2 km over other lines' links join the pieces.
  Without it Drammenbanen lost Asker - Lier (11.3 km) and Skøyen - Lysaker.
- `snap`: Banenettverk's links meet on coordinates up to a metre apart (only 3,595 of 7,558
  link ends share one exactly), and some end up to 70 m short of the link they join (Oslo S
  mot Gardermobanen 69 m, Bratsbergbanen at Nordagutu 72 m). Vertices within 1 m become one
  point; a link end touching nothing gets a bridge to the nearest other link within 100 m.
  Without it Bergensbanen was 6 pieces and the Oslo S approaches were cut off.
- Stops are Entur's rail stop places that some pattern calls at, placed on the lines their
  trains' paths run over near them (Oslo S on Hovedbanen, Gardermobanen and Østfoldbanen);
  halts (Bane NOR's type I) only on their own line (Leirsund is 13 m from Gardermobanen's
  track but a Hovedbanen halt). Bus stops in rail patterns (rail replacement; 219 stop
  places) are left out by the stop place's transport mode.
- Sections: n02.py's absorbing search, plus border points (eEU00234-37) and junctions between
  lines trains run over as section ends. A line ending within 1.5 km of a stop on the line it
  leaves starts at that stop (Arendalsbanen at Nelaug, Bergensbanen at Hønefoss, Sørlandsbanen
  at Gulskogen, Vestfoldbanen at Porsgrunn). Sections lying along others of their own line are
  dropped (`redundant`: Ofotbanen's Rombak - Søsterbekk past Katterat's loop, 15.0 km;
  Sørlandsbanen's Nodeland - Vennesla over the Dalane curve, 20.2 km, which no train takes:
  they all reverse at Kristiansand), and junctions only one line still reaches are fused away.
- Served: each of the 455 distinct pairs of consecutive calls credits the shortest track
  between its two stops (4 found no path: Ski - Vevelstad, Langhus - Ski and two Hønefoss
  pairs, all replacement-bus patterns); sections no pair credits are left out.

## What is in the register (build of 2026-10-03)

25 register lines, **3,689 km**, 352 stations of which 15 are junctions or border points; 51
OSM lines beside them (Oslo T-bane 1-5, the Oslo trams, Trondheim's Gråkallbanen, Bergen's
Bybanen and Fløibanen, and the train products: F4-F7, RE10/11/20/30, R12-R75, L1/L2/L4,
FLY1/2, F1;70 Oslo - Stockholm, F8/30 Luleå - Narvik, Nattåg 93); 76 lines, 583 stations,
9,377 route-km in all. Named trains (`rules/no.py`): Nattåg 93 Stockholm - Narvik (38 km in
Norway).

Left out as no train runs over them (the reader's log): Roa-Hønefossbanen (Roa - Hønefoss, 32
km, freight), Østfoldbanen østre linje Rakkestad - Sarpsborg (24 km; R22 ends at Rakkestad),
Stavne-Leangenbanen Leangen - Lerkendal (4.4 km), a Hovedbanen connection at Lindeberg (3.1
km). Lines in use with nothing built: Solørbanen, Numedalsbanen (Kongsberg - Flesberg),
Brevikbanen, Filipstadbanen (all freight). Not read at all (status or purpose): Valdresbanen,
Namsosbanen, Krøderbanen, Thamshavnbanen, Setesdalsbanen, Flekkefjordbanen, Rjukanbanen,
Gamle Vossebanen, Ålgårdbanen, Urskog-Hølandsbanen (closed or museum), Ringeriksbanen and the
new alignments Drammen - Kobbervik, Nykirke - Barkåker, Sandbukta - Moss - Såstad and
Kleverud - Sørli - Åkersvika (planned in the 2025 data; OSM has no track or routes on them,
and RINF still files the old Nykirke - Skoppum - Barkåker sections as valid to 2026-11-26).

Borders: all four crossings to Sweden end at RINF's border points, the ids Sweden's register
uses: eEU00234 Kornsjø (Østfoldbanen ends 1 m from it), eEU00235 Magnor - Charlottenberg
(Kongsvingerbanen, 7 m), eEU00236 Kopperå - Storlien (Meråkerbanen, 16 m), eEU00237 Bjørnfjell
- Riksgränsen (Ofotbanen, 3 m). No `borders.EXTRA` is needed. A border section counts as run
over when a train calls at its Norwegian stop and next at a stop abroad (Kongsvinger -
Charlottenberg, Kornsjø - Ed, Kopperå - Storlien, Bjørnfjell - Riksgränsen).

## Check

`python check_model.py --region no`: 15 lines in `REGISTER`, all within 0.92 - 1.01 or off by
exactly what their note says:

| line | built | published | ratio | |
|---|---|---|---|---|
| Nordlandsbanen | 722.3 | 729 | 0.99 | WP |
| Dovrebanen | 479.7 | 482.8 | 0.99 | BN chainage Eidsvoll - Trondheim S |
| Sørlandsbanen | 540.0 | 549 | 0.98 | WP |
| Rørosbanen | 381.1 | 382 | 1.00 | JiN |
| Bergensbanen | 369.0 | 380.6 | 0.97 | BN chainage |
| Østfoldbanen vestre linje | 166.9 | 170 | 0.98 | WP |
| Vestfoldbanen | 127.8 | 129 | 0.99 | WP (ca.) |
| Gjøvikbanen | 121.4 | 123.8 | 0.98 | WP |
| Raumabanen | 113.8 | 114.2 | 1.00 | WP |
| Kongsvingerbanen | 114.9 | 113.4 | 1.01 | BN chainage to the border |
| Meråkerbanen | 70.6 | 71.5 | 0.99 | BN chainage to the border |
| Arendalsbanen | 36.1 | 36.3 | 0.99 | BN chainage |
| Flåmsbana | 19.2 | 20.2 | 0.95 | WP |
| Follobanen | 20.2 | 22.0 | 0.92 | from where it leaves Østfoldbanen's tracks |
| Ofotbanen | 38.0 | 43 | 0.88 | Narvik station is ~4 km short of km 0 at the ore harbour |

Sum of the register lines (3,689 km) against no.wikipedia's "Jernbane i Norge": 3,962 km of
standard-gauge network, of which the closed, museum and freight-only lines above are about 270
km.

## Still off, and why

- **Planned alignments**: Banenettverk's file is the 2025 one. When Nykirke - Barkåker (Horten
  station; RINF's old sections end 2026-11-26) and Drammen - Kobbervik open, refetch
  (`python no_register.py --fetch`) and rebuild: the status turns I and the old track's
  sections lose their trains. Entur already lists Horten stasjon (off the network, 0.8 km
  from the old line).
- **Junction sections OSM has no route on** are dropped by build_model even when Entur's
  trains run there (one, 0 km, this build). A national feed through `gtfs_served.py` would
  rescue them; Entur's GTFS is 609 MB (below).
- **Stops on a parallel line**: a station (not a halt) whose trains' paths run over a parallel
  line's track within 400 m goes on that line too: Lindeberg on Gardermobanen as well as
  Hovedbanen (Gardermobanen gets a Lindeberg - Kløfta section). Holding stations to their Bane NOR node's line broke Lillestrøm and Oslo
  S (Bane NOR files each junction station under one line) and was dropped.
- **Refs from OSM**: build_model hands an OSM twin's ref to the register line, so Rørosbanen
  carries "25" (NSB's old route number on an outdated OSM relation) and Raumabanen "R 65".
- **OSM noise kept as lines**: "Bybanen til Fyllingsdalen" and its "deltrinn 1/2" (project
  relations mapped as route=light_rail, 0.9 - 4.8 km) and a 0.1 km tram "Linje 1". The rules
  hook only judges route=train.
- **Colours**: no `colours/no.csv`. Bane NOR publishes no line colours for banestrekninger;
  Entur's colours are per product (Vy's red regional, light-blue regional express, green L),
  which the OSM product lines already carry.
- **English names** are the en.wikipedia article titles (`NAME_EN`).

## Big files not fetched

- Entur national GTFS, `https://storage.googleapis.com/marduk-production/outbound/gtfs/rb_norway-aggregated-gtfs.zip`
  (609 MB on 2026-10-03, NLOD, all modes; `rb_norway-aggregated-gtfs-basic.zip` 106 MB;
  Transitous mirror `https://api.transitous.org/gtfs/no_Entur.gtfs.zip` 523 MB). Not needed by
  the reader (the journey planner gives the patterns); only for gtfs_served's check.
