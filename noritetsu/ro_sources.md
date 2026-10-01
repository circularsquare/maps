# Romania register sources (built 2026-10-01)

What the Romanian build reads, where each piece came from, and what is still wrong with it.
Romania is built with `rinf.py`; how that reader works is in its docstring, and the per-country
entry is `rinf_countries/ro.py`, whose docstring explains every table in it. Downloads live in
`data/raw/rinf/ro/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch ro                                             # RINF, ~20 s
curl -L -o data/raw/romania-260930.osm.pbf https://download.geofabrik.de/europe/romania-260930.osm.pbf   # 330 MB
$env:OSMIUM_POOL_THREADS=1; python extract.py --region ro --pbf data/raw/romania-260930.osm.pbf         # 35 s; delete the .pbf after
python build_model.py --region ro --register rinf:data/raw/rinf/ro       # 30 s
python build_tiles.py --region ro                                        # 15 s
python check_model.py --region ro
python rinf.py --dry ro          # the reader alone, with its full log (every rejection)
```

The dated file came from https://download.geofabrik.de/europe/romania.html (data to
2026-09-30). `--fetch` also pulls Wikidata, which the build does not use (`"wikidata": None`
in ro.py); delete `data/raw/rinf/ro/wikidata.json` after a fetch, or rinf.py reads it for
English names.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01 (`sections.json`: 2,329 sections, 271 line ids, 10,542 km; `points.json`: 2,187
  points, every one with a coordinate, many of them 0.5-4 km from the station). Only CFR SA
  (`0053_IM`) is registered; there are no validity periods.
- **OpenStreetMap**, Geofabrik `romania-260930.osm.pbf`, ODbL: track, stations, 241 route
  relations (77 trains, 152 trams, 12 metro) and 256 `route=railway`/`route=tracks` relations.
  Two numberings sit on the same `ref` key there: the timetable's ("Secția 202 Simeria -
  Filiași", route=railway, old_ref holding the 2011-2021 number) and CFR SA's ("Linia 316
  Brașov - Războieni", route=tracks), plus Hungarian, Serbian and Bulgarian border relations.
  Only the Secția relations are read (`ro_rel`).
- **ro.wikipedia**, the infobox length (`LÄNGE`, `Lungime`) of each line article and the
  "Magistrale feroviare în România" table, via the MediaWiki API, retrieved 2026-10-01. These
  are the figures in `check_model.REGISTER["ro"]`; mostly uncited.
- **Wikidata** P2043 for Oravița - Anina (Q12723301) only. Its P1671 route numbers were
  looked at and not used: they mix the current timetable, the 2011-2021 one (906 Pitești -
  Curtea de Argeș beside 106) and CFR SA's ids, and label 100 "Magistrala CFR 900".
- Not reached: data.gov.ro, where the railway authority (ARF) publishes every operator's
  timetable as XML, timed out on 2026-10-01 (connection, not a refusal). That is what would
  say which lines still have trains; see "Still off".

## Names: the timetable's numbers, not RINF's

RINF's ids are CFR SA's infrastructure line numbers from its network statement ("300",
"100A", "316", "301Bb"). Riders never see them. The CFR Călători timetable, ro.wikipedia
("Magistrala CFR 300", "Calea ferată 202") and OSM number lines by the timetable's secții,
and the two schemes share digits but not meanings: CFR SA's 203 is Copșa Mică - Sibiu - Podu
Olt - Piatra Olt, the timetable's 203 is Bartolomeu - Zărnești. So the ref is the timetable
number and the id never becomes one by rule.

- Most CFR SA ids lie on one Secția relation, and rinf.py numbers them from it.
- 36 ids are in `FIXED`: relations mapped only in part (108 has 11 ways), not at all
  (Căciulați - Snagov Plajă, 706), or ids lying on two (Ploiești's links; Teiuș - Coșlariu).
- 13 ids are split (`SPLIT`, through rinf.py's `fix` hook) at the stations where the
  timetable changes number: 316 Brașov - Deda - Războieni into 400 and 405; 203 into 208, 200
  and 201; 412 into 400 and 401; 100A into 100 and 119; and 123, 130, 200A, 219, 222, 511,
  804A, 807A, 143.
- 100 ids are left out (`NOT_LINES`): Bucharest's freight belt (301x), yard links at Ploiești,
  Brașov, Galați and Barboși, Constanța's port, industrial branches and connecting curves.
- Where two timetable lines share track, the register gives it to one: București - Ploiești
  is 300 (500 and 1000 start at București Nord too, so 1000 is no register line at all);
  Beclean pe Someș - Dej is 400 though 401 runs over it.

Names are "Magistrala 300 București – Brașov – Cluj-Napoca – Oradea" for the main lines and
"Secția 202 Simeria – Filiași" for the rest: "Magistrala CFR N" is ro.wikipedia's title for the
main lines and what riders call them, "Secția N" is how OSM and the timetable head the others
(ro.wikipedia titles those by route, "Calea ferată Simeria–Petroșani–Târgu Jiu–Filiași", with
no form for the number). The route comes from the Secția relation's name. English names are
"Line 202 (Simeria – Filiași)".

No `colours/ro.csv`: CFR publishes no line colours.

## Stations

RINF writes a halt's kind after its name ("Aghires PO", "Apa HM", "Alius hcv.") and OSM does
not, or writes its own ("Aghireș hc", "Halta Balda"). `ro_fix` puts the kind in brackets so
rinf.py tries the name with and without it, writes out G-ral, Tr. and I. L., and moves 113
points whose RINF coordinate is far from the OSM station of exactly their name (Buzău 2.1 km,
Satu Mare 2.4, Drăgășani 4.1, Beclean pe Someș 4.2, Oradea 4.1); `name_m` is 1500 m. Matched
passenger points went from 1,450 to 1,738 of 2,019. Lalașinț is made a junction
(`NOT_STOPS`): its OSM halt is 1 km off on 200's other alignment and cost 200 12.3 km.

The 281 still unmatched are mostly halts on closed lines with no OSM station (Banloc, Ciacova,
Jamu Mare, Oravița - Iam) and RINF points with no OSM node at all, among them two real
stations: Țăndărei and Slobozia Nouă.

## Named trains

OSM Romania maps every train as its own relation, by number ("IR 1582 Constanța => București
Nord", "R 3127 Arad => Brad", "Tren R9132/4: Calafat - Craiova"). `build_model.looks_like_service`
has a `ro` branch (landed 2026-10-01): a train number in the name or ref makes it a named train,
except service=commuter (the Gara de Nord - Aeroport and Nord - Obor shuttles). 38 relations,
29 lines after grouping, 4,157 km. Romania has no branded interval lines, so OSM's train
relations are nearly all named trains.

## Counts (2026-10-01)

- 86 register lines, 8,968 km (CFR SA's RINF has 10,542 km; the rest is freight, closed or
  untraceable, see below). 1,664 stations on a register line.
- 200 lines in all: those 86, 29 named trains, 85 other OSM lines (trams in eight cities, the
  Bucharest metro M1-M5, the airport and Obor shuttles listed in both directions, MÁV's border
  patterns). 2,339
  stations. ro.pmtiles 2.1 MB.

## Check

`python check_model.py --region ro`: against RINF's own section lengths, 86 lines of 2 km or
more, median 0.999, 5 off by more than 5% (all 1.05-1.08, short lines whose stops RINF places
far from their stations). Against ro.wikipedia, 31 lines, 25 within 5%. The other six:

- **Magistrala 500** (0.88): built from Ploiești Sud; București - Ploiești Sud (59 km) is 300's.
- **203 Bartolomeu - Zărnești** (0.85), **210 Alba Iulia - Zlatna** (0.88), **126 Timișoara -
  Cruceni** (0.90): each starts on another line's track (200, 200A, 122), which the article
  counts; 126 also loses Ram. Modoș - Timișoara Vest (2.8), a junction section no OSM route
  covers.
- **125 Oravița - Anina** (1.05): Brădișoru de Jos - Dobrei traces 7.8 km on the line's own
  track for RINF's 6.3.
- **Magistrala 800** (1.04): it also has București Nord - Băneasa (CFR SA's 301P, on OSM's 800
  relation), and the trace runs 5.8 km longer than RINF at Neptun and Mangalia.

Left out of REGISTER: 107, 213 and 314 (closed or unbuilt in part), 102, 105, 119, 306, 701 and
806 (the article covers a different stretch), 400 (the article's 560 km does not add up from
its own sub-articles; the build has 516) and 600 ("395 / 187").

## Still off, and why

- **Which lines still have passenger trains is not settled.** OSM's passenger routes cover
  2,930 of the register's 9,148 km (32%), so they cannot say which lines are closed. What the
  build drops is what OSM no longer maps as `railway=rail`: 73 merged sections are rejected
  and 182 RINF pieces cannot be traced (171 with a point more than 1.5 km from any track), which takes out Roșiori - Turnu Măgurele, Alexandria - Zimnicea, Jimbolia -
  Lovrin, Oravița - Iam and - Berzovia, Jebel - Liebling and - Giera, Cărpiniș - Ionel,
  Buziaș - Jamu Mare, Caransebeș - Bouțari, Remetea Mică - Radna, Bălăușeri - Praid, Holod -
  Oradea Est, Vișeu de Jos - Borșa, Ilva Mică - Rodna, Botiz - Bixad, Mărășești - Panciu,
  Dorohoi - Leorda, Vama - Moldovița, Siret - Dornești, Crasna - Huși, Căciulați - Snagov and
  Medgidia - Negru Vodă. A line closed to passengers but still mapped as rail with its stations
  stays drawn, since a section between two stops is never questioned. The ARF timetable XML on
  data.gov.ro would settle it.
- **Gaps in lines that run.** Magistrala 300 has a 2.8 km hole at Valea Drăganului - Poieni
  (the OSM track graph has no path there, from either RINF's points or OSM's stations).
  119 Timișoara - Jimbolia starts at Săcălaz: Ram. Pav. CFR - Ram. 2 Jimbolia has no OSM path,
  and the end-to-end retrace lies under half on 119's own relation, so rinf.py rejects it.
  107 lost Costești - Colțu (20.5 km) the same way.
- **Junction sections dropped for want of an OSM route** (26 sections, 201 km): the border
  stubs (Curtici, Vicșani, Halmeu, Carei, Stamora Moravița, Golenți, Fălciu, Valea Vișeului),
  the Jilava pieces of 102, Ograda - Țăndărei on 701 (no OSM station at Țăndărei), and whole
  short lines ending at a halt OSM does not have: 114 Strehaia - Motru, 222 Turceni -
  Drăgotești, 225 Ucea - Victoria, 303 I. L. Caragiale - Filipeștii de Pădure, 305 Câmpia
  Turzii - Turda (RINF types Turda a switch), 318 Ineu - Cermei, 220, 221A.
- 104 sections are kept although their length disagrees with RINF's (`length off` in the
  log), because the trace runs on the line's own OSM relation or two traces agree; RINF's
  points are often far from the track, so one piece reads short and the next long.
