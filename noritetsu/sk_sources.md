# Slovakia register sources (built 2026-10-01)

What the Slovak build reads, where each piece came from, and what is still wrong with it.
Slovakia is built with `rinf.py`, the generic ERA RINF reader; the per-country settings and
corrections are `rinf_countries/sk.py` (its docstring has the reasoning). Downloads live in
`data/raw/rinf/sk/` and `data/raw/sk_wikidata.json` (gitignored). Nothing needed a login or a
key.

## Run

```powershell
python rinf.py --fetch sk                                                # RINF, ~10 s
curl -L -o data/raw/sk-slovakia-260930.osm.pbf https://download.geofabrik.de/europe/slovakia-260930.osm.pbf
$env:OSMIUM_POOL_THREADS = 1; python extract.py --region sk --pbf data/raw/sk-slovakia-260930.osm.pbf   # 35 s; delete the .pbf after
python build_model.py --region sk --register rinf:data/raw/rinf/sk       # 35 s
python build_tiles.py --region sk                                        # 15 s
python check_model.py --region sk
python rinf.py --dry sk          # the reader alone, with its full log
```

The Geofabrik file is 345 MB; the dated name from https://download.geofabrik.de/europe/slovakia.html
was used. `rinf.py --fetch sk` also writes `data/raw/rinf/sk/wikidata.json`; it was moved to
`data/raw/sk_wikidata.json`, because the build must not read it (below). Do the same after a
re-fetch.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01 (`sections.json`: 1,081 sections on 133 line ids, all ŽSR's `0056_IM`, one
  version each; `points.json`: 1,040 points).
- **OpenStreetMap**, Geofabrik `slovakia-260930.osm.pbf` (data to 2026-09-30), ODbL: track,
  stations, 271 route relations, and 242 `route=railway`/`route=tracks` relations, which carry
  the timetable numbers (below).
- **Wikidata**, CC0: items with a route number (P1671) and P17 Slovakia, labels in sk and en,
  lengths (P2043), 206 rows in `data/raw/sk_wikidata.json`. Not read by the build; it is where
  the published lengths in `check_model.REGISTER["sk"]` come from. P2043 is the "dĺžka" in the
  line's sk.wikipedia infobox; 120, 143, 160, 170 and 180 were checked against the articles.
  143's article says 49 km where Wikidata has 52; the article's figure is used.
- **sk.wikipedia**, "Zoznam železničných tratí na Slovensku" (retrieved 2026-10-01), for which
  lines have no passenger trains (below).

## Line ids and numbers

RINF's id is ŽSR's own internal number for a stretch of line (`SK_2601`, `SK_2701`), which
riders never see and which does not follow the public lines. **The number riders know is the
timetable (KCP) number: "trať 120 Bratislava - Žilina"** heads each table of ZSSK's timetable,
and is what sk.wikipedia and Wikidata's P1671 use. So the ref is the KCP number and the name
the timetable's heading, "120 Bratislava – Žilina".

There is no rule from ŽSR's id to the KCP number, so numbers come from OSM's `route=railway`
relations, which carry the KCP number in `ref` (their note says they represent the line in the
timetable). OSM also has a second numbering, **ŽSR's network-statement sections, on
`route=tracks` relations** ("105A" Košice - Kraľovany); `sk_rel` ignores every route=tracks
relation, every number outside 100-199 and every relation of another network (MÁV, PKP, ÖBB
and SŽ lines reach over the border; PKP's Id-12 numbers 487-489 and 559 are mapped on ŽSR's
border stubs). Rimavská Sobota - Poltár (175) has no ref in OSM and is recognised by its
Wikidata item. The route part of the name is the relation's sk.wikipedia title (the KCP
relations have no `name`). Wikidata is kept out of the build (`"wikidata": None`): its labels
are "železničná trať A – B" and rinf.py would prefer them.

Of RINF's 133 ids, 9 are skipped (below) and 3 split into 8 (below), leaving 129: 9 numbered
by `fixed`, 99 from OSM relations, and 21 with no number (yard links and curves; most end at
junctions nobody rides and are dropped).

## What ŽSR's RINF gets wrong, and the fixes

All in `sk_fix`, run through rinf.py's `fix(secs, points)` hook (added for Slovakia on
2026-10-01):

- **Stations typed 110** ("technical change"): 89 points, among them Piešťany, Pezinok, Bytča,
  Považská Bystrica, Senica, Levoča, Lipany, Sabinov and Tatranská Lomnica. As 110 they were
  junctions and merged away: line 120 had no stop between Leopoldov and Trenčín. The 79 named
  like a station are retyped 10; yard groups ("Barca St. 1"), km posts, AH/EE points and Modrý
  most stay 110.
- **Coordinates 1-3.4 km off** for 12 passenger points (RINF's Prakovce zastávka sits on
  Gelnica zastávka, Malá Maňa on Maňa zastávka, Turzovka zastávka 2.6 km out). Their RINF
  coordinate is dropped and rinf.py places each at the OSM station of its exact name (all 12
  placed).
- **Lengths that are not lengths**: Čremošné - Horná Štubňa obec on 170 is 5856.0 (metres,
  read as 5.856), Horná Štubňa obec - Odb. Dolná Štubňa 238.421 and Varín - Žilina-Teplička
  214.0/201.0 (kilometre positions, dropped). Unfixed, RINF's total reads 9,163 km where the
  network is about 3,600; that is the 9,660 km in multi_sources.md.
- **A junction at a stop** joined to it by a 0 km section (Lastovce odb. - Lastovce on 191)
  becomes that stop, or build_model drops the 0 km section and 191 falls into two pieces.
- **One id, several timetable lines**: SK_2902 runs Fiľakovo - Lučenec - Zvolen - Kremnica -
  Horná Štubňa - Martin - Vrútky (160, the end of 150, 171 and 170), SK_3104 Zvolen - Banská
  Bystrica - Podbrezová (170, 172), SK_3012 carries 144 on the end of 140. `SPLITS` cuts them
  at the stations where the timetable lines meet. Where two timetable numbers share track it
  goes to the line it completes: Zvolen - Hronská Dúbrava to 150 (Nové Zámky - Zvolen) rather
  than 171, Odb. Dolná Štubňa - Diviaky to 170 rather than 171.
- **ŽSR's section lengths leave out station track**: they add up to 3,151 km against 3,255 km
  of crow-fly distance between the same points, and 120 is 157.9 km in RINF against 203
  published. Hence `tol_abs` 1.0 (as Hungary's MÁV), and check_model's comparison against
  RINF's own lengths reads a median 1.105: that column says nothing here.

Skipped ids (`skip_line`): the Žilina-Teplička marshalling yard and its links (SK_2601L7543,
L7544, SK_2606, 2607, 2609, 2610, 2612; their traces ran along 180 and got 180's own Varín -
Potok odb. section dropped as a second track pair) and the 1520 mm broad gauge (SK_3321
Haniska - Maťovce, SK_3341 Čierna nad Tisou), freight only.

## Lines with no passenger trains

sk.wikipedia's list marks 22 timetable lines "pravidelná osobná prevádzka prerušená". 18 are
built: 112, 113, 117, 124, 134, 142, 144, 161, 163, 164, 165, 166, 167, 168, 186, 187, 192,
195 (320 km). OSM independently has no passenger route on exactly those 18 (0.00-0.06 of each
line covered, where every line with trains is 0.47-1.00). They stay on the map, greyed as not
running and out of completion (`SUSPENDED` in sk.py, through rinf.py's `suspended` hook and
not_running.py), as Japan's closed lines are, so a rider who went before can still record it.
Chvatimech - Hronec (SK_3141, unnumbered) likewise. The list also names 115 (not built: only
its 3 km border stub is in RINF and it is dropped), 122 (the Trenčianske Teplice tram, not in
RINF; an OSM tram line), 136 (Komárno - Kolárovo: OSM has no track, its RINF sections are
unplaceable) and 135 Nové Zámky - Komárom, which OSM has an Os service on, so 135 stays
running.

## Not in RINF: OSM lines

Bratislava's trams (DPB, the 4 routes OSM maps) and Košice's (DPMK, 14 routes), the
Trenčianske Teplice electric tram (TREŽ), the Starý Smokovec - Hrebienok funicular, and the
Čiernohronská železnica (760 mm heritage, its three routes). The Tatra electric railway
(183, 184) and the Štrba rack railway (182) ARE in RINF, under ŽSR, and are register lines.
Other heritage narrow gauge (Oravská and Kysucká lesná železnica, Vihorlatská úzkokoľajka,
the Košice children's railway, Nitrianska poľná železnica) has no OSM passenger route and is
among the 97 narrow-gauge ways build_tiles leaves out as track no line touches.

Every train service (ZSSK's Os, REX, R, Ex, IC, the S lines around Bratislava, Žilina and
Košice, RegioJet, Leo Express, Arriva) is an OSM line over the register. Since 2026-10-01 `sk`
is in build_model's EU_TRAIN_REGIONS: the EC, EN, ICE and Nightjet trains are named trains
(3 route_masters after merging, 595 km); ZSSK's IC, Ex and R products stay lines.
SuperCity Pendolino Košičan, R Zakarpatia and R Kysučan are single named trains the rule does
not flag. No `colours/sk.csv`: ŽSR publishes no line colours.

## Counts (2026-10-01)

- 70 register lines, 3,376 km: 66 with a timetable number, 4 short unnumbered links
  (Jelšovce - Zbehy, Bratislava-Rača - Bratislava-Vajnory, Bratislava-Nové Mesto - Odb.
  Vinohrady, Chvatimech - Hronec). 19 lines (354 km) greyed as not running.
- 172 lines in all with the OSM ones, 1,082 stations, 9,383 route-km. Tiles 1.3 MB.
- 929 RINF passenger-typed points (after the retyping), 902 an OSM station (891 distinct), 1 by
  distance alone (Pezinok zastávka, now Pezinok-Grinava). The 27 without one are freight
  points, closed halts and the unbuilt Kolárovo line; Banská Belá zastávka, Lisková, Uľanka
  and Lábske Jazero zastávka are worth a look in OSM.

## Check

`python check_model.py --region sk`, 37 lines against published lengths, worst 0.93:

| line | built | published | ratio | |
|---|---|---|---|---|
| 113 Zohor – Záhorská Ves | 14.1 | 14.5 | 0.97 | no trains |
| 116 Kúty – Trnava | 68.8 | 67.5 | 1.02 | |
| 117 Jablonica – Brezová pod Bradlom | 11.6 | 11.7 | 1.00 | no trains |
| 120 Bratislava – Žilina | 197.2 | 203.0 | 0.97 | RINF ends it at Žilina predmestie |
| 124 Trenčianska Teplá – Lednické Rovne | 17.2 | 17.3 | 1.00 | no trains |
| 126 Žilina – Rajec | 21.2 | 21.3 | 1.00 | |
| 127 Žilina – Mosty u Jablunkova | 37.2 | 37.3 | 1.00 | |
| 128 Čadca – Makov | 26.7 | 26.2 | 1.02 | |
| 129 Čadca – Zwardoń | 19.9 | 21.0 | 0.95 | figure runs on to Zwardoń |
| 133 Galanta – Leopoldov | 43.6 | 43.9 | 0.99 | with Trnava – Sereď |
| 134 Šaľa – Neded | 18.9 | 18.9 | 1.00 | no trains |
| 140 Nové Zámky – Prievidza | 103.3 | 111.6 | 0.93 | Nové Zámky – Šurany is on 151 |
| 143 Trenčín – Chynorany | 48.5 | 49.0 | 0.99 | article; Wikidata 52 |
| 144 Prievidza – Nitrianske Pravno | 10.8 | 11.1 | 0.98 | no trains |
| 145 Prievidza – Horná Štubňa | 37.6 | 37.0 | 1.02 | |
| 152 Štúrovo – Levice | 52.1 | 52.0 | 1.00 | |
| 153 Zvolen – Čata | 105.1 | 106.0 | 0.99 | |
| 154 Hronská Dúbrava – Banská Štiavnica | 19.6 | 19.7 | 1.00 | |
| 160 Zvolen – Košice | 225.4 | 233.0 | 0.97 | |
| 162 Lučenec – Utekáč | 41.0 | 41.2 | 1.00 | |
| 163 Katarínska Huta – Breznička | 9.8 | 10.0 | 0.98 | no trains |
| 165 Plešivec – Muráň | 40.7 | 40.9 | 1.00 | no trains |
| 166 Plešivec – Slavošovce | 23.7 | 24.0 | 0.99 | no trains |
| 167 Rožňava – Dobšiná | 26.0 | 26.1 | 1.00 | no trains |
| 168 Moldava nad Bodvou – Medzev | 15.4 | 15.3 | 1.01 | no trains |
| 170 Zvolen – Vrútky | 96.9 | 96.0 | 1.01 | |
| 173 Červená Skala – Margecany | 90.1 | 92.6 | 0.97 | |
| 174 Brezno – Jesenské | 77.3 | 82.0 | 0.94 | Brezno – Brezno-Halny is on 172 |
| 180 Žilina – Košice | 241.0 | 238.9 | 1.01 | |
| 181 Kraľovany – Trstená | 56.5 | 56.5 | 1.00 | |
| 182 Štrbské Pleso – Štrba | 4.6 | 4.8 | 0.97 | rack railway |
| 183 Poprad-Tatry – Starý Smokovec – Štrbské Pleso | 28.2 | 29.1 | 0.97 | Tatra electric railway |
| 184 Starý Smokovec – Tatranská Lomnica | 5.9 | 5.9 | 1.00 | Tatra electric railway |
| 186 Spišská Nová Ves – Levoča | 12.6 | 12.7 | 1.00 | no trains |
| 187 Spišské Vlachy – Spišské Podhradie | 9.3 | 9.3 | 1.00 | no trains |
| 192 Trebišov – Vranov nad Topľou | 32.0 | 31.9 | 1.00 | no trains |
| 193 Prešov – Humenné | 60.2 | 60.4 | 1.00 | |

Left out of REGISTER because the published figure covers a different stretch: 110, 114, 121,
123, 125, 132, 135, 161, 164, 169, 191 (figures run into Czechia, Austria, Hungary, Poland or
Ukraine), 130 (the built line adds Bratislava hl. st. - Devínska Nová Ves and Palárikovo -
Šurany), 150 and 171 (shared track went to 150 and 170), 172 (Wikidata's 39 km is a different
stretch), 185 (built with the Studený Potok - Tatranská Lomnica branch), 188 (Košice - Kysak
is on 180), 190 (built with Trebišov - Čeľovce - Výh. Slivník), 194 (below).

## Still off, and why

- **Timetable lines share track; a section goes to one line.** 141 Leopoldov - Kozárovce is
  three pieces (Lužianky - Nitra is on 140, Zlaté Moravce - Odb. Topoľčianky on 151), 140
  starts at Šurany, 174 at Brezno-Halny, 171 at Hronská Dúbrava. Riders crossing those stretches
  record them on the other line.
- **194 Prešov - Bardejov is 29.4 km**: Prešov - Kapušany is on 193, and Kľušov - Bardejov is
  not in RINF at all (no section, no Bardejov point), so the line stops at Kľušov.
- **120 stops at Žilina predmestie**, short of Žilina station; RINF has no section between
  them. The unnumbered Žilina predmestie - Budatín odb. curve is dropped as unridden.
- **190 is two pieces**: Trebišov - Čeľovce - Výh. Slivník meets the main line at a junction
  RINF's main-line sections have no point at. 161 is two pieces because it runs through
  Hungary (Kalonda - Ipolytarnóc - Malé Straciny), and 188's Orlov - Čirč stub is cut off where
  the junction sections at Plaveč have no OSM route.
- **OSM route coverage is patchy** and is what decides junction-ended sections: 131
  Bratislava - Komárno, a busy line, has routes over only 26% of it. Stop-to-stop sections
  never depend on it, so nothing on 131 is lost, but 28 junction-ended sections (61 km) were
  dropped, mostly yard links and curves.
- **Rejected or untraceable**: 112's Plavecké Podhradie - Plavecký Mikuláš (no track mapped
  near Plavecký Mikuláš), 136 Komárno - Kolárovo (no track in OSM), Piešťany - Vrbové (SK_2751,
  no track), and four freight links in Bratislava and Lužianky.
