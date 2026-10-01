# Croatia register sources (built 2026-10-01)

What the Croatian build reads, where each piece came from, and what is still wrong with it.
Croatia is built with `rinf.py`; how the reader works is in its docstring, and its
per-country entry is `rinf_countries/hr.py`. Downloads live in `data/raw/rinf/hr/` and
`data/raw/hr_*` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch hr                                              # RINF + Wikidata, ~10 s
curl -L -o data/raw/hr-latest.osm.pbf https://download.geofabrik.de/europe/croatia-latest.osm.pbf   # 200 MB
$env:OSMIUM_POOL_THREADS=2; python extract.py --region hr --pbf data/raw/hr-latest.osm.pbf   # 20 s; delete the .pbf after
python inspect_region.py --region hr
python build_model.py --region hr --register rinf:data/raw/rinf/hr     # 15 s
python build_tiles.py --region hr                                      # 5 s
python check_model.py --region hr
python rinf.py --dry hr          # the reader alone, with its full log
```

`--station-areas` on the extract makes no difference here: Croatia maps no station only as
an area. The HŽI reports below were fetched with `curl -k`: hzinfra.hr's certificate chain
does not verify with Windows curl (exit 60). Both are public PDFs.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01, EU open data. `sections.json` has 1,166 rows, which is 583 sections, each listed
  twice. None has a validity period. `points.json` has 1,120 rows (points are listed twice too),
  every point with a WKT coordinate. Everything is under HŽ Infrastruktura (`0078_IM`): 53 line ids,
  2,435 km.
- **OpenStreetMap**, Geofabrik `croatia-latest.osm.pbf`, downloaded 2026-10-01, ODbL. It
  gives track, stations and 287 train route relations (HŽPP 262), and 115 `route=railway` /
  `route=tracks` relations. 55 of those carry HŽI's line number as `ref` with HŽI's full
  name ("M202 Zagreb Glavni kolodvor – Karlovac – Rijeka").
- **HŽ Infrastruktura, "Statistika HŽ Infrastrukture za 2025."** (July 2026),
  https://www.hzinfra.hr/wp-content/uploads/2026/07/Statistika-HZ-Infrastrukture-za-2025.pdf
  (`data/raw/hr_statistika_2025.pdf`). Two tables from it are used:
  - Table 1.4, "Građevinske dužine pojedinih pruga u 2025.", is every line's label, name and
    constructional length, plus the sections closed to traffic. It is the source of the
    names in `hr.py`'s `ROUTES` and of the lengths in `check_model.REGISTER["hr"]`.
  - Table 3.7, "Promet putničkih vlakova po dionicama pruga", gives passenger train
    movements for 2025 on every line section. It is the source of `hr.py`'s `SUSPENDED`.

  The tables were read by word position with PyMuPDF; the scripts are throwaways and were not
  kept.
- **HŽ Infrastruktura, Izvješće o mreži 2026** (network statement),
  https://www.hzinfra.hr/wp-content/uploads/2025/05/2026_I_IOM.pdf (`data/raw/hr_iom_2026.pdf`).
  Only its network total is used: 2,617 km, of which 315 km is double track. It has no
  per-line length table, so the statistics report above was used for those.
- **Wikidata** (`wikidata.json`), CC0: 56 rows, items with P1671 and P17 Croatia, Croatian
  labels. These are used only for the "built as no line" log line. Names come from `ROUTES`.

## How `hr.py` reads the ids

RINF has no `era:nationalLine` for Croatia, which is why `multi_sources.md` and HANDOFF said
"no line ids". **But every section's URI carries HŽI's line number.**
`SectionOfLine_M202__M20214J` is section 14 of M202. `SectionOfLine_M101__M10103D_M10103L` is
section 3 of M101, with its right (D) and left (L) tracks together. These are the numbers HŽI
publishes:
- M is an international line;
- R is a regional line;
- L is a local line.

Every one of the 53 ids has an OSM `route=railway` or `route=tracks` relation of that ref
(M502's is tagged "M502-1;M502-2"), and most have a Wikidata item with that P1671. `hr_fix`, through rinf.py's existing `fix` hook, reads each section's line from its
URI, and the id is the number (`rule_certain`). **No rinf.py change was needed.** Two lines
are filed in two parts, and each pair is one line:
- M402A/M402B are the two tracks of Sava – Zagreb Klara around the marshalling yard.
- M502_1/M502_2 are HŽI's own M502-1 Zagreb GK – Velika Gorica and M502-2 Velika Gorica –
  Novska.

The OSM relations were not needed for numbers. They confirm them.

`hr_fix` rebuilds the section list from `sections.json` rather than correcting the one
`load_rinf` made. `load_rinf` chooses between validity versions by the key (line id, start,
end). With every Croatian line id empty, two lines' sections between the same two points
became one key, and one of them was lost: Čakovec – Čakovec Buzovec is on both L101 and M501,
and Zagreb Klara – Zagreb Rk PS on both M402 and M403. Here a section is simply one URI.

`hr_fix` also corrects one RINF error. R106's last section, R10611J, is filed as Hromec →
Đurmanec, 2.724 km, the same pair as the section before it read backwards. It is really
Hromec → the Slovenian border towards Rogatec. HŽI's table 3.7 gives Đurmanec – Đurmanec DG
as 5.922 km, which is 3.198 + 2.724. Croatia's RINF has no point at that border. SŽ's RINF
does: "Rogatec d.m.", EU00220, at 46.21231 N 15.77691 E. It is added under that uopid, and
R106 builds at 0.99 of HŽI's length instead of 0.89.

**Names** are HŽI's own, from table 1.4: "M202 Zagreb GK – Rijeka", in English "Line M202
(Zagreb GK – Rijeka)". HŽI marks a border end "DG" (državna granica, the state border). It is
left off where two places remain, as `si.py` does with "d. m.":
- "M101 Savski Marof – Zagreb GK"
- "R101 Buzet – Pula"

GK is Glavni kolodvor (main station), ZK is Zapadni kolodvor, and RK OS/PS are the
marshalling yard's departure and arrival groups.

**Not running.** `SUSPENDED` lists lines with no passenger trains in table 3.7: a "-" in every
section, or "zatvoreno za promet" (closed to traffic). It also lists three lines with only a
handful of trains in the year: M602 (1), M408 (24) and M401 (30). The `suspended` hook greys
them through `not_running.py`, as in Slovakia. Three of them are actually drawn, because
they run stop to stop and would otherwise show as running:
- M606 Knin – Zadar, 94.6 km;
- L103 Karlovac – Kamanje, 28.7 km;
- L213 Lupoglav – Učka, 5.6 km.

The rest are dropped anyway as unridden junction sections.

## What RINF has and what is built

**RINF compared with HŽI's network.** HŽI's network is 2,617 km. RINF's 53 ids hold 2,435 km.
What RINF leaves out is all closed:
- R103 Knin – Ličko Dugo Polje – border, the Una line, 57.9 km closed;
- L210 Sisak Caprag – Petrinja;
- L213 beyond Učka to Raša;
- L102 beyond Harmica to Kumrovec;
- L205's Čaglin – Našice cement stretch.

**Built: 38 register lines, 2,317 km.** That is 2,425 km traced against 2,435 km of RINF
length, before build_model drops unridden junction sections. Against RINF's own section
lengths, the median is 0.996 and one line is off by more than 5%: R102, where RINF puts
8.11 km on Hrvatska Kostajnica – Volinja for 5.26 km of track. The traces agree both ways.

Dropped by build_model as unridden freight track. None of these has an OSM passenger route:
- M304 Metković – Ploče (22.7 km). OSM has no station on it at all.
- The Zagreb yard and bypass lines M401, M402, M404, M405, M407, M408, M409 and M410. M403
  is rejected in rinf.py: 2.29 km of trace for 1.71 km.
- M602 Škrljevo – Bakar, M603 Sušak – Rijeka Brajdica, L207 Bizovac – Belišće, L211 Ražine –
  Šibenik Luka and L212 Rijeka Brajdica – Rijeka.

Border stubs with no OSM route are dropped:
- Novo Drnje – Koprivnica DG on M201 (3.9 km);
- Slavonski Šamac – DG on M303;
- Kotoriba – DG on M501;
- Tovarnik – DG on M104;
- Volinja – DG on R102;
- Erdut – DG on R104;
- Gunja – Drenovci DG on R105.

**Other lines** (88 OSM lines in all) are HŽPP's trains, grouped by OSM under the timetable
line number:
- "Vlak 23" is Vlak 2300, 2301 and so on;
- the B, IC and ICN long-distance trains are one master each;
- SŽ's and MÁV-Start's cross-border trains, ŽRS's Dobrljin – Banja Luka, Zagreb's ZET trams
  and Osijek's GPP trams.

`build_model.looks_like_service` has an hr branch (`HR_TRAIN`, 2026-10-01). It flags B
(brzi, fast), IC, ICN, EC and EN trains as named trains, and leaves the "Vlak NN" groups as
lines. That flags 12 lines, 3,261 km:
- B 188 Dalmacija, B 182, ICN 52, B 78, B 74, B 174, IC 58 Podravka and "Vlak B 170";
- B 175, and three short fast trains with `service=regional`: B 70 Duga Resa, B 76 Siscia
  and B 79 Zagorje. These are still single trains, not an interval product.

## Counts (2026-10-01)

- 126 lines (38 register, 88 OSM, of which 12 are named trains), 638 stations, 10,218 route-km
  (6,957 without the named trains). Tiles 0.6 MB.
- 519 RINF passenger-typed points. 464 are an OSM station (462 distinct), 7 of them by
  distance alone; all 7 are spelling variants such as "Zagreb GK" → "Zagreb Glavni kolodvor"
  and "Staro Toplje" → "Staro Topolje".
- 55 have no OSM rail station within 1 km and are not stops:
  - Knin – Zadar's 14 halts: Benkovac, Škabrnje, Bibinje... There are no trains on that line.
  - M304's Metković, Opuzen, Rogotin and Kula Norinska.
  - Halts on M604 in Lika: Medak, Lički Osik, Ličko Cerje, Raduč, Rudopolje, Pađene, Plavno,
    Zrmanja, Malovan, Štikada...
  - Halts on R202: Nemetin, Novi Dalj, Borovo-Trpinja, Ladimirevci...
  - Valpovo (L207, no trains), Markovac and Zagreb Borongaj.

## Check

`python check_model.py --region hr` compares against HŽI's table 1.4: 35 lines, 27 within 5%.

| line | built | published | ratio |
|---|---|---|---|
| M604 Oštarije – Knin – Split | 320.0 | 322.1 | 0.99 |
| R202 Varaždin – Dalj | 249.1 | 249.8 | 1.00 |
| M202 Zagreb GK – Rijeka | 227.5 | 227.9 | 1.00 |
| M104 Novska – Tovarnik | 183.0 | 185.4 | 0.99 |
| M502 Zagreb GK – Sisak – Novska | 117.0 | 116.8 | 1.00 |
| R201 Zaprešić – Čakovec | 99.3 | 100.7 | 0.99 |
| L204 Banova Jaruga – Pčelić | 95.7 | 95.8 | 1.00 |
| M606 Knin – Zadar (greyed) | 94.6 | 95.4 | 0.99 |
| R101 Buzet – Pula | 90.3 | 91.1 | 0.99 |
| M103 Dugo Selo – Novska | 84.1 | 83.4 | 1.01 |
| M201 Botovo – Dugo Selo | 74.9 | 79.5 | 0.94 |
| L203 Križevci – Bjelovar – Kloštar | 60.9 | 62.0 | 0.98 |
| R105 Vinkovci – Drenovci | 49.1 | 50.9 | 0.96 |
| M302 Osijek – Strizivojna-Vrpolje | 48.0 | 48.4 | 0.99 |
| M501 Čakovec – Kotoriba | 38.9 | 42.4 | 0.92 |
| L205 Nova Kapela – Našice | 38.9 | 42.0 | 0.93 |
| L201 Varaždin – Golubovec | 33.7 | 34.6 | 0.97 |
| L208 Vinkovci – Osijek | 33.7 | 33.8 | 1.00 |
| M301 Beli Manastir – Osijek | 31.9 | 32.5 | 0.98 |
| M203 Rijeka – Šapjane | 30.8 | 30.9 | 1.00 |
| L103 Karlovac – Kamanje (greyed) | 28.7 | 28.8 | 1.00 |
| L209 Vinkovci – Županja | 27.6 | 28.1 | 0.98 |
| R106 Zabok – Đurmanec | 26.9 | 27.2 | 0.99 |
| M101 Savski Marof – Zagreb GK | 26.7 | 26.8 | 1.00 |
| L206 Pleternica – Velika | 25.1 | 25.0 | 1.01 |
| R104 Vukovar-Borovo naselje – Erdut | 21.9 | 26.1 | 0.84 |
| M607 Perković – Šibenik | 21.4 | 22.5 | 0.95 |
| M102 Zagreb GK – Dugo Selo | 20.7 | 21.2 | 0.98 |
| M303 Strizivojna-Vrpolje – Slavonski Šamac | 19.8 | 23.3 | 0.85 |
| R102 Sunja – Volinja | 19.6 | 21.6 | 0.91 |
| M601 Vinkovci – Vukovar | 18.4 | 18.9 | 0.97 |
| L101 Čakovec – Mursko Središće | 17.4 | 17.9 | 0.97 |
| L214 Gradec – Sveti Ivan Žabno | 12.4 | 12.5 | 0.99 |
| L202 Hum-Lug – Gornja Stubica | 10.6 | 10.8 | 0.98 |
| M605 Ogulin – Krpelj | 5.8 | 6.2 | 0.95 |

Every miss is a border stub, or a figure where HŽI's two tables disagree:
- **M303** (0.85), **R104** (0.84), **M501** (0.92), **R102** (0.91), **M201** (0.94): a
  stub to the border with no OSM route is dropped. The stubs carried 15, 2, 14, 16 and 823
  passenger trains in 2025. R102 also has a RINF misallocation: table 3.7's Sunja – Volinja
  is 19.669 km, and the build gives 19.6.
- **L205** (0.93): the published figure is HŽI's 60.493 less the closed Čaglin – Našice
  cement stretch (18.515). The build also lacks the freight end Našice Grad – Našice cement
  (1.9 km).
- **M605** (0.95) and **M607** (0.95): table 1.4's constructional lengths are 6.153 and
  22.503. Table 3.7 and RINF give 5.842 and 21.449, and the build matches those.

Not listed in the check: the lines dropped as unridden (above), L102 (4.8 km open of 38.5)
and L213 (5.6 km, closed).

## Still off, and why

- **Passenger track dropped with the freight.** These sections have passenger trains in
  HŽI's 2025 figures, but no OSM route, so build_model drops them as unridden:
  - Novo Drnje – Koprivnica DG on M201: 823 trains, the trains to Gyékényes.
  - M304 Metković – Ploče: 137 trains, which looks like a seasonal summer service (the
    Sarajevo – Ploče train). OSM has no stations there.
  - M405 Zagreb ZK – Trešnjevka: 546 trains.
  - M407 Sava – Velika Gorica: 118 trains.

  The GTFS check (HŽPP feed) should decide each of these.
- **Lines kept as running with almost no trains**, which only a per-section check can catch:
  - R104 Vukovar-Borovo naselje – Dalj: 1 passenger train in 2025, but it runs stop to stop
    so it is kept.
  - M303 Strizivojna-Vrpolje – Slavonski Šamac: 106 trains.
  - L204 Daruvar – Pčelić: 124 trains.

  The `suspended` hook greys whole lines only.
- **No stops where OSM has no station.** The Lika halts on M604 and several R202 halts have
  no OSM station node, so trains that call there do not show them. Some may be closed halts.
- **Zagreb's funicular (Uspinjača)** is left out of the tiles: build_tiles drops it as
  "track no line touches", because OSM has no route relation for it.
- Lines have no colour. HŽPP publishes none, and no OSM route relation carries one.
