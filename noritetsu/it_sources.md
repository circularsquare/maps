# Italy register sources (built 2026-10-02)

What the Italian build reads, where each piece came from, and what is still wrong with it.
Italy is built with `rinf.py`; how the reader works is in its docstring, and the per-country
entry is `rinf_countries/it.py` (its docstring has the rules in short). Downloads live in
`data/raw/rinf/it/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch it                                              # RINF + Wikidata, ~30 s
curl -L -o data/raw/it/italy-latest.osm.pbf https://download.geofabrik.de/europe/italy-latest.osm.pbf   # 2.24 GB
$env:OSMIUM_POOL_THREADS=2; python extract.py --region it --pbf data/raw/it/italy-latest.osm.pbf   # 3.5 min; delete the .pbf after
python inspect_region.py --region it
python build_model.py --region it --register rinf:data/raw/rinf/it     # 2 min
python build_tiles.py --region it                                      # 1.5 min
python check_model.py --region it
python rinf.py --dry it          # the reader alone, with its full log
```

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-02, EU open data. 3,650 sections (one row each; 79 carry a validity start, none an
  end), 3,206 points, every one with a coordinate. 327 line ids, 19,357 km. Infrastructure
  managers: RFI `0083_IM` (3,203 sections), FER `3525_IM`, FERROVIENORD `0064_IM`, Ferrovie
  del Sud Est `3572_IM`, Ferrotramviaria `3857_IM`, EAV `3856_IM`, La Ferroviaria Italiana
  `3456_IM`, Ferrovie del Gargano `PY63_IM`, GTT `3908_IM`, Ferrovie Udine Cividale `3379_IM`.
- **OpenStreetMap**, Geofabrik `italy-latest.osm.pbf`, downloaded 2026-10-02, ODbL. 97,310
  track ways, 39,901 stops, 841 route relations (618 train) under 254 route_masters, 424
  `route=railway`/`route=tracks` relations. A passenger route relation covers 76% of main-line
  km and 68% of branch-line km (`inspect_region.py`), less than most built countries: Sardinia,
  much of the south and the Milano - Venezia high-speed services have no route relations.
  The extract includes **San Marino** (4 ways of the Rimini - San Marino narrow gauge, a short
  restored stretch: no route, so build_tiles leaves it out as track no line touches) and **the
  Vatican** (the Roma San Pietro - Città del Vaticano branch, no route: drawn, owned by no line).
- **it.wikipedia**, CC BY-SA, retrieved 2026-10-02 through the API: the station tables
  (`{{Percorso ...}}` rows with progressive km) of 50 line articles, used for the chainage
  (CH) figures in `check_model.REGISTER["it"]` and the Altavilla - Vicenza fix; infobox
  `lunghezza` (WP) where an article has one; the regional network pages ("Rete ferroviaria
  della Lombardia" etc.) for which lines exist. Throwaway scripts, not kept.
- **Wikidata**, CC0: P2043 (length) of the 670 Italian railway-line items (Q728937 and
  subclasses, trams included) that have one and an it.wikipedia article (WD in REGISTER). `wikidata.json` from `--fetch` holds 111 rows with P1671 (route
  number), which are not line numbers here (tram and metro lines, a few FS timetable tables)
  and name nothing: Italian lines get no ref.
- **RFI, "La rete oggi"** (https://www.rfi.it/it/rete/la-rete-oggi.html, 30 June 2026): network
  16,909 km (fondamentali 6,450, AV/AC 1,096, complementari 9,502, nodo 957). The per-line list
  is in the PIR on RFI's ePIR portal and was not used.

## How `it.py` reads the ids

RFI's ids are internal codes, not public numbers: `F1-F2` ... `F77-F78` the fundamental lines
(one number per direction), `C1` ... `C262` complementary, `N1` ... `N8` the city nodes
(Torino, Milano, Venezia, Genova, Bologna, Firenze, Roma, Napoli), `AV/AC1-AV/AC2` ... high
speed, plus `0000` (a mixed bag), `M` (the Messina strait ferry berths) and `TR0965`. RINF
holds them as F lines that run node to node: F41-F42 is Milano Rogoredo - PM Lavino, the
Milano - Bologna line between the two nodes, and N2 holds every line inside Milano. The other
managers' ids are names ("Lecce-Gallipoli", "SALV" = Saronno - Laveno, "FNI1" = Brescia -
Edolo).

**No public numbering exists** (I looked: RFI's PIR lists lines by name and class, the
"Fascicolo Linea" numbers are internal; Wikidata's P1671 on Italian items is tram, metro and a
few old timetable numbers), so lines carry no ref, as in Greece. `NAMES` gives each id an
RFI-style name with an en dash:
- complementary lines by their ends: "Empoli – Siena", "Lucca – Aulla";
- fundamental lines by the cities whose node they start from: "Milano – Bologna", "Firenze –
  Roma (Direttissima)", "Firenze – Roma (linea lenta)", "Roma – Formia – Napoli";
- high speed: "AV Roma – Napoli", "AV Milano – Bologna", "AV Treviglio – Brescia";
- node lines: "Nodo di Milano" etc. (one line per city: Milano's is 141 km of 48 sections).

Pieces of an id that stay apart after the fixes are named alone ("Cuneo – Limone" and
"Breil-sur-Roya – Ventimiglia" are C10). Twelve groups join pieces that it.wikipedia treats
as one line (it.py's GROUP): Domodossola – Novara (C29-C34: RFI files its own single track
beside the Simplon line as four ids), Mantova – Monselice, Paola – Cosenza, Battipaglia –
Potenza – Metaponto, Castel Bolognese – Ravenna, Ferrara – Ravenna – Rimini, Cremona – Mantova,
Vicenza – Treviso, Lecco – Sondrio – Tirano, Savigliano – Saluzzo – Cuneo, Decimomannu –
Iglesias, Modena – Sassuolo.

**Fixes to RINF** (`it_fix`, each logged as `RINF: fix:`):
- **Point type 140 read as a station.** 140 marks where two managers meet, and in Italy that is
  often a main station: Foggia, Taranto, Lecce, Modena, Reggio Emilia, Bologna Centrale, Udine,
  Ferrara, Cancello, Benevento, Arezzo, Seregno... Typed 140 they were never stops, so every
  section into them ended at a "junction" and build_model dropped the ones with no OSM route:
  Apricena - Foggia (39 km of the Adriatica), Grottaglie - Taranto, Massafra - Taranto. With
  the fix, 133 km more is built.
- **Id 0000** (46 sections, 209 km) is shared out by the uopids of each piece (ZERO): border
  stubs to the Brenner, Tarvisio, San Candido, Ventimiglia, Luino and Iselle borders go to their
  lines; Germagnano - Ceres to Torino – Ceres; Bicocca - Catenanuova to Caltanissetta Xirbi –
  Catania; Pollenza - Tolentino to Civitanova Marche – Albacina; Orte - Capena to the
  Direttissima. Two pieces are lines of their own: **Napoli – Cancello – Dugenta** (the first
  open sections of the Napoli - Bari line: Napoli Afragola - Acerra - Cancello, and Valle di
  Maddaloni - Dugenta) and **Villa Opicina – Sežana**.
- **Altavilla Tavernelle - Vicenza** is missing from RINF; added to Verona – Padova at 7.667 km
  (it.wikipedia's chainage, 191+471 to 199+138).
- EAV's Benevento Rione Libertà - Benevento Appia (955.0) and Sant'Angelo in Formis - S. Iorio
  (683.0) are in metres.
- FSE's Triggiano and Capurso are 30-40 km off (near Bitonto); placed at their OSM stations.
- F61-F62 also files Vairano - Sesto Campano - Venafro, which is C141 Vairano – Isernia;
  dropped from F61-F62. About 100 other sections are filed under two or more ids (mostly station throats
  and shared stretches: Palermo's passante under both C181 and C184); logged, left to
  ownership, which gives each piece of track to one line.
- Names in capitals are written in the usual case (`title_it`: "Bivio/PC S.Lucia", "Lentini
  Diramazione", "Mondovì"); these show only for junctions and the 81 RINF stations with no OSM
  station. Matching is case- and accent-blind.

Left out (`skip_line`): GTT's copy of the Canavesana (RFI's C262 is the same track), the
Messina strait ferry berths (M), and the Padova Interporto freight pieces.

**Named trains**: `build_model.looks_like_service` has an `it` branch (IT_TRAIN, landed by
maps-ee 2026-10-02): Frecciarossa, Frecciargento, Frecciabianca, Italo, InterCity/IC, ICN, plus
the shared EU_TRAIN rule (EC, EN, NJ, TGV, European Sleeper). OSM maps the Frecce and Italo
route by route with no train number. Regionale, Regionale Veloce, RegioExpress, the Leonardo
Express and the S, FL, SFM, FM suburban lines are lines. 22 lines, 9,790 km.

## Built (2026-10-02)

- **302 register lines, 16,604 km**, against RINF's own section lengths median 0.997 (298 lines
  of 2 km or more, 3 off by more than 5%: Bari – Barletta, where RINF gives Andria Sud - Andria
  and Corato - Corato Sud 13.06 km each for about 1.4 km of track; Cagliari – Decimomannu;
  Busto Arsizio – Malpensa). By manager: RFI 264 lines 15,098 km, FSE 8 / 471, FER 10 / 354,
  FERROVIENORD 10 / 319, Ferrovie del Gargano 2 / 93, EAV 2 / 90, LFI 2 / 83, Ferrotramviaria
  3 / 82, FUC 1 / 15.
- rinf.py traced 17,276 km of 17,349 km of RINF length; 20 sections rejected (no track, below);
  build_model then dropped 154 junction-ended sections (672 km) no OSM passenger route runs over.
- 653 lines in all: 351 OSM lines (26,002 km), of which 22 are named trains (9,790 km). 4,448
  stations. 99 lines carry a colour (all from OSM; no `colours/it.csv` yet). Tiles 6.8 MB;
  775 ways on pieces of track no line touches left out (Sardinia's and Calabria's narrow gauge
  with no route, funiculars, San Marino).
- 2,736 RINF passenger-typed points: 2,655 matched an OSM station, 276 by distance alone (all
  spelling variants, "S.MARIA A VICO" -> "Santa Maria a Vico"); 81 have no OSM station within
  1 km and are junctions (closed halts like Lisiera, Noceto, Palo Laziale, and junction stations
  such as Fiumetorto and Lentini Diramazione).
- Border: 20 sections to a border point added (110.6 km), 13 border points named neutrally.
  Route ends with no RINF border point on their track: Chiasso, Cantello-Gaggiolo, Monte
  Generoso and Re (Switzerland, no RINF), Nova Gorica (no RINF point at Gorizia), and two TER
  routes from Ventimiglia (a border point EU00126 exists on F9-F10, the TER routes still did
  not meet it).
- **Direttissima and linea lenta** are separate lines: Firenze – Roma (Direttissima) 238.1 km
  against a published 237.6, on OSM's `highspeed=yes` track; the linea lenta 270.1 km. They
  share only the Orte interconnection (3.0 km counted for the Direttissima).

## Check

`python check_model.py --region it`: 98 published lengths, 85 within 5%, every miss explained
in its note. The F-line figures are it.wikipedia chainage between the built line's two ends.

| line | built | published | ratio | why |
|---|---|---|---|---|
| Alessandria – Piacenza | 110.6 | 96.5 | 1.15 | RINF also files the Bressana - Broni leg (13.3 km) |
| Fiumetorto – Messina | 199.0 | 180.6 | 1.10 | old coast line via Falcone and the new Patti - Terme Vigliatore line both built |
| Milano – Bologna | 217.5 | 199.2 | 1.09 | RINF also files Rogoredo - Bivio Melegnano - Tavazzano (18.2 km) |
| AV Bologna – Firenze | 85.7 | 78.5 | 1.09 | built starts at Bologna's underground AV station (it.wikipedia says 86) |
| Nocera Inferiore – Codola | 4.3 | 4.0 | 1.07 | rounded figure |
| Genova – Pisa | 157.1 | 147.7 | 1.06 | both Vezzano - La Spezia routes, and Pisa Centrale |
| Verona – Bologna | 109.3 | 103.0 | 1.06 | the Verona Porta Vescovo leg (6.8 km) |
| Bari – Bitritto | 9.5 | 9.0 | 1.05 | rounded figure |
| Reggio Emilia – Guastalla | 30.5 | 29.0 | 1.05 | the Reggio San Lazzaro spur (2.1 km) |
| Domodossola – Novara | 85.1 | 89.6 | 0.95 | Vignale - Novara is C24, a line of its own |
| Bologna – Portomaggiore | 41.6 | 45.0 | 0.93 | FER's line starts at Bologna Roveri |
| Saronno – Seregno | 13.3 | 15.0 | 0.89 | RINF starts at Saronno Sud (13.15) |
| Cremona – Mantova | 36.4 | 62.0 | 0.59 | Bozzolo - Mantova has no track in the extract |

Among those within 5%: Brennero – Verona 1.01, Firenze – Roma (Direttissima) 1.00, Bologna –
Ancona 1.00, Ancona – Foggia 0.99, Venezia – Trieste 1.00, Milano – Torino 0.99, Battipaglia –
Potenza – Metaponto 1.00, Lecco – Sondrio – Tirano 1.00, every FSE line 0.98-1.00.

## Still off, and why

- **Real passenger track dropped because no OSM route covers it** (junction-ended sections,
  `drop_unridden_sections`). The largest: **AV Treviglio – Brescia** (53.8 km, the whole line,
  ridden by every Milano - Venezia Frecciarossa and Italo); Giave - Chilivani on Decimomannu –
  Ozieri Chilivani (24.8 km, the Cagliari - Sassari/Olbia main line; Sardinia has no Trenitalia
  route relations) and Ardara - Chilivani on Ozieri – Sassari (8.3 km); the Foggia approaches of
  Dugenta – Benevento – Foggia (17.6 km) and Foggia – Bari (10 km); the Signa - Bivio
  Samminiatello sections of Firenze – Pisa (about 16 km, where the OSM regional route runs on
  track RINF's trace did not take); Villa Opicina - Trieste on Venezia – Trieste (36 km;
  whether the Trieste - Ljubljana trains use it is unchecked); Udine – Palmanova (34.5 km,
  whole line, passenger service unchecked); the AV interconnections. 672 km in all. The GTFS check should
  decide each of these; Italy is the country where it matters most so far.
- **No track in the extract** (OSM has these as `railway=construction` with no passenger route,
  so extract.py leaves them out): Bergamo - Ponte San Pietro (doubling works), Bozzolo -
  Mantova, Decimomannu - Villamassargia - Iglesias/Carbonia (Villamassargia – Carbonia built
  as 0.6 km), Lugo - Sant'Agata. If these have reopened, OSM has not caught up.
- **The node lines** (Nodo di Milano, di Roma...) are RFI's unit, not a rider's: Nodo di Roma
  is 194 km over FL1-FL8 track. Splitting them into the it.wikipedia lines inside each node
  (Passante di Milano, Milano - Gallarate, Roma - Fiumicino...) would need a section list per
  node; not done.
- Four register lines took a ref from the OSM line merged into them ("R" on Novara – Biella and
  Santhià – Biella, "2" on Martina Franca – Lecce, "FUC" on Udine – Cividale): not RFI numbers.
- The Simplon tunnel's Italian half (Iselle - border, 0.4 km in RINF) has no path; the EC Basel
  - Milano owns about 25 km of tunnel and approach way the register does not cover.
- Lines have no colours of their own beyond OSM's (99 lines, mostly metros and Trenord).

## Timetable feed (not wired; for the managing session)

Per `gtfs_sources.md`: Trenitalia's GTFS converted from its NeTEx on the national access point
(`raw.githubusercontent.com/deryclem/trenitalia-gtfs/refs/heads/main/gtfs-trenitalia.zip`, CC
BY 4.0, weekly, high speed + Intercity + regional), plus Trenord (Transitland
`f-u0n-trenord`) for Lombardy, Italo (`github.com/deryclem/italo-gtfs`) for the AV lines, and
for the regional managers in this register FNM/Trenord (FERROVIENORD lines), FER/TPER, FSE, EAV,
Ferrotramviaria and FdG feeds where they exist. The Trenitalia feed alone should rescue the
dropped sections listed above (AV Treviglio - Brescia, Sardinia, Foggia, Signa) and grey any
line with no trains (Udine – Palmanova is the first to look at).
