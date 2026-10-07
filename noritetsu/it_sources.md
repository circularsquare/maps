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
python gtfs_served.py --fetch it  # refresh the timetable feeds (data/raw/gtfs/it/), then rebuild
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
- node lines: split since 2026-10-03 into the lines inside each city (below, "City nodes").

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

## Built (2026-10-03: timetable check live, city nodes split)

- **310 register lines, 16,856 km** (2026-10-02: 302 lines, 16,604 km), against RINF's own
  section lengths median 0.997 (306 lines of 2 km or more, 4 off by more than 5%: Bari –
  Barletta, where RINF gives Andria Sud - Andria and Corato - Corato Sud 13.06 km each for
  about 1.4 km of track; Cagliari – Decimomannu; Busto Arsizio – Malpensa; a 2.4 km Venezia
  yard piece). 809 km of it is now greyed as not running (below).
- rinf.py traced 17,317 km of 17,392 km of RINF length; 20 sections rejected (no track, below);
  build_model then dropped 107 junction-ended sections (461 km) that neither a train in the feed
  nor an OSM passenger route runs over (2026-10-02: 154 sections, 672 km).
- What follows in this section is from the 2026-10-02 build and still holds.
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

`python check_model.py --region it` (2026-10-03): 101 published lengths, 83 within 5%, every
miss explained in its note. The F-line figures are it.wikipedia chainage between the built
line's two ends; with the city nodes split most F lines now reach their city stations, so the
figures were re-taken (Bologna – Ancona from Bologna Centrale 0.997, Bologna – Padova 0.993,
Torino – Arquata Scrivia from Porta Nuova 1.016, Firenze – Roma (linea lenta) Firenze SMN -
Roma Termini 1.021 against it.wikipedia's 314, Bologna – Porretta Terme 0.995, Roma – Avezzano
from Termini 1.024).

| line | built | published | ratio | why |
|---|---|---|---|---|
| Milano – Mortara | 54.1 | 44.0 | 1.23 | the southern belt Rogoredo - Romolo - San Cristoforo (9.8 km, from the Milano node) |
| Alessandria – Piacenza | 110.6 | 96.5 | 1.15 | RINF also files the Bressana - Broni leg (13.3 km) |
| Venezia – Trieste | 159.8 | 141.1 | 1.13 | Bivio d'Aurisina - Villa Opicina (14.9 km, kept by the timetable) and Mestre Olimpia - Carpenedo |
| Fiumetorto – Messina | 199.0 | 180.6 | 1.10 | old coast line via Falcone and the new Patti - Terme Vigliatore line both built |
| AV Bologna – Firenze | 85.7 | 78.5 | 1.09 | built starts at Bologna's underground AV station (it.wikipedia says 86) |
| Milano – Bologna | 232.7 | 214.5 | 1.08 | RINF also files Rogoredo - Bivio Melegnano - Tavazzano (18.2 km) |
| Bologna – Firenze (Direttissima) | 104.6 | 97.0 | 1.08 | now Bologna San Vitale - Firenze SMN, with the Castello - Olmatello link |
| Nocera Inferiore – Codola | 4.3 | 4.0 | 1.07 | rounded figure |
| AV Torino – Milano | 133.3 | 125.0 | 1.07 | Novara interconnection (kept by the timetable), Rho and Stura links |
| Firenze – Roma (Direttissima) | 265.4 | 249.5 | 1.06 | Chiusi and Valdarno interconnections (15.8 km, kept by the timetable) |
| Genova – Pisa | 168.0 | 158.5 | 1.06 | both Vezzano - La Spezia routes, and Pisa Centrale |
| Verona – Bologna | 117.0 | 110.8 | 1.06 | the Verona Porta Vescovo leg (6.8 km) |
| Bari – Bitritto | 9.5 | 9.0 | 1.05 | rounded figure |
| Reggio Emilia – Guastalla | 30.5 | 29.0 | 1.05 | the Reggio San Lazzaro spur (2.1 km) |
| Domodossola – Novara | 85.1 | 89.6 | 0.95 | Vignale - Novara is C24, a line of its own |
| Bologna – Portomaggiore | 41.6 | 45.0 | 0.93 | FER's line starts at Bologna Roveri |
| Saronno – Seregno | 13.3 | 15.0 | 0.89 | RINF starts at Saronno Sud (13.15) |
| Cremona – Mantova | 36.4 | 62.0 | 0.59 | Bozzolo - Mantova has no track in the extract |

Among those within 5%: Brennero – Verona 1.01, Ancona – Foggia 0.99, Milano – Torino 1.03,
Battipaglia – Potenza – Metaponto 1.00, Lecco – Sondrio – Tirano 1.00, every FSE line
0.98-1.00.

## City nodes (split 2026-10-03; Anita's yes of 2026-10-02)

RFI files every line inside eight cities under one node id (N1 Torino ... N8 Napoli; Nodo di
Roma was 194 km). `NODE_SPLIT` in it.py hands each node section, by the uopids of its two ends,
to the it.wikipedia line it belongs to ("Rete ferroviaria del Lazio" and the line articles'
station tables, retrieved 2026-10-03). Mostly that is the F/C line that used to stop at the
node's edge, which keeps its line id and now reaches the city's stations:

- **Roma**: Firenze – Roma (linea lenta) to Roma Termini by Settebagni, Fidene, Nuovo Salario
  and Tiburtina (the FL1 pair); the Direttissima to Tiburtina; AV Roma – Napoli's Termini and
  Tiburtina links; Roma – Avezzano (was "Guidonia – Avezzano") from Termini by Prenestina and
  Lunghezza (FL2); Roma – Formia – Napoli from Termini by Casilina; Roma – Cassino by Capannelle;
  Pisa – Roma to Termini by Ostiense, Trastevere, San Pietro and Aurelia (FL5), with the old
  Ponte Galeria - Maccarese route; Roma – Viterbo from Trastevere by Quattro Venti, Monte Mario
  and La Storta (FL3). New lines: **Roma – Fiumicino** (Trastevere - Ponte Galeria - Fiumicino
  Aeroporto, 23.0 km; keeps the Nodo di Roma's id) and **Valle Aurelia – Vigna Clara** (7.1 km).
- **Milano**: new **Milano – Gallarate** (Gallarate - Rho - Certosa - Porta Garibaldi /
  Centrale, 47.7 km; keeps the Nodo di Milano's id) and **Passante di Milano** (Bovisa -
  Rogoredo, 15.2 km); Milano – Como – Chiasso (was "Seregno – ...") from Centrale and Garibaldi
  by Monza; Milano – Verona from Centrale by Lambrate and by Forlanini - Segrate; Milano –
  Bologna from Lambrate; Milano – Mortara by the southern belt from Rogoredo.
- **Bologna**: new **Bologna – Porretta Terme** (the Porrettana's Bologna half, Santa Viola -
  Porretta, 54.0 km; keeps the Nodo di Bologna's id; Pistoia – Porretta Terme stays its own
  line as RFI files it); Milano – Bologna, Verona – Bologna, Bologna – Padova, Bologna – Ancona
  and the Direttissima to Bologna Centrale or San Vitale.
- **Torino**: new **Passante di Torino** (Lingotto - Porta Susa - Rebaudengo - Stura, 19.8 km;
  keeps the Nodo di Torino's id); Torino – Modane to Porta Nuova and Porta Susa; Torino –
  Arquata Scrivia to Porta Nuova; Milano – Torino to Stura.
- **Firenze**: Firenze – Roma (linea lenta) from Rovezzano to Santa Maria Novella and Rifredi;
  the Bologna Direttissima from Castello; Firenze – Pisa from Cascine; Firenze – Borgo San
  Lorenzo by Le Cure to Campo di Marte.
- **Venezia**: Padova – Venezia to Santa Lucia; Venezia – Udine – Tarvisio to Mestre; Venezia
  Mestre – Castelfranco Veneto (was "Castelfranco Veneto – Maerne") by Spinea.
- **Genova**: Genova – Savona to Principe (and by Via di Francia to Principe sotterranea);
  Genova – Pisa from Principe by Brignole; Arquata Scrivia – Genova by the Giovi lines; Genova –
  Ovada to Torbella.
- **Napoli**: Villa Literno – Napoli Gianturco (was "Villa Literno – Pozzuoli"; the Passante,
  metro line 2); Napoli – Salerno, Roma – Formia – Napoli and Napoli – Cancello – Dugenta to
  Napoli Centrale and Gianturco.

Track no line takes (yards, freight belts, depot leads) stays under `<node>R`, named "Cintura
di Milano", "Cintura di Bologna" or "Nodo di ..."; drop_unridden_sections drops what no train
runs over. Kept as drawn track: Cintura di Milano (Lambrate - Smistamento - Rogoredo 17.9 km,
Greco - Turro - Centrale 6.7 km), Cintura di Bologna (4.1 + 5.9 km of second routes), Nodo di
Genova (Bivio Polcevera - Voltri bretella 10.7 km), Nodo di Torino (Orbassano lead 4.8 km),
Nodo di Venezia (2.4 + 1.5 km).

**Saved rides.** A ride names a line id and two stations. Roma, Milano, Bologna and Torino keep
their node id on one new line, so a ride between stations of that line still credits. Genova,
Firenze, Napoli and Venezia's node ids are gone: `line_alias` in COUNTRY (NODE_ALIAS) maps each to
the line most of its stations went to (Genova -> Genova – Savona, Firenze -> Firenze – Roma
(linea lenta), Napoli -> Villa Literno – Napoli Gianturco, Venezia's two pieces -> Padova –
Venezia and Venezia – Trieste), and rinf.py keeps it as `rinf.LINE_ALIAS`; build_model does not
ship it in aliases.json yet (a one-line change, asked for). A ride between stations that went
to two different lines cannot be moved by a one-to-one alias at all.

## Still off, and why

- **AV Treviglio – Brescia (53.8 km) is still dropped**, though every Milano - Venezia
  Frecciarossa and Italo runs over it. The timetable check calls it "ambiguous": the Frecce run
  Milano - Brescia non-stop and the classic line by Romano and Rovato is about as short, and
  Trenitalia's shapes are straight lines between stops, so nothing tells the two apart. Needs a
  high-speed rule in gtfs_served (high-speed products prefer `highspeed` sections), not in it.py.
  The other 407 km still dropped are freight curves, yard throats and AV interconnections the
  check found no train on or only long non-stop runs over ("weak": Bologna San Ruffillo - Bivio
  Emilia 8.3 km and Bivio Modena Ovest - Quattro Ville 4.4 km carry Frecce, 516 and 241 trips,
  but only as parts of runs over 40 km).
- **No track in the extract** (OSM has these as `railway=construction` with no passenger route,
  so extract.py leaves them out): Bergamo - Ponte San Pietro (doubling works), Bozzolo -
  Mantova, Decimomannu - Villamassargia - Iglesias/Carbonia (Villamassargia – Carbonia built
  as 0.6 km), Lugo - Sant'Agata. If these have reopened, OSM has not caught up.
- Four register lines took a ref from the OSM line merged into them ("R" on Novara – Biella and
  Santhià – Biella, "2" on Martina Franca – Lecce, "FUC" on Udine – Cividale): not RFI numbers.
- The Simplon tunnel's Italian half (Iselle - border, 0.4 km in RINF) has no path; the EC Basel
  - Milano owns about 25 km of tunnel and approach way the register does not cover.
- Lines have no colours of their own beyond OSM's (99 lines, mostly metros and Trenord).

## Timetable check (live 2026-10-03)

The feeds fetched on 2026-10-02 (`gtfs_served.FEEDS["it"]`) were moved from
`data/raw/gtfs_pending/it/` to `data/raw/gtfs/it/`, which switches gtfs_served on for Italy:

| feed | what is in it | window |
|---|---|---|
| `it_trenitalia.gtfs.zip`, deryclem's conversion of Trenitalia's NeTEx (CC BY 4.0) | 13,671 rail trips (Frecce, IC, regional, SFM; Trenitalia's buses left out) | 26 Sep - 12 Dec 2026 |
| `it_trenord.gtfs.zip` (Transitous) | 6,745 trips, every Trenord line incl. FERROVIENORD's and TILO | 26 Jul - 12 Dec 2026 |
| `it_eav.gtfs.zip` (Transitous) | 625 trips, EAV's railways (Cancello - Benevento, the Alifana, and the Vesuviana and Flegree lines that are not in RINF) | 30 Sep 2026 - |
| `it_ferrotramviaria.gtfs.zip` (Transitous) | 852 trips, Bari - Barletta; **stale**, its calendar ends 31 Dec 2025 | 2023 - 2025 |
| `it_gtt.gtfs.zip`, `it_tft.gtfs.zip` (Transitous, slimmed) | nothing: no rail routes survive the slimming | |

Not in any feed: Trenitalia Tper (Emilia-Romagna's regional trains, on FER's lines and on RFI
lines such as Ferrara – Ravenna – Rimini, Castel Bolognese – Ravenna, Lugo – Lavezzola and the
Porrettana), FSE, Ferrovie del Gargano, FUC, Busitalia (Perugia's FCU lines), TFT's own trains
(TFT's dati.toscana.it feed, checked 2026-10-03, has only 118 route_type 3 trips on its two
lines, so it was not added), Italo. Their lines come out "unknown" where OSM routes run over
them or their manager is missing, and stay drawn. Searched for open feeds: Transitous' Italian
index, the Mobility Database catalogue, TPER's open-data page (buses only).

2,271 of 2,548 feed stations match a register station (names; Trenitalia's stop code
`830008409` is RINF's IT08409 for 1,640 stations but not all: Fidene, Nuovo Salario and the FL3
stations are numbered off by a few, so no `CODE` rule).

**Kept** (junction-ended sections OSM routes alone dropped): 46 sections, 209 km on 24 lines.
The big ones: Udine – Palmanova (17.9 km, the whole line, 45 trips: Udine - Cervignano trains);
Giave - Chilivani and Ardara - Chilivani in Sardinia (34.9 km); Firenze – Pisa by Signa (25 km);
the Direttissima's Chiusi and Valdarno interconnections (15.8 km); Foggia's approaches on
Foggia – Bari and Dugenta – Foggia (13 km); Sarno - Bivio Santa Lucia (11.2 km); Villa Opicina -
Bivio d'Aurisina on Venezia – Trieste (14.9 km, but on only 2 trips in the window).

**Closed** (no train in the window; greyed): 101 sections, 809 km on 21 lines. Checked one by
one against the feed's own buses and the web:
- works closures with buses in the feed: Sibari – Catanzaro Lido's Sibari - Crotone part
  (112.5 km; electrification, trains suspended 13 Sep - 12 Dec 2026), Termoli – Campobasso
  (87.0; electrification), Ponte nelle Alpi – Calalzo (35.5; buses to Calalzo in the feed),
  Melfi - Rocchetta on Rocchetta Sant'Antonio – Potenza (16.1; trains run Potenza - Melfi,
  buses Melfi - Foggia), Mercato San Severino – Montoro (4.4; buses)
- suspended for years, buses in the feed: Barletta – Spinazzola (65.5), Gravina – Gioia del
  Colle (46.5), Cecina – Volterra (29.4), Oleggio – Laveno Mombello (33.9), Fabriano – Pergola
  (28.8), Novara – Romagnano Sesia (25.6), Bra - Cavallermaggiore (12.8)
- tourist trains only (Fondazione FS): Sulmona – Carpinone (114.8, the Transiberiana d'Italia),
  Asciano – Monte Antico (51.0), Agrigento – Porto Empedocle (10.1)
- freight: Gemona – Osoppo (4.8)
- a second filing of running track: Palermo Notarbartolo - Imperatore Federico (2.1; trains run
  the parallel Libertà sections), Carbonia Serbariu - Carbonia Stato (0.6)
- **wrong**, waiting on two gtfs_served changes (asked for; with both patched in at run time
  only these three lines change, to 681 km closed on 18 lines):
  - San Severo – Peschici (73.7; Ferrovie del Gargano reopened it on 1 June 2026) and Zollino –
    Gagliano del Capo (46.3; FSE, state unknown): their managers count as "in the feed" because
    "Ferrovie" and "del" match Ferrotramviaria's agency name, "Ferrovie del Nord Barese"
    (`ORG_WORDS` needs the Italian generic words)
  - Valle Aurelia – Vigna Clara (7.1): about 20 trains a day Monday to Saturday, but the feed
    places "Vigna Clara PES" 5.5 km away (41.9676, 12.4094), so it matches nothing

**Still dropped**: AV Treviglio – Brescia (above, "Still off").
