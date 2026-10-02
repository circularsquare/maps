# Spain register sources (built 2026-10-02)

What the Spanish build reads, where each piece came from, and what is still wrong with it.
Spain is built with `rinf.py`; how the reader works is in its docstring, and the per-country
entry is `rinf_countries/es.py`, whose docstring explains each fix. Downloads live in
`data/raw/rinf/es/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch es                                   # RINF + Wikidata, ~1 min
curl -L -o data/raw/es/spain-latest.osm.pbf https://download.geofabrik.de/europe/spain-latest.osm.pbf
$env:OSMIUM_POOL_THREADS=2; python extract.py --region es --pbf data/raw/es/spain-latest.osm.pbf   # ~4 min, 1.5 GB; delete the .pbf after
python inspect_region.py --region es
python build_model.py --region es --register rinf:data/raw/rinf/es   # ~80 s
python build_tiles.py --region es                                    # ~40 s, es.pmtiles 4.7 MB
python check_model.py --region es
python rinf.py --dry es          # the reader alone, with its full log
```

`inspect_region.py` crashed at its very end on this extract (a route kind of `None` in the
route-kinds table, line 118 of the script); everything it prints before that is complete.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-02: 2,520 sections, one version each (no validity periods), 2,338 point rows; 466
  line ids, 15,537 km, all under one manager code, Adif's `0071_IM` (Adif and Adif AV are not
  told apart). All three gauges are in it: Iberian broad gauge, the standard-gauge high-speed
  lines, and the metre gauge that was FEVE's (lines 116, 360, 740-794). Not in it: FGC (except
  Lleida - La Pobla, see below), Euskotren, FGV, SFM, Tren de Sóller, every metro and tram.
- **OpenStreetMap**, Geofabrik `spain-latest.osm.pbf` downloaded 2026-10-02, ODbL: 45,594
  track ways, 44,286 stops, 785 route relations (396 `route=train`), 224 `route=railway`
  relations, 122 of them with a three-digit Adif number as `ref`. Geofabrik files the **Canary
  Islands** under Africa (`africa/canary-islands-latest.osm.pbf`), so Tenerife's tram (lines 1
  and 2) is not in this build; adding it needs that second extract merged into `es`. **Andorra**
  has no railway. Ceuta and Melilla have none. The Balearics are in (SFM T1-T3, Palma metro M1
  and M2, Tren de Sóller, Tranvía de Sóller).
- **The line catalogue**: es.wikipedia "Anexo:Líneas de la Red Ferroviaria de Interés General"
  (https://es.wikipedia.org/wiki/Anexo:L%C3%ADneas_de_la_Red_Ferroviaria_de_Inter%C3%A9s_General,
  raw wikitext, retrieved 2026-10-02, CC BY-SA 4.0): every Adif line by number, its two ends,
  length, gauge and administrator, 281 rows. It cites Orden FOM/710/2015, the ministry's
  catalogue of lines (https://www.boe.es/boe/dias/2015/04/23/pdfs/BOE-A-2015-4382.pdf), and
  Adif's Declaración sobre la Red (https://www.adif.es/sobre-adif/declaracion-red; the 2026
  book is https://www.adif.es/documents/20124/51645141/20260327_01_DR+ADIF_V1_LIBRO_2026.pdf,
  whose Annex F is the same catalogue; not read page by page). Names, the Adif/Adif AV split
  and the check lengths come from it. Three numbers RINF uses are not in the annex (128, 630,
  924); their names come from RINF's own end points.
- **Wikidata** (`wikidata.json`), CC0: 357 rows, items with P1671 and P17 Spain, Spanish labels
  only. They use Adif's numbers (100 "Línea Madrid-Hendaya") but are used only for the "built
  as no line" list in the log: the catalogue names every line.

## How `es.py` reads the ids

Every RINF id is "ESL" + Adif's three-digit line number + six digits of an older section code:
ESL100100010 is part of line 100 (Hendaya - Madrid), ESL050210000 of 050 (the Madrid -
Barcelona - French border high-speed line; standard-gauge lines are 0xx), ESL740874000 of 740
(Ferrol - Pravia, metre gauge). So the number is characters 4-6, and it is Adif's number
(`rule_certain`): 250 of 487 ids are confirmed by an OSM relation of that number, 157 lie near
no numbered relation, 78 rest on the rule.

Names are the catalogue's: "100 Hendaya – Madrid-Chamartín-Clara Campoamor", English "Line 100
(Hendaya – Madrid-Chamartín-Clara Campoamor)". The annex writes Castilian exonyms; they are put
back to the names Adif's stations carry (Lleida-Pirineus, Girona, A Coruña, Xàtiva, Alcoi,
Sant Vicenç de Calders, L'Hospitalet de Llobregat, València-Estació del Nord, Alacant-Terminal).
The operator is "Adif AV" for the 33 lines the annex says Adif AV administers whole (the
high-speed lines, the Mediterranean corridor 600, the Atlantic axis 824), "Adif" otherwise.

- **Lleida - La Pobla de Segur**: Adif kept its first 1.9 km (line 206); the other 87.9 km were
  handed to the Generalitat and FGC runs the line, but RINF still lists that part, under a
  catch-all id ESL893002600. It is joined to 206 (`fixed`) and shown as one line, operator FGC.
- **Left out** (`skip_line`): the other ids numbered 870-899, port, factory and yard sidings
  (Puerto de Barcelona, Mercabarna, Cepsa, Ensidesa, Fasa-Renault, Cádiz-Puerto, the Pedralba and
  Taboadela gauge changers), none in the catalogue.
- **The LFP line**: the Spanish 20 km of the Perpignan - Figueres high-speed line belong to LFP
  Perthus, not Adif, and RINF stops at "LÍMITE ADIF - LFPSA". It is added as its own line,
  "Figueres – Perpignan (LFP)", no number, operator LFP Perthus, to the French border point.

## What RINF gets wrong, and what es.py does about it

Spain's RINF is less complete than the other countries' so far. All of this is in `es_fix`,
and every item is logged by the build.

- **Stretches missing outright** (checked against the ERA endpoint for any country: no section
  has those points as an end): 21 gaps, 526 km, added as sections between the RINF points
  either side. The big ones: 982 Olmedo - Zamora - Ourense (217 km in three pieces), 050
  Barcelona-Sants - Riells through La Sagrera (58) and Alcover AV - Camp de Tarragona (12), 084
  Palencia - León (79), 080 Valladolid - Venta de Baños (22), 300 Vallada - La Encina (30) and
  Silla - Benifaió (9), 026 Peñas Blancas - Bif. La Isla (21), 120 Tejares - Barbadillo (17),
  plus seven short ones. Lengths: the catalogue's shortfall where it is one gap, else
  crow-fly x 1.06 on high-speed lines and about 1.1 on others. That length only has to be within
  rinf.py's tolerance (15% + 0.3 km); the built length is the traced one.
- **Stations missing**: Sanabria AV and A Gudiña-Porta de Galicia on 982 are added as stops
  (OSM's coordinates). Without them Zamora - Ourense is one 213 km section ending at the
  Taboadela junction, and because only one track of the high-speed pair carries OSM's Alvia
  relations, build_model judged it 43% ridden and dropped all of it. The three Taboadela
  gauge-changer links are left out for the same reason (they cut the line at junctions).
- **Border links missing**: line 508 Badajoz - border towards Elvas (5.3 km in the catalogue; not
  in RINF at all) and the LFP line, both to the border points Portugal's and France's RINF
  carry (EU00125, EU00121).
- **Lengths**: Montijo - Bif. San Nicolás 333.00 km for 21 km crow-fly (set to 22.3); Almassora -
  Castelló 0.21 km for 4.3 km (4.4); line 320 sums to 163.1 km against the catalogue's 146.2,
  all of the excess in Hellín - Cieza (62.45 in RINF, 45.7 traced), taken off KM 378.0 - Cieza.
  Another 89 sections disagree with RINF's length but lie on their own line's track and are
  kept (the build log's "length off" list); most are a length split unevenly between two
  neighbouring pieces.
- **Coordinates**: P.B. El Villar (line 300) is 39 km from where its sections put it; its
  coordinate is dropped so Chinchilla - Alpera is traced end to end. Vilavella AV, Miamán and
  Bif. Pedralba lie nearer the conventional Zamora - A Coruña line than the high-speed line
  beside it (Vilavella 50 m against 224 m); they are moved onto the high-speed track.
- **Wrong stations** (`stop_name`): comparing the gauge of the track at RINF's coordinate with the
  track at the OSM station each point matched found six: Ciaño and La Felguera (Iberian line 140)
  matched FEVE's stations of that name, Zorrotza (FEVE) matched Renfe's, the high-speed
  technical points "L'Espluga de Francolí A.V." and "Campomanes AV" matched the towns'
  conventional stations, and La Sagrera (the unopened through station) matched an OSM "La
  Sagrera" 690 m away on the Meridiana tunnel, which broke the R1/R2 trunk. Ciaño and Zorrotza
  take their exact OSM names ("Ciañu", "Zorrotza Zorrotzgoiti"), the other four stay section
  ends. Portbou is typed 40 (technical) in RINF and is made a stop. Widening the 1 km name
  radius was tried and rejected: past 1 km the same-name matches are nearly all wrong (ports,
  yards, "A.V." points against the town station).
- **Tracing** (`osm_rel`): OSM's relations "LAV Olmedo-Zamora-Galicia", "LAV Variante de
  Pajares" and "Liña Zamora-A Coruña" carry no ref; they are read as 982, 984 and 822 so the
  second trace pass prefers their ways. Before that, 108 km of the 982 gaps traced over the
  conventional line beside the high-speed one.

## What is in the register and what stays OSM

Kept: **166 register lines, 14,366 km** (Adif 131 lines 10,561 km, Adif AV 33 lines 3,697 km,
FGC 1 line 89 km, LFP 1 line 20 km). The reader built 248 lines (14,987 km); build_model then
dropped 107 junction-ended sections (621 km) that no OSM passenger route runs over, which
emptied 82 lines, nearly all freight links, yards and closed lines.

Left out as unridden or untraceable, which is right as far as OSM knows:
- **Closed or under works in OSM**: R3 Montcada - La Garriga (track is `railway=construction`,
  doubling works; OSM's R3 runs La Garriga - Puigcerdà), Murcia - Lorca - Águilas (322;
  construction/disused for the Murcia - Almería high-speed works), Huércal-Viator - Almería
  (410), Madrid - Aranda beyond Colmenar Viejo (102, no trains since 2011), Cercedilla - Los
  Cotos (116), Aranjuez - Tarancón and Tarancón - Utiel (310), Córdoba - Almorchón (528),
  Valencia de Alcántara - border (502), Toral de los Vados - Villafranca del Bierzo (802), the
  old Bobadilla - Granada line (466).
- **In service, but no OSM route over it** (these are what a timetable check would rescue):
  **984, the Pajares base-tunnel line** (La Robla - Pola de Lena, 49 km, open since 2023): it
  traces correctly (49.0 km) but OSM's Asturias Alvia/AVE relations still run over the old
  Pajares ramp, so 98% of it is under no route; and **320 Chinchilla - Hellín** (51 km;
  Hellín - Cieza - Murcia stays, being stop to stop).

Stay OSM lines (256): Renfe's services over the register lines (Cercanías in Madrid,
Barcelona/Rodalies, València, Sevilla, Málaga, Murcia/Alicante, Bilbao, Asturias, Cantabria,
Cádiz, Zaragoza; Media Distancia, Regional, Avant), and every railway outside RINF: FGC
(Barcelona-Vallès, Llobregat-Anoia, Montserrat and Núria racks), Euskotren, FGV (Metrovalencia,
TRAM d'Alacant), SFM, Tren de Sóller, the metros of Madrid, Barcelona, Bilbao, València,
Sevilla, Málaga, Granada, Palma, the light rail and tram systems (Madrid's Metro Ligero, Parla,
Zaragoza, Murcia, Vitoria, Bilbao, Barcelona's Trambaix/Trambesòs, Alicante), and the
funiculars.

**Named trains** (`build_model.looks_like_service`, the `es` branch, landed 2026-10-02): Renfe's
long-distance products and the open-access operators are mapped one relation per train or
pair, and are named trains: AVE, AV City, Alvia, Avlo, Euromed, Intercity, Iryo, Ouigo,
Trenhotel, Talgo/TLG, Renfe-SNCF, Intercités, the Celta ("Train IN"), plus the networks Renfe
AVE, Iryo, Ouigo España, Renfe InterCity and TGV Europe. 80 of 527 train relations; 20 lines
after grouping by route_master. Avant, Media Distancia and Regional stay lines: they run as
regular services a rider uses as a line. The network tag "Renfe Alvia" is not used (it is also
on a Bilbao Cercanías C-3 relation), nor bare "TGV" (on a liO TER relation).

No `colours/es.csv`: Adif publishes no line colours; 171 lines (41%) carry OSM's colour.

## Counts (2026-10-02)

- 166 register lines, 14,366 km; 418 lines in all, 2,742 stations (1,597 on a register line),
  42,832 route-km (32,439 without named trains).
- 2,018 RINF passenger-typed points: 1,532 are an OSM station (1,500 distinct), 115 by distance
  alone (spellings: "LLOVIO" and "Lloviu", "S.MARIA DE GRADO" and "Santa María Grau"); 482 have
  none and become junctions, nearly all freight points, technical "A.V." points on the
  high-speed lines, closed halts, or halts OSM names differently.
- Ownership: 23,533 ways owned by a register line; 78 places (495 km of way length, both tracks
  counted) settled by the fixed rule, nearly all FGC, FGV, Euskotren, SFM, metro and tram
  lines sharing track with each other; 70 km of likely register gaps in 23 places; 39 km of
  track only named trains run over (14.7 km near Portbou under an old Renfe-SNCF relation, 9.2
  km of Iryo track in south Madrid).

## Check

`python check_model.py --region es`: against RINF's own section lengths, 141 lines of 2 km or
more, median 0.999, 30 off by more than 5%, all short links and junction curves where RINF's
length is itself off (162 Solvay - Sierrapando is 0.00 km in RINF, 314 0.14 km for 2.8 km of
track, 726 1.05 for 2.4). Against the catalogue, 53 lines, 50 within 5% and 41 within 2%:

| line | built | catalogue | ratio |
|---|---|---|---|
| 050 Límite ADIF-LFPSA – Madrid-Puerta de Atocha | 750.4 | 752.4 | 1.00 |
| 200 Madrid-Chamartín – Barcelona-Estació de França | 697.1 | 699.7 | 1.00 |
| 100 Hendaya – Madrid-Chamartín | 636.2 | 640.9 | 0.99 |
| 400 Alcázar de San Juan – Cádiz | 574.0 | 576.9 | 1.00 |
| 300 Madrid-Chamartín – València-Estació del Nord | 494.3 | 480.6 | 1.03 |
| 010 Madrid-Puerta de Atocha – Sevilla-Santa Justa | 468.9 | 470.5 | 1.00 |
| 822 Bifurcación Valorio – A Coruña | 435.4 | 436.3 | 1.00 |
| 040 Madrid-Chamartín – Valencia-Joaquín Sorolla | 396.9 | 397.6 | 1.00 |
| 982 Taboadela aguja km 234,0 – Bifurcación Medina | 327.8 | 313.9 | 1.04 |
| 080 Burgos-Rosa Manzano – Madrid-Chamartín | 302.6 | 304.0 | 1.00 |
| 740 Pravia – Ferrol | 267.9 | 269.0 | 1.00 |
| 222 La Tor de Querol-Enveitg – Bifurcació Aigües | 127.5 | 149.7 | 0.85 |
| 320 Chinchilla – Murcia del Carmen | 95.6 | 146.2 | 0.65 |
| 036 Antequera-Santa Ana – Granada | 114.0 | 125.7 | 0.91 |

(the other 39 are in the check output, all 0.97-1.01.)

- **222** (0.85): Montcada - La Garriga is under doubling works and `railway=construction` in
  OSM; OSM's R3 runs only La Garriga - Puigcerdà.
- **320** (0.65): Chinchilla - Hellín has no OSM passenger route and ends at a junction, so it is
  dropped as unridden.
- **036** (0.91): RINF's own line is 114.6 km; the catalogue figure covers more than RINF does.
- **982** (1.04): the build runs on over the mixed-gauge Taboadela - Ourense stretch RINF files
  under 982 (15.4 km, shared with 822); less that, 312.5 against 313.9.
- **300** (1.03): RINF's own is 492.0; the catalogue's 480.6 is shorter than RINF's sections.
- Not in `REGISTER`, because the catalogue figure covers a different extent or looks wrong: 700
  (233.8, but Bilbao - Castejón - Casetas is 322 in RINF and 326 built), 750/752/762 (the annex
  gives round 29 km for lines RINF has at 49-50, 49, 39), 102, 310, 322, 528 (closed parts).

## Borders

Seven border points (border_points.json), all named neutrally. Joined by a register line: Tui
(814), Fuentes de Oñoro (120), Badajoz (508, added), Puigcerdà (222), and the LFP line. At
**Portbou** the French TER and Intercités relations reach the border point (1.2 km tails); line
270's own Portbou - border piece is dropped, since no Iberian-gauge route crosses in OSM (R11
ends at Portbou). At **Irun/Hendaye** nothing reaches the point: no passenger train crosses on
Adif track (Renfe stops at Irun, SNCF at Hendaye). The only passenger crossing there is
Euskotren's metre-gauge E2 (Irun Ficoba - Hendaia), which RINF has no point for; the build logs
it ("no border point: m10025161 from Irun Ficoba") and it needs a `borders.EXTRA` point to be
drawn across.

## Still off, and why

- **984 Pajares base-tunnel line** missing (49 km, in service), and **320 Chinchilla - Hellín**
  (51 km): no OSM route over them. A timetable check would put them back.
- **520 Ciudad Real - Badajoz** lies on the 010 high-speed rails for 33.7 km near Puertollano:
  the conventional track runs about 10 m beside the high-speed one, both stations are shared,
  and no OSM relation exists to make the trace prefer the conventional ways. That stretch counts
  for 010, so riding a Ciudad Real - Badajoz train credits 010 there.
- **Gap lengths are estimates**, accepted only where the trace lands within tolerance; the built
  length is OSM's.
- **Line 300 Albacete - Almansa** traces at 91 km against RINF's 80 (P.B. El Villar's coordinate
  is wrong, and the stretch is traced end to end).
- **Stations RINF lists but OSM names differently** become junctions (482 of 2,018); the ones
  with passengers are few (Cocentaina, Lebrija, Limpias, Bellaterra, Villena AV) and the
  sections around them are kept wherever an OSM route runs.
- **Canary Islands** (Tenerife tram) not in the extract.

## Timetable check, for later

Not wired (gtfs_served.py is shared). The feeds to use, per `gtfs_sources.md`:
- Renfe long and medium distance: `https://ssl.renfe.com/gtransit/Fichero_AV_LD/google_transit.zip`
- Renfe Cercanías: `https://ssl.renfe.com/ftransit/Fichero_CER_FOMENTO/fomento_transit.zip`
- Renfe FEVE (metre gauge) via data.renfe.com; FGC, Euskotren, FGV, SFM and Ouigo are separate
  feeds. All CC BY 4.0 for Renfe.
- Renfe's `stop_id` is Adif's 5-digit station code, which is RINF's uopid without "ES"
  (ES17000 Madrid-Chamartín = stop 17000, ES60000 Madrid-Puerta de Atocha = 60000), so stops
  join by code, as in Czechia. It would rescue 984 and 320 and grey stop-to-stop sections with
  no trains.
