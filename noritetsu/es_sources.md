# Spain register sources (built 2026-10-02)

What the Spanish build reads, where each piece came from, and what is still wrong with it.
Spain is built with `rinf.py`; how the reader works is in its docstring, and the per-country
entry is `rinf_countries/es.py`, whose docstring explains each fix. Downloads live in
`data/raw/rinf/es/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch es                                   # RINF + Wikidata, ~1 min
curl -L -o data/raw/es/spain-latest.osm.pbf https://download.geofabrik.de/europe/spain-latest.osm.pbf
curl -L -o data/raw/ic/canary-islands-latest.osm.pbf https://download.geofabrik.de/africa/canary-islands-latest.osm.pbf
python tools/slot.py -- python extract.py --region ic --pbf data/raw/ic/canary-islands-latest.osm.pbf   # ~7 s, 57 MB
python tools/slot.py 2 -- python extract.py --region es --pbf data/raw/es/spain-latest.osm.pbf   # ~2 min, 1.5 GB
python -m rinf_countries.es --canaries   # fold data/proc/ic into data/proc/es; after EVERY es extract
# delete both .pbf files after; data/proc/ic too once the build is checked (re-extract it next time)
python inspect_region.py --region es
python build_model.py --region es --register rinf:data/raw/rinf/es   # ~80 s
python build_tiles.py --region es                                    # ~40 s, es.pmtiles 4.7 MB
python check_model.py --region es
python rinf.py --dry es          # the reader alone, with its full log
python gtfs_served.py --fetch es # the timetable feeds into data/raw/gtfs/es/ (then delete es_feve.gtfs.zip
                                 # unless FEEDS has dropped it; see "Timetable check")
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
  relations, 122 of them with a three-digit Adif number as `ref`. Re-extracted 2026-10-08
  (45,614 track ways, 44,403 stops, 785 relations). Geofabrik files the **Canary
  Islands** under Africa (`africa/canary-islands-latest.osm.pbf`, 2026-10-08), so Tenerife's
  tram comes from that second extract, folded in (see "The Canary Islands" below). **Andorra**
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

Since the timetable check (2026-10-03): **171 register lines, 14,660 km**, of which 252 km
(27 sections, 11 lines) are drawn as not running; 2,755 stations, 43,126 route-km. One
station id went with nothing to carry it to: Riquelme - Sucina (no longer a stop). The rest
of this section is the build before it.

Kept on 2026-10-02: **166 register lines, 14,366 km** (Adif 131 lines 10,561 km, Adif AV 33 lines 3,697 km,
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
- **In service, but no OSM route over it**: 984, the Pajares base-tunnel line (OSM's Asturias
  Alvia/AVE relations still run over the old ramp) and 320 Chinchilla - Hellín. Both are kept
  since 2026-10-03 by the timetable check (below).

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
track, 726 1.05 for 2.4). Against the catalogue, 53 lines, 50 within 5% and 41 within 2% (2026-10-03: 56 lines with 422,
500 and 984 added, all but 222 and 036 within 5%; 320 now 1.00):

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
- **320** was 0.65 before the timetable check kept Chinchilla - Hellín (2026-10-03); about 1.00
  now.
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

- **120 west of Salamanca and 822 Ourense - A Gudiña** are drawn as running though no
  passenger train runs (timetable section, "Left drawn").
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

## Timetable check (live since 2026-10-03)

Later on 2026-10-03 `gtfs_served.FEED_COMPLETE = {"es"}` landed (Spain's feeds carry every
operator on Adif track, so an OSM route over a section no train runs is stale): 120 Villar
Formoso – Medina del Campo closes 90.8 km west of Salamanca, and 822 closes 138.9 km, Ourense –
A Gudiña plus Puebla de Sanabria – A Gudiña (the Valladolid regional ends at Puebla de
Sanabria). Spain's closed total: 31 sections, 482 km on 13 lines (was 252 km on 11).

`gtfs_served.py` reads `data/raw/gtfs/es/` (fetched 2026-10-02; landed from
`data/raw/gtfs_pending/es/` on 2026-10-03, the parked folder deleted):

| file | what is in it | window |
|---|---|---|
| `es_renfe_av_ld.gtfs.zip` (Renfe, CC BY 4.0) | AVE, AVE INT, Avlo, Alvia, Euromed, Intercity, Avant, MD, Regional, Regional Exprés, Proximidad, the Celta, and the four metre-gauge regionals (Ferrol - Oviedo, Oviedo - Santander, Santander - Bilbao, Bilbao - León) | 1 Oct 2026 - 24 Jan 2027 |
| `es_renfe_cercanias.gtfs.zip` (Renfe) | every Cercanías/Rodalies network, metre gauge included (Asturias C4-C8, Cantabria C2/C3, Bilbao C4/C5, Ferrol, León, Cartagena), Rodalies' regional R lines (R11-R17, RG1, RL3, RL4, RT1, RT2); replacement buses as route_type 3 | 24 Sep - 23 Oct 2026 |
| `es_fgc.gtfs.zip` (Transitous) | FGC, slimmed to rail: Lleida - La Pobla (RL1, RL2) is the only FGC line in the register | 15 Sep - 31 Dec 2026 |
| `es_ouigo.gtfs.zip` (Transitous) | Ouigo España | 26 Jun - 12 Dec 2026 |

Left out: Transitous' `es_Feve` (FEED entry `es_feve.gtfs.zip`). It is the Cercanías
feed's León, Ferrol and Cartagena networks again, dated 13 April - 13 May 2026, and it types
the León FEVE - La Asunción bus as a train (route_type 2). FEEDS still lists it; the managing
session is asked to drop it, or `--fetch es` brings it back.

Not in any feed, and why it does not matter here: Iryo (runs only on high-speed lines Renfe
and Ouigo also run), Euskotren, FGV, SFM, Tren de Sóller, metros and trams (none of their
track is in the register). Renfe's stop_id is Adif's station code, RINF's uopid less "ES"
(`CODE["es"]`); Ouigo's 9-digit ids and FGC's letters match by name. 2,104 of 2,252 feed
stations matched.

**A trap in Renfe's long-distance feed**: road legs can be typed as trains. Its "Intercity"
Murcia - Lorca - Águilas and València - Murcia - Totana - Lorca trips are route_type 2 and run
daily, but the line beyond Murcia has had no train for five years (the Murcia - Almería
high-speed works; the Cercanías feed has the same C2 trips as buses). It does no harm here
only because the register has no traceable Murcia - Lorca track.

**Branch lines meeting nothing** (`CUT_AT` in es.py, rinf.py `cut_at_junctions` taking a set
of point names since 2026-10-03). The reader merges a junction point away inside a line when
it has two neighbours on that line, so a branch leaving there ends at a node no other section
touches, and the check finds no path onto it: 984 had 2 trips, Chinchilla - Hellín none.
Cutting at every junction (as Portugal does) was tried: it rescued those, but split main lines
into junction-ended pieces that a parallel line made "ambiguous" or long non-stop runs "weak",
and dropped real track (Valladolid - Venta de Baños on 080, 38 km; Albacete - Chinchilla on
300, 19 km; pieces of 084, 822, 400). So es.py cuts at eight named points only: Chinchilla
aguja km 298,4, Bif. Pajares, Bif. Utrera, Bif. Casa de la Torre, El Reguerón aguja km 522,1,
Bif. Angueira, Bif. San Amaro, Bif. Teruel. And Riquelme-Sucina is no stop (no train calls;
as a stop it left El Reguerón - Riquelme a piece crossed only by the 41.5 km Murcia - Balsicas
run, "weak" by 1.5 km).

**Kept that OSM routes alone dropped** (20 sections, 419 km before ownership):
- 984 Pola de Lena - Bif. Pajares, the Pajares base tunnel (49.0 km, 75 trips)
- 320 Chinchilla - Hellín (50.4 km, 10 trips; 320 is now 146.0 km against the catalogue's
  146.2)
- 500 Cañaveral - Bif. Casa de la Torre (39.3 km, Madrid - Cáceres regionals) and 026
  Plasencia - Bif. Casa de la Torre (70.6 km, the Extremadura high-speed line; it was in the
  build before as one Cáceres - Plasencia section)
- 422 Arahal - Bif. Utrera (27.0 km, Sevilla - Málaga/Osuna MD)
- 352 Balsicas - El Reguerón (32.7 km, of which 22.6 new: Murcia - Cartagena)
- 300 Albacete - Chinchilla aguja - Almansa (two pieces now, both served)
- 640 Camp de Tarragona - Cambiador de La Boella (12.0), 444 Sevilla Santa Justa - La Salud
  (7.7), 818 Padrón - Bif. Angueira (6.8), 702 Cabañas de Ebro - Grisén (6.0), 402
  Mengíbar-Artichuela - Espeluy (4.2), 354 Murcia - El Reguerón (3.8), 828 A Portela - Bif.
  San Amaro (3.8), 130 La Robla - Bif. Pajares and León approaches, 132 Olloniego (1.6), 270
  Portbou - border (1.1)

**Closed, no train in the feed** (27 sections, 252 km, 11 lines), each checked:
- 430 Córdoba - Montilla - Puente Genil - La Roda - Fuente de Piedra (112.6 km): no passenger
  train since the Córdoba - Málaga high-speed line opened; Puente Genil is served only at
  Puente Genil-Herrera (cordopolis, puentegenilok.es, 2024-2026). 50.2 km of it (Montilla -
  Córdoba) was dropped before and is now drawn grey.
- 310 Utiel - Requena - Siete Aguas - Buñol (45.4 km): works after the October 2024 DANA;
  Buñol reopened 22 Dec 2025, Buñol - Utiel by bus until the end of 2026 (eldiario.es,
  utiel.es). Refetch when it reopens.
- 764 Trubia - Fuso de la Reina - Soto de Ribera - Peñamiel (17.1 km): no passengers since
  4 May 2009 (C-8 runs Baiña - Collanzo only).
- 154 Lugo de Llanera - Tudela-Veguín (13.8 km): freight bypass of Oviedo.
- 322 Águilas - Jaravía (11.9 km): Murcia - Lorca - Águilas closed for the Murcia - Almería
  high-speed works, buses (C2) in the feed and in the news (see the trap above).
- 792 La Robla - Matallana (10.9 km, metre gauge): freight branch; Bilbao - León runs via
  Matallana to León.
- 754 Sotiello - Aboño - El Musel (9.4 km): port freight.
- 222 Granollers-Canovelles - Parets del Vallès (9.4 km): R3 doubling works, Montcada - La
  Garriga closed 7 Oct 2025 to January 2027, buses in the feed (3cat). Refetch then.
- 782 Ariz - Basurto Hospital (8.1 km, metre gauge): closed, greyed in the catalogue.
- 116 Puerto de Navacerrada - Los Cotos (7.1 km): C9 Cercedilla - Los Cotos runs as buses.
- 260 Figueres-Vilafant - Vilamalla (6.5 km): freight branch.

**Left drawn though no train runs** ("unknown": OSM routes still run over it, and the check
reads that as an operator missing from the feed). Both are closed to passengers:
- 120 La Alamedilla - La Fuente de San Esteban - Ciudad Rodrigo (90.8 km) and Ciudad
  Rodrigo - Fuentes de Oñoro border ("border", 32.2 km): no passenger train since March 2020
  (salamancahoy.es 2025-2026; a return is only being discussed).
- 822 Ourense - A Gudiña (89.1 km) and Puebla de Sanabria - A Gudiña ("ambiguous", 49.8 km):
  the conventional line beside the high-speed one; the Valladolid regional ends at Puebla de
  Sanabria.
Closing them needs gtfs_served to stop reading OSM routes as a missing operator for Spain,
whose feeds carry every operator on Adif track (a diff is in the 2026-10-03 report).

Still dropped as "weak" (on long non-stop runs only, no OSM route): gauge-changer links at
Antequera, Valdestillas, Plasencia de Jalón, Alcolea (a few km), 024 Yeles - Los Blancales (5.6
km), 460 Fuente de Piedra - Bif. Las Maravillas (11.7), 610 Cuarte de Huerva - Bif. Teruel (3.5;
cutting at Bif. Teruel did not help, the Zaragoza junctions are a tangle), 536 (2.6), 302
Alcázar curve (1.9), 512 Gibraleón - Huelva-Mercancías (14.1, rightly: only a nonsense 739 km
path crosses it), Madrid's Santa Catalina freight links.

## The Canary Islands (built 2026-10-08)

Tenerife's tram is in `es` (Anita, 2026-10-08: "sure we can fold it into spain"); research in
`canaries_survey.md`. It is the islands' only railway. Geofabrik ships the islands as their own
extract, so they are extracted as region `ic` and folded into `data/proc/es` by `python -m
rinf_countries.es --canaries` (`canaries()` in es.py): the two extracts' ways, relations, stops
and coordinates joined by OSM id, nothing shared with the mainland file. extract.py is
unchanged. No register, no rules: two OSM tram lines, operator MetroTenerife.

| line | built | stops | published (en.WP) |
|---|---|---|---|
| m20282579 Tranvía Línea 1 (L1) Intercambiador - La Trinidad | 12.4 km | 21 | 12.5 km, 21 |
| m16267950 Tranvía Línea 2 (L2) La Cuesta - Tíncer | 3.4 km | 6 | 3.6 km, 6 |

Hospital Universitario and El Cardonal are one station each, on both lines (25 stations).
L2's route_master (16267950) has no tags in OSM but a wikidata id, so `CANARY_MASTERS` gives it
the name and ref in the form L1's master has ("Tranvía Línea 2", L2; Metrotenerife's feed calls
the lines "Linea 1"/"Linea 2", L1/L2), and its routes' operator and network. No colours: the
feed has none and OSM's "blue" is on L2's routes only.

Checked: Spain's outline (religiondots' country shape) holds the islands (the tram's stops lie
0.4-7 km inside it), and regions.json's es bbox reaches -18.2 W. The timetable check judges
register sections only, so the tram, with none, is never greyed for missing from Renfe's feeds.

## Rebuild of 2026-10-08 (fresh extract + the Canaries)

422 lines, 2,773 stations, 43,087 route-km; 168 register lines (unchanged). `compare_lines.py
diff es`: 8 register lines differ, all by 0.01-0.03 km. ab.py of the fresh mainland extract
alone against the 2026-10-02 one, then with the islands folded in: the islands add exactly the
two tram lines and their 25 stations. What the newer OSM moved:
- Station renames in OSM (ids change, sections the same): València Cabanyal, Lantueno, Curuxona,
  Montiana, Ḷḷinares-Congostinas, Candás Apeaderu, Veiga d'Anzu and others; OSM re-mapped
  several Galician and Asturian metre-gauge halts as new nodes.
- **Las Mazas/Les Maces** became "Les Maces", which no spelling of RINF's "LAS MAZAS" reaches,
  so 760 Oviedo - Trubia lost the stop: `STOP_NAMES["LAS MAZAS"] = "Les Maces"` puts it back.
- Halts OSM no longer has: Bolunburu (C4/R4 and 790, now La Herrera - Ibarra in one section) and
  El Turujal (Regional Oviedo - Santander, Cabezón de la Sal - Treceño).
- 204 Canfranc gains Villanúa-Letranz as a stop (now mapped as a station); the Alvia Ferrol -
  Madrid relation no longer calls at A Gudiña (named train, -18.6 km); Madrid Metro 10 +0.3 km,
  11 -0.04 km; Euskotren's Larreineta funicular took an operator.
