# France: sources and how to run it

Written 2026-09-30. The reader is `fr_register.py`; its docstring explains how it works. It was
developed and checked on the Ile-de-France extract. The whole-country build is still to run
(commands at the end).

## Register files (data/raw/fr, fetched by `python fr_register.py --fetch`)

All open, no login. SNCF Réseau data is ODbL (https://data.sncf.com/pages/licence); Wikidata is
CC0. The export URL pattern is
`https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/<id>/exports/geojson`.

| file | dataset | what the reader takes from it |
|---|---|---|
| `lignes-par-statut.geojson` | "Lignes par statut", updated 2026-02-26, 1,638 portions | geometry, PK at each end, status; only `Exploitée` portions are read |
| `lignes-par-type.geojson` | "Lignes par type", 2026-02-19 | Ligne / Rac / Vmère / Vport; only `Ligne` becomes a register line |
| `lignes-lgv-et-par-ecartement.geojson` | "Lignes LGV et par écartement", 2026-02-26 | LGV (highspeed) and narrow gauge |
| `liste-des-gares.geojson` | "Liste des gares", 2024-03-28, 6,469 rows | UIC code, line, PK and `voyageurs` O/N for each station on each line |
| `wikidata_lines.json` | Wikidata SPARQL: items with P1671 (route number) and P17 = France | line names (fr label), English labels, P2043 length |

What these files do not have:

- **Line names.** SNCF used to publish `lib_ligne` as a name; it now carries the code. Names come
  from Wikidata (769 codes). OSM `route=railway` relations whose `ref` is the code fill a few
  more (22 in Ile-de-France). Anything left is "Ligne 830 000".
- **Passenger status per line.** The `voyageurs` flag is per station, and SNCF projects a
  station onto every line near it. The reader therefore asks OSM whether passenger routes run
  over each stop-to-stop section (see the docstring).
- **Some stations.** Paris-Montparnasse (only "Paris-Montparnasse 3 Vaugirard" is listed),
  Versailles-Rive-Droite and Châtelet - Les Halles are missing. OSM fills them.
- **Line colours.** No per-RFN-line colour exists. Transilien and RER colours are service
  colours and come from the OSM route relations (77 of 130 lines carry one in the Ile-de-France
  build). `colours/fr.csv` was deliberately not made.

`formes-des-lignes-du-rfn` (the shapefile dataset) holds the same 1,638 portions as
lignes-par-statut, with the same keys, so it is not used.

### The per-track file (added 2026-10-02, optional)

| file | dataset | what the reader takes from it |
|---|---|---|
| `voies-de-ligne.geojson` | "Fichier de formes des voies du réseau ferré national" (`fichier-de-formes-des-voies-du-reseau-ferre-national`), line tracks only (`type_voie` VPL, 1,602 of 9,956 tracks), 39 MB, ODbL | each line code's first track (V1, V1B, UNIQUE...) with its PK range |

Fetched by `--fetch` with `?where=type_voie="VPL"` on the export URL. It fills two holes in the
line files:

- **Lines the line files lack.** lignes-par-statut, lignes-par-type and liste-des-gares have no
  rows at all for these exploited lines, which the per-track file has (checked against the
  live API 2026-10-02, not a stale download): 226 310 LGV Interconnexion Est (57 km), 262 000
  Douai - Blanc-Misseron (the main line to Valenciennes), 657 000 Lamothe - Arcachon, 768 300
  the LGV branch Pasilly - Aisy to Dijon, 958 000 Bondy - Aulnay (T4). Their stations are
  missing from SNCF's list too (Arcachon, Marne-la-Vallée - Chessy), so such a line takes the
  passenger stations SNCF lists under other lines on its track, and OSM's stations. Codes the
  per-track file adds that carry no passenger trains (145 306, 245 300, 272 326, 312 300,
  354 000, 365 000, 811 000) fall out in build_model like any junction-ended freight section.
- **Coarse geometry.** One portion has a vertex only every 1.5 km: LGV Est from PK 301.5
  (Baudrecourt) to Vendenheim. 46 of its 124 km lay more than 40 m off OSM's rails, so its
  ways went unowned and 111 km of LGV Est counted for nobody. It now takes the track's shape
  (one vertex per ~90 m). No other portion is above COARSE_M (1 km per vertex); LGV SEA,
  Rhin-Rhône and BPL (670-720 m) already lie on the rails.

Without the file the reader skips both and builds as before.

**Not RFN, correctly absent**: Perpignan - Figueras (837 000, the concession line, 55 km of
track only the Renfe-SNCF trains use) is in lignes-par-type but in neither the statut file nor
the per-track file.

## Checks

- **Chainage.** `check_model.py --region fr` compares every line with its PK difference. On
  Ile-de-France: 48 lines, median 0.997, one line off by more than 5%: LGV Atlantique (431 000),
  1.10. Its only section here is Paris-Vaugirard junction to Massy-TGV, 14.4 km of geometry for
  13.2 km of PK. On the whole register without OSM (a dry run): 389 lines, median 0.997, 11
  lines off by more than 5%. They are register quirks, not build errors:
  - 899 000 Saint-Pierre-d'Albigny to Bourg-Saint-Maurice (1.12): the exploited portion's
    geometry includes a 10 km branch.
  - 180 000 Metz to Zoufftgen (0.90): the section from Hettange-Grande to the border spans
    13.8 km of PK for 8.7 km of track.
  - 289 000 Fives to Abbeville is fixed: its exploded parts arrive out of order and now chain in
    PK order.
- **Published lengths.** `REGISTER["fr"]` has 16 lines inside Ile-de-France, each against the
  "longueur" in its fr.wikipedia infobox (retrieved 2026-09-30). 13 are within 5%. The three
  that are not are lines where the build matches SNCF's own PK extent and Wikipedia measures
  something longer: 334 900 (0.92), 326 000 (0.84), 396 000 (0.92). The notes in check_model
  give the figures.

## Known limits

- **The Grande Ceinture (990 000) is mixed.** RER C, T12 and T13 each run on part of it, and a
  register line has one kind. It stays rail; T12 runs on rail track and credits it, T13's
  track is light_rail and does not (see "Track ownership gaps"). T11 (960 000) and
  Esbly-Crécy (071 000) are light_rail and are credited.
- **Parallel lines out of Paris-Saint-Lazare.** 334 000, 334 900, 340 000, 973 000 and 975 000
  are separate RFN lines on parallel track pairs, and each has its own Saint-Lazare to Asnières
  sections. Riding one credits the others over the shared corridor, which is how the register
  is built.
- **Connecting curves (Rac) are not lines.** 820 of them nationally, mostly unnamed and under
  3 km. A ride over one credits nothing on the register.
- **Station matches.** 393 of 403 stops in the Ile-de-France build took an OSM station's name and
  position. Of the other ten, eight lie just outside the extract. The remaining two are
  "Roissy-CDGX 2" on the unopened CDG Express line (025 000) and "La Défense CNIT" on EOLE
  (979 000), which OSM names differently. The build log lists the OSM stations beside a line that were refused (Rougemont
  Chanteloup is a T4 stop; Les Baconnets is RER B beside 985 000).

## Track ownership gaps (2026-10-02, cleanup)

What the ownership log (`own:` lines) showed as register gaps or named-train-only track, and
why:

- **Border stubs** (Longwy, Thionville - Apach, Morteau, Modane, Basel, Portbou): fixed in
  the reader, `snap_borders` (docstring). 30 line ends now end at their RINF border point
  under its `eEU` id. Measured on a trial build against an unchanged one: Basel 99.3 -> 100%,
  Portbou 99.2 -> 100%, Besançon - Le Locle 99.1 -> 100%, Lyon - Genève 99.6 -> 100% (and 15 km
  of Swiss track it claimed, greyed as not running, is gone). Thionville - Apach and Modane
  were already 100% under ownership. **Longwy (202 000) stays at 95%**: its last 1.2 km, from
  where the trains to Luxembourg leave it on the curve 202 100 to the Belgian border at Athus,
  has no passenger trains, and the line has no node there to end a section at. Modane's
  register stops 430 m short of the point inside the Fréjus tunnel and is left so (a straight
  line on to it was a stretch nothing could credit).
- **Tram-trains on rail track**: Nantes - Châteaubriant (519 000) was 2.6% creditable and
  Lyon-Saint-Paul - Montbrison (782 000) 0%, because the reader made them light_rail (their
  routes are tram-trains) while their track is railway=rail, so they owned none of it. The
  light_rail test now reads the track's tag: 519 000 -> 100%, 782 000 -> 93%; T11 (960 000)
  and Esbly - Crécy (071 000), on light rail track, unchanged. Side effect: 457 000 (2.4 km
  at Nantes-État) is now kept, as tram kind.
- **Lines SNCF's line files lack** (Interconnexion Est, Douai - Valenciennes, Arcachon,
  Pasilly - Aisy, Bondy - Aulnay) and **LGV Est's coarse geometry**: the per-track file above.
  Trial build with the file, against the same build without it: +5 register lines, +125.7 km;
  LGV Est 86.8 -> 99.5% creditable; 226 310 57.3 km, 262 000 30.1 km, 657 000 15.8 km,
  768 300 15.1 km all 100%, 958 000 7.8 km 97%; named-train-only track 556 -> 314 km.
  258 000 (a 0.6 km stub at Somain) drops out, lying on 262 000's rails; 025 000 (CDG
  Express, not open) loses its 57% to 226 310, whose track it shares at CDG 2.
- **Perpignan - Figueras** (55 km, named trains only): not RFN. Left.
- **800 000 near Valence** (19 km, only the ICN Paris - Briançon night train): its stop-to-stop
  section Saint-Péray - Le Teil is 39% ridden and dropped as freight; the train leaves 800 000
  at a junction with no node in that section. Left.
- **Long connecting curves (Rac)** the reader leaves out by design: Arras Sud (226 306, 19 km
  of way near Boisleux), Migné-Auxances, Sablé, Laval-Ouest, Pont-de-Veyle, La Couronne,
  Monts, Annet, Herny... about 100 km of way only TGVs use, counted nowhere. 48 exploited Racs
  are 3 km or longer (250 km). Whether to keep long Racs is open.
- **T13 on the Grande Ceinture** (990 000 Saint-Germain - Saint-Cyr, 13.5 km): T13's track is
  railway=light_rail and 990 000 is rail, so 990 000 owns its sections but no ride credits
  them, and T13 owns the same ways as an OSM line. T12 (Massy - Évry, rail track) credits
  990 000. Open: a register line has one kind.

## Running it

Ile-de-France (the development region, `data/proc/fr` as it stands now):

```
python extract.py --region fr --pbf data/raw/fr/ile-de-france-260929.osm.pbf     # 35 s
python build_model.py --region fr --register fr_register:data/raw/fr             # 20 s
python build_tiles.py --region fr                                                # 20 s
python check_model.py --region fr
```

The whole country replaces the Ile-de-France extract under the same region name, `fr`. Nothing
in the reader changes: the clip to the extract then keeps everything, and the border
sections the regional build drops come back.

```
curl -L -o data/raw/fr/france-260929.osm.pbf https://download.geofabrik.de/europe/france-260929.osm.pbf
    # 5.1 GB; the plain france-latest.osm.pbf URL may redirect-loop, as Switzerland's did
python fr_register.py --fetch                                                    # 30 s, refreshes the register files
python extract.py --region fr --pbf data/raw/fr/france-260929.osm.pbf            # about 8 min
python build_model.py --region fr --register fr_register:data/raw/fr             # about 3-6 min (estimate)
python build_tiles.py --region fr                                                # about 2-4 min (estimate)
python check_model.py --region fr
python tools/build_regions.py                                                    # once, after the first full build
```

Delete the .pbf once the extract is checked. Then add `REGISTER["fr"]` rows for lines outside
Paris (Paris-Marseille 862, Paris-Brest 622.4, Paris-Lille 251 are the easy ones), and read the
build log's lists of refused stations and dropped stop-to-stop sections.
