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
lignes-par-statut, with the same keys, so it is not used. `fichier-de-formes-des-voies-du-
reseau-ferre-national` (per track) was not tried.

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
  register line has one kind. It stays rail, so T12 and T13 rides do not credit it; T11 (960 000)
  and Esbly-Crécy (071 000) are light_rail and are credited.
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
