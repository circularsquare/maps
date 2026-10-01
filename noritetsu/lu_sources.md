# Luxembourg register sources (built 2026-10-01)

What the Luxembourg build reads, where each piece came from, and what is still wrong with it.
Built with `rinf.py`, the generic ERA RINF reader, and `rinf_countries/lu.py`. Downloads live
in `data/raw/rinf/lu/` and `data/raw/lu_*` (gitignored); this file is the tracked record of
them. Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch lu                                              # RINF + Wikidata, ~10 s
curl -L -o data/raw/luxembourg-260930.osm.pbf https://download.geofabrik.de/europe/luxembourg-260930.osm.pbf
python extract.py --region lu --pbf data/raw/luxembourg-260930.osm.pbf   # 5 s; delete the .pbf after
python inspect_region.py --region lu
python build_model.py --region lu --register rinf:data/raw/rinf/lu       # 4 s
python build_tiles.py --region lu                                        # 3 s
python check_model.py --region lu
python rinf.py --dry lu          # the reader alone, with its full log
```

The plain `luxembourg-latest.osm.pbf` URL answered with a redirect to itself on 2026-10-01
(as Belgium's and Switzerland's did); the dated file listed on
https://download.geofabrik.de/europe/luxembourg.html worked (45 MB, data to 2026-09-30).

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01: 98 sections of line, 19 line ids, 263.7 km, and 94 operational points (8 of
  them border points). One infrastructure manager, CFL (`0082_IM`). Licence: EUPL 1.2 for
  the service, CC BY 4.0 for ERA's Zenodo dump of the same graph.
  **LU's points have no coordinate where rinf.py first looked**: no `wgs:lat` on the
  netReference, no `geo:hasGeometry` on the point. Every point has one on its netReference's
  geometry, which `"netref_wkt": True` in lu.py makes `--fetch` read (the `NETREF_WKT` join
  in rinf.py, added for Luxembourg, a no-op for every other country).
- **CFL's network statement**, *Document de Référence du Réseau 2026* (version 0.5,
  2024-11-20), published by the Administration des chemins de fer:
  https://acf.gouvernement.lu/dam-assets/sillon/documents-de-reference-du-reseau/2026/drr-2026-fr-publication.pdf
  (`data/raw/lu_drr_2026_fr.pdf`, 210 pages). Annex 2A has one sheet per line (pages 109-183)
  with its number, name, "Distance", stations, coordinates and speeds. Those distances are
  the published figures in `check_model.REGISTER["lu"]`. Section 2.3.1 gives the network as
  100.3 km single and 160.3 km double track (260.6 km). No licence is stated; it is a public
  regulatory document, and only line lengths are taken from it.
- **Wikidata**, `wikidata.json`, CC0: P1671 route numbers with P17 Luxembourg. Every RINF id
  has an item with its CFL number (Q802704 is line 1, Q16763490 is 1a...), with lengths that
  match the DRR. Fetched with fr/lb/de labels only (see Names).
- **OpenStreetMap**, Geofabrik `luxembourg-260930.osm.pbf`, ODbL: track geometry, stations
  (each with CFL's own code in `railway:ref`, the same as the RINF uopid less "LU"), 83 route
  relations (CFL, SNCB, SNCF TER, DB Regio, Luxtram, the funicular, two heritage railways),
  and 9 CFL `route=tracks` relations carrying 1a, 1b, 5, 6b, 6c, 6e, 6g, 6h, 6j, 6k.

## Names

RINF's line id is CFL's line name, "Luxembourg - Troisvierges-frontière", not its number. The
19 ids map one to one onto CFL's numbers 1, 1a, 1b, 2b, 3, 4, 5, 6, 6a-6h, 6j, 6k and 7
(Wikidata P1671 and the DRR's annex 2A agree), so they are fixed in `LU_FIXED` rather than
read from OSM, which numbers only nine of them and put all three Pétange - Rodange ids
under 6j. The name is CFL's own form, number then line name: "Ligne 1 Luxembourg –
Troisvierges-frontière", English "Line 1 (Luxembourg – Troisvierges-frontière)". Wikidata's
English labels ("CFL Line 1", "Ettelbréck - Dikrech railway line") are left out of the fetch
because rinf.py would put them in place of, or after, that template.

No `colours/lu.csv`: these are infrastructure lines, which CFL does not colour. The CFL
service lines (10, 30, 50, 60, 60a, 60b, 70) stay OSM lines with the colours OSM gives them.

## Counts (2026-10-01)

- **16 register lines, 237 km**. 18 lines came out of RINF (258 km traced against 261 km of
  RINF length, every section traced, none rejected); build_model then dropped five
  junction-ended sections (21 km) that no OSM passenger route runs over: all of 2b
  (Ettelbruck - Bissen) and 6k (Brucherberg - Scheuerbusch), which are freight, and line 4's
  Berchem - Syren - Oetrange freight bypass.
- 45 lines in all with the OSM ones: CFL service lines, SNCB IC-33 and L-12, TER Grand Est,
  DB Regio RB82/83/87/RE11, the IC 37 to Düsseldorf, **Luxtram T1** (16.1 km, Luxexpo /
  Findel - Stadion), **the Pfaffenthal-Kirchberg funicular** (0.2 km), and the Minièresbunn
  and Train 1900 heritage lines at Fond-de-Gras. 118 stations, 957 route-km.
- 67 RINF passenger points, all 67 an OSM station; two by distance alone (Goebelsmuhle is
  "Goebelsmühle" in OSM, 31 m; "Luxembourg secteur Hollerich" is OSM's Hollerich, 35 m, which
  CFL-70 and the Longwy RE call at). Every CFL station in the extract is on a register line.

## Check

`python check_model.py --region lu`: against RINF's own section lengths, 14 lines of 2 km or
more, median 0.993, none off by more than 5%. Against the DRR, 14 lines, 13 within 3%. The
one flagged, 6g at 1.07, is a 1.6 km line: OSM's Rodange station is 135 m east of RINF's
Rodange point, and the trace starts at the station.

Not listed in REGISTER, each explained in the comment there: 4 (DRR 16.2, 7.4 built: only
Luxembourg - Fentange Sud - Berchem has passenger routes), 6e (DRR 2.7 is to Audun-le-Tiche
station in France; built to the border, 1.5), and the freight lines 2b, 6d, 6k. 6g and 6h are
checked from Rodange: the DRR counts them from Pétange, but RINF files the shared Pétange -
Rodange (2.6 km) under 6j only.

## Border sections

Every CFL line into a neighbour stops at the border point in this build; RINF lists each one,
and all eight are within 40 m of OSM's national boundary except the one corrected below.

| Line | Border section | km | Continues as |
|---|---|---|---|
| 1 | Troisvierges - Troisvierges frontière | 7.86 | Infrabel L42 to Gouvy |
| 3 | Wasserbillig - Wasserbillig frontière | 0.54 | DB 3140 (Igel) to Trier |
| 5 | Kleinbettingen - Kleinbettingen frontière | 0.93 | Infrabel L162 (Sterpenich) to Arlon |
| 6 | Bettembourg - Bettembourg frontière | 5.17 | SNCF 180 000 (Zoufftgen) to Thionville |
| 6e | Esch-sur-Alzette - Esch frontière | 1.52 | SNCF/CFL 186 000 to Audun-le-Tiche |
| 6g | Rodange - Rodange frontière (Aubange) | 1.61 | Infrabel L165/1 to Aubange |
| 6h | Rodange - Rodange frontière (Mont-St-Martin) | 2.68 | SNCF 202 100 to Longwy |
| 6j | Rodange - Rodange frontière (Athus) | 1.60 | Infrabel L167 to Athus |

6b is the exception: RINF carries it to Volmerange-les-Mines station, in France, and the
build does too (Dudelange-Usines - Volmerange-les-Mines, 1.8 km). The Moselle line on the
German bank (Perl - Trier, RB82) is DB's and is an OSM line here only because Geofabrik's
extract reaches over the river.

## Still off, and why

- **RINF's Athus border point is misplaced**: "Rodange frontière B A" (EU00098) is at 5.8106
  49.5518, 870 m inside Belgium, on the longitude of the Mont-Saint-Martin point and the
  latitude of the Aubange one. `lu_fix` moves it to where 6j's OSM relation and Infrabel's
  L167 cross the border (5.82501 49.55180), which matches RINF's 1.478 km from Rodange.
  Unfixed, 6j's border section traced 2.53 km.
- **6d, Tétange - Langengrund, is left out by `skip_line`**. It is a freight branch, but
  its first 1.6 km run beside 6c into Rumelange, where CFL's 60b runs, and that alone kept the
  whole 3.3 km section as ridden.
- **Line 4 shares a corridor with line 6.** Luxembourg - Fentange Sud is line 4's own track
  pair beside line 6 (85% of it within 30 m of line 6's trace, 25% within 8 m), and
  Berchem - Berchem Nord lies on line 6's alignment. Both carry passenger routes, so both
  lines are drawn there and a click on that stretch offers both, as with Belgium's 36/36N.
- **Line 8**, the new Luxembourg - Bettembourg line under construction, is not in RINF and
  builds nothing (Wikidata has its number).
- **Foreign OSM lines inside the extract**: Geofabrik's Luxembourg file reaches a few km
  past the border, so DB's RB82 (Perl - Trier, 25 km of it), SNCB's L-12 (17.5 km near Arlon)
  and a piece of TER P51 / RE16 at Apach are built as lu OSM lines. Apach appears twice, as
  two station records 30 m apart. Nothing is clipped to the country outline. This happens to
  every country built from a Geofabrik extract.
- **Timetable check not wired**: the national GTFS is open (Administration des transports
  publics on data.public.lu, CC BY 4.0, weekly; AVL, CFL, Luxtram, RGTR and TICE in one
  feed): https://data.public.lu/en/datasets/horaires-et-arrets-des-transport-publics-gtfs/ ,
  latest file on 2026-10-01
  https://download.data.public.lu/resources/horaires-et-arrets-des-transport-publics-gtfs/20261001-055928/gtfs-20260930-20261212.zip
  (16.8 MB, valid 2026-09-30 to 2026-12-12). The file name changes every week, so take the
  newest resource from https://data.public.lu/api/1/datasets/horaires-et-arrets-des-transport-publics-gtfs/ .
  Mobility Database lists CFL as mdb-1108.
