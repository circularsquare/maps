# Austria register sources (built 2026-09-30)

What the Austrian build reads, where each piece came from, and what is still wrong with it.
Austria is the second country built with `rinf.py`; how the reader works is in its docstring,
and its per-country entry is `COUNTRY["at"]`. Downloads live in `data/raw/rinf/at/` and
`data/raw/at_*` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch at                                            # RINF + Wikidata, ~15 s
curl -L -o data/raw/austria-260929.osm.pbf https://download.geofabrik.de/europe/austria-260929.osm.pbf
python extract.py --region at --pbf data/raw/austria-260929.osm.pbf     # 1 min; delete the .pbf after
python inspect_region.py --region at
python build_model.py --region at --register rinf:data/raw/rinf/at      # 70 s
python build_tiles.py --region at                                       # 45 s
python check_model.py --region at
python rinf.py --dry at          # the reader alone, with its full log
```

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-09-30 (`sections.json`: 1,489 sections, each returned twice by the query, which the
  version step folds; `points.json`: 1,430 points). 160 line ids, 4,590 km.
- **OpenStreetMap**, Geofabrik `austria-260929.osm.pbf` (data to 2026-09-29), ODbL: track,
  stations, and 431 `route=tracks`/`route=railway` relations. ÖBB's `route=tracks` relations
  carry the route number as `ref="101 04"`; the `route=railway` ones mostly carry Kursbuch
  (timetable) numbers, 100, 300, 901, which is why `COUNTRY["at"]` trusts RINF's id outright
  (`rule_certain`).
- **Wikidata** (`wikidata.json`), CC0: items with a route number (P1671) and P17 Austria, with
  their labels and lengths (P2043). ÖBB's route numbers are Wikidata's P1671 written "101 01",
  so they join RINF's ids directly. Names come from here; the lengths in
  `check_model.REGISTER["at"]` too (they are de.wikipedia's infobox figures).
- **ÖBB-Infrastruktur**: the route directory "VzG-Streckennummern" valid from 2025-12-14
  (`at_streckenverzeichnis.pdf`, https://infrastruktur.oebb.at/de/geschaeftspartner/schienennetz/dokumente-und-daten/oebb-streckenverzeichnis.pdf),
  which names every route ("10801 Wiener Neustadt Hbf = Staatsgrenze nächst Loipersbach"),
  and the 2026 route descriptions (`at_streckenbeschreibung_2026.pdf`, SNNB 2026 annex), which
  give speed, tunnels and signalling per route. Neither gives a route's length.

## What RINF carries in Austria

ÖBB-Infrastruktur (`0081_IM`) and seven others, which RINF names only by code; their names in
`COUNTRY["at"]["im"]` are read off the lines they hold:

- `3023_IM` **Steiermärkische Landesbahnen**, one id "StB" for three unconnected lines:
  Peggau - Übelbach, Feldbach - Bad Gleichenberg, Gleisdorf - Weiz, plus the Graz Süd
  terminal. `rinf.py` splits an id into its connected pieces; the three come out named from
  their OSM relations: Übelbacherbahn, Gleichenberger Bahn, Weizer Bahn.
- `3035_IM` **Montafonerbahn**, Bludenz - Schruns (id 62101).
- `3787_IM` **Raaberbahn** (GySEV), Neufeld an der Leitha - Wulkaprodersdorf - Baumgarten
  (60101), and `3782_IM` **Neusiedler Seebahn**, Bad Neusiedl am See - Pamhagen (60201).
- Three freight stubs with no passenger service: Hafen Krems (3865), WienCont (3764), Linz AG
  (3882).

Not in RINF, so they stay OSM lines as before: GKB (Graz-Köflacher Bahn), Wiener Lokalbahnen,
Stern & Hafferl, Zillertalbahn, Pinzgauer Lokalbahn, Salzburger Lokalbahn, Achenseebahn,
NÖVOG's Mariazellerbahn and Wachaubahn, the Stubaitalbahn, and every metro and tram.

Every point of the Landesbahnen, the Montafonerbahn and the Raaberbahn lacks a coordinate in
RINF (93 points). `rinf.py` places a point with no coordinate at the OSM station of its name,
nearest a neighbour already placed; 55 of the 93 are placed that way. The rest are sidings
("AB ...") and the handover points to ÖBB ("Grenze OEBB-STLB Km 190,945"), which cannot be
placed, so the stub from each private line to its ÖBB junction station is lost (0.5-3.5 km
each).

## Names

Wikidata's label for the route number, in German: "Salzburg-Tiroler-Bahn", "Pyhrnbahn",
"Bahnstrecke Wien - Knoten Wagram". Where several items carry one number, the established name
beats a generic "Bahnstrecke A - B" item, and any item whose own length is under 0.4 or over
2.5 times the route's is skipped: route 413 01 (Bruck an der Mur to the border at Thörl-Maglern,
226 km) carries both "Bahnstrecke Bruck an der Mur-Leoben" (23 km) and "Rosentalbahn" (63 km),
and is neither. With no fitting label the name is "first - last" station. The route number is
the line's `ref`, "101 03".

No `colours/at.csv`: ÖBB publishes no line colours for its routes, and the S-Bahn colours
belong to services, which stay OSM lines with the colours OSM gives them.

## Counts (2026-09-30)

- 125 register lines, 4,342 km, after build_model dropped junction-ended sections no OSM
  passenger route runs over (57 sections, 262 km).
- 354 lines in all with the OSM ones, 2,220 stations, 19,811 route-km.
- 1,289 RINF passenger-typed points, of which 1,091 are an OSM station (1,054 distinct), 30 by
  distance alone (checked: "Ybbs a.d.Donau" against "Ybbs an der Donau" and the like). The
  other 196 are closed halts, freight points and sidings ("AB ...").

## Check

`python check_model.py --region at`: against RINF's own section lengths, 98 lines of 2 km or
more, median 1.000. Against Wikidata/de.wikipedia, 15 lines, 9 within 5%. The other 6:

- **Salzburg-Tiroler-Bahn, Pyhrnbahn, Steirische Ostbahn** (0.94-0.95): RINF's own figure is
  the same fraction of the article's, so the gap is between the two published figures, not
  in the build.
- **Neue Unterinntalbahn** (0.92): RINF's 38.5 against the article's 40.2.
- **Übelbacherbahn, Montafonerbahn** (0.86, 0.84): the stub from the ÖBB handover point, which
  has no coordinate.

Most Austrian articles describe the historic railway (Südbahn, Ostbahn, Nordbahn,
Franz-Josefs-Bahn, Ennstalbahn) over a different extent from ÖBB's numbered route, so they are
not in REGISTER; the RINF comparison covers every line.

## Still off, and why

- **OSM is missing the S-Bahn trunk (Stammstrecke) through Wien Mitte.** There is no
  `railway=rail` between Praterstern and Rennweg in the extract, most likely mapped as
  construction during the works there. Three sections of 122 01 and 191 01 are left out; the
  crow-fly guard in `rinf.py` stopped them being "traced" at 0.1 km onto the same distant
  track.
- **OSM gap at Katzelsdorf**: the Mattersburger Bahn (108 01) stops 416 m short of Wiener
  Neustadt in OSM, so Wiener Neustadt - Katzelsdorf - Neudörfl has no path. It is not bridged,
  since that would be a straight-line guess.
- **RINF's length is misallocated** in 64 places, kept because the trace runs on the line's own
  OSM track or two independent traces agree ("length off" in the log). The largest: 170 01
  Deutschkreutz to the Hungarian border, 8.87 km in RINF for 2.9 km of track (why
  Burgenlandbahn reads 0.32 against RINF); 413 01 Leoben - St. Michael, 15.3 km for 9.1.
- **29 merged sections rejected**: the Stammstrecke and Katzelsdorf above, Feldkirch - Tisis
  (303 01, no path to the Liechtenstein border in OSM), the private lines' handover stubs, and
  port sidings.
- **Short connecting routes** (101 12, 114 11, 130 11 and the like) come out as their own lines
  named "first - last", because ÖBB numbers every connecting curve. They are real routes that
  passenger trains use, but a rider will rarely think of them as lines.
- **RINF leaves stretches out of lines** (2026-10-05). Our fetch has no section of 222 01, the
  Tauernbahn, from Mühldorf-Möllbrücke to Pusarnitz-Süd, Markt Paternion to Paternion-
  Feistritz, or Gummern to Villach, under any id, so it came out in four pieces; 47 of
  Austria's lines were in pieces. `fill_holes: True` (rinf.fill_holes) joins a line's pieces
  over its own OSM `route=tracks` relation: 113 gaps, 373 km, the Tauernbahn whole (114.5 km,
  Pusarnitz a stop) and the Kamptalbahn whole (Stiefern - Schönberg am Kamp, once a pair
  that lost to a shorter fill was asked again). 10 lines stay in pieces, each with a reason
  in RINF_FILL_DEBUG=1's log: no own relation (10111), the gap over another line's track
  (10101 Wien Penzing - Hütteldorf), too little of it on the own relation (10105 into
  Innsbruck Hbf, 62%), no path in OSM (12201 Stammstrecke through Wien Mitte; 10701 the
  Leobersdorfer Bahn's Weissenbach - Hainfeld, probably closed). Unnumbered lines whose ends
  moved are renamed "first - last" accordingly (ids unchanged). RINF itself has no section
  of the Tauernbahn's three holes under any id or country (queried 2026-10-05).
