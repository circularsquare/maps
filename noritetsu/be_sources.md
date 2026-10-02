# Belgium register sources (built 2026-09-30)

What the Belgian build reads, where each piece came from, and what is still wrong with it.
Belgium is the first country built with `rinf.py`, the generic ERA RINF reader; how that
reader works is in its docstring. Downloads live in `data/raw/rinf/be/` and `data/raw/be_*`
(gitignored); this file is the tracked record of them. Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch be                                          # RINF + Wikidata, ~10 s
curl -L -o data/raw/belgium-260929.osm.pbf https://download.geofabrik.de/europe/belgium-260929.osm.pbf
python extract.py --region be --pbf data/raw/belgium-260929.osm.pbf   # 1 min; delete the .pbf after
python inspect_region.py --region be
python build_model.py --region be --register rinf:data/raw/rinf/be    # 45 s
python build_tiles.py --region be                                     # 20 s
python check_model.py --region be
python rinf.py --dry be          # the reader alone, with its full log (every rejection)
```

The plain `belgium-latest.osm.pbf` URL answered with a redirect to itself on 2026-09-30; the
dated file from https://download.geofabrik.de/europe/belgium.html worked (data to 2026-09-29).

## Sources

- **ERA RINF** (Register of Infrastructure), SPARQL at
  `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched 2026-09-30 by
  `rinf.py --fetch be` into `sections.json` (1,762 sections of line, 444 line ids, 3,972 km)
  and `points.json` (1,325 operational points). Licence: ERA states EUPL 1.2 for the service;
  the Zenodo dump of the same graph is CC BY 4.0. Only Infrabel (`0088_IM`) is registered.
- **OpenStreetMap**, Geofabrik `belgium-260929.osm.pbf`, ODbL: track geometry for every
  section, the station each RINF passenger point is, and the 316 `route=railway` relations
  (`ref=L36`, `L50A`, `L01`-`L03` for line 0's three track pairs) that give public line
  numbers to Infrabel's internal RINF ids.
- **Wikidata**, `wikidata.json`: 197 route numbers (P1671) with P17 Belgium and their labels,
  CC0. Used for English names where the label says more than "line N" (HSL 1-3, line 0
  "North-South connection", 40 "Liège-Maastricht railway").
- **Infrabel's network statement**: the 2024 draft (for how Infrabel writes line numbers:
  "L.36", "L.3 en L.4" in running text) and annex D (`be_nv_bijlagenD.zip`, from
  https://infrabel.be/nl/netverklaring, "Bijlage D. RIEI"), whose list D.1 names every line
  with its ends, tracks, speed, signalling and electrification. It gives no lengths.
- **nl.wikipedia** "Lijst van spoorlijnen in België" (`nlwiki_lijst.txt`) and the article of
  every line it links (`nlwiki_lines.json`, via the MediaWiki API), retrieved 2026-09-30. The
  infobox `LENGTE` is the published length in `check_model.REGISTER["be"]`. It carries no
  citation; it agrees with the chainage in the article's own route diagram except on lines 16,
  73, 96 and 112, where the note in REGISTER says which figure is which.

## Names

RINF has Infrabel's internal four-digit ids, not the public numbers. `NNN0` is line NNN
(0360 is 36), but other endings are Infrabel's variant codes and do not map onto the public
letters (0503 is 50A, 0366 is 36N), and even NNN0 is sometimes a short curve rather than the
line (0580 is 0.7 km at Ledeberg; line 58 is 0582). So the rule is trusted only where the
traced line lies on an OSM relation of that number, and otherwise the relation decides. Line
0's six ids (0010-0060, one per track) and HSL 1-4 (9010-9040) are fixed in `BE_FIXED`.

Of 444 ids: 10 fixed, 87 by the rule confirmed on the map, 14 by the rule with no relation
near, 191 from an OSM relation, 53 joining a number another id already has (on its track), 89
with no number. The ones with no number are almost all freight sidings and yard throats that
build_model then drops; 17 survive as short register lines named "first - last", such as
"Charleroi-Central - Marcinelle" and "Hasselt - Y.West Driehoek Hasselt".

Line names are Infrabel's form, `L.36`; English names `Line 36`, or Wikidata's label where it
says more (`High Speed Line 1`, `HSL 2`, `Line 0 (North-South connection)`).

No `colours/be.csv`: Infrabel publishes no line colours, and the SNCB S-train and IC colours
belong to services, which stay OSM lines with the colours OSM gives them.

## Counts (2026-09-30)

- 145 register lines, 3,196 km, after build_model dropped the junction-ended sections no OSM
  passenger route runs over (freight lines, port and yard track).
- 285 lines in all with the OSM ones (SNCB services, STIB metro and trams, De Lijn's coast
  tram and Antwerp trams, TEC's Charleroi metro), 1,383 stations, 11,231 route-km.
- 752 RINF passenger-typed points, of which 601 are an OSM station (554 distinct stations; SNCB
  has about 550). The other 151 are freight "stations" (Antwerpen-D.S.-*, yard bundles) and
  closed halts, and become junction ends.

## Check

`python check_model.py --region be`: against RINF's own section lengths, 111 lines of 2 km or
more, median 0.997. Against nl.wikipedia, 61 lines, 50 within 5%. The other 11, all explained
in their REGISTER notes:

- **Different extent**: 26 (Wikipedia counts from Schaerbeek; RINF's line starts at Y.Rue de
  Bruel), 69 (closed beyond Poperinge), 1 (RINF counts HSL 1 from Y.Noord Halle), 161D and 0
  (station centre to station centre against end of track), 36N (RINF files the Leuven curves
  under it).
- **The Wikipedia infobox disagrees with its own route diagram and with RINF**: 16, 73, 96,
  112 (L.112 reads 1.43 because the infobox says 14.1 where the diagram says 20.9 and RINF
  20.2).
- **A rejected trace**: 167, the last 1.4 km into Athus.

Lines left out of REGISTER because the article counts a closed or foreign stretch the build
rightly does not have: 12's Dutch part (REGISTER uses the Belgian part), 21A (Genk-Maaseik),
29 (Dutch part), 40, 44 (Spa-Stavelot), 52 (Dendermonde-Puurs), 54 (Dutch part), 57, 58
(Eeklo-Brugge), 82, 86, 90, 123, 132, 154. Lines with no passenger service (10, 11, 17, 20,
24, 39, 55, 141, 155) build nothing, as they should.

## Still off, and why

- **RINF's length is misallocated between neighbouring sections** in 62 places, kept because
  the trace runs on the line's own OSM relation track, or because tracing through every point
  and straight end to end agree and are no detour (`length off` in the log). The largest:
  line 66 Torhout-Zedelgem, 13.94 km in RINF for 7.96 km of track, which is why L.66 reads
  0.90 against RINF but 0.96 against Wikipedia.
- **16 merged sections rejected** (`rejected` in the log), nearly all freight: Antwerp and
  Ghent port sidings, Zeebrugge bundles, cement works connections. Passenger ones: Athus to
  Athus-Frontière (L.167) and De Panne to its buffer stop (L.73).
  Eight point pairs are too far from any track to trace: L.155 Marbehan-Croix-Rouge (a
  freight stub, 11.6 km), 17 (Ham), 48/49 (Raeren, the Vennbahn, lifted), Ath-Ghislenghien,
  Y.Saint-Lambert-Gantaufet.
- **Parallel lines on one corridor share track in the app.** 36 and 36N now trace on their own
  OSM relations, but where OSM puts all four tracks in one relation, or none, both lines draw
  over the same pair and a click there offers both.
- **161A on 161's rails, Bakenbos - La Hulpe - Genval - Rixensart** (6.3 km of L.161A counts as
  L.161 under track ownership). Not a tracing error (checked 2026-10-02): RINF's 1613 is the
  second track pair of the quadrupled Brussels - Ottignies line, which OSM calls 161A, and
  RINF lists 1613 sections over Bakenbos - La Hulpe - Genval - Rixensart beside 1610's. OSM
  maps four tracks at Hoeilaart (L161 and L161A ways) but only one pair at La Hulpe, where
  its 161A relation runs on the L161 ways. So both lines lie on one pair there and ownership
  gives it to the lower ref, as it should for shared rails; riding either credits L.161. If
  the quadrupling is built there, it is OSM that lacks the second pair; nothing for be.py.
- **17 register lines have no number**: connecting curves and depot leads that passenger
  routes do run over, named "first - last" from their ends. Their public numbers exist
  (Infrabel's annex D.1 lists them) but OSM has no relation to carry them onto the RINF id.
