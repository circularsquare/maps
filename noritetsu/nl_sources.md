# Netherlands register sources (built 2026-09-30)

What the Dutch build reads, where each piece came from, and what is still wrong with it. The
Netherlands is the third country built with `rinf.py`; how the reader works is in its
docstring, and its per-country entry is `COUNTRY["nl"]`. Downloads live in
`data/raw/rinf/nl/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch nl                                                 # RINF + Wikidata, ~10 s
curl -L -C - -o data/raw/netherlands-260929.osm.pbf https://download.geofabrik.de/europe/netherlands-260929.osm.pbf
python extract.py --region nl --pbf data/raw/netherlands-260929.osm.pbf     # 1.5 min; delete the .pbf after
python inspect_region.py --region nl
python build_model.py --region nl --register rinf:data/raw/rinf/nl          # 40 s
python build_tiles.py --region nl                                           # 20 s
python check_model.py --region nl
python rinf.py --dry nl          # the reader alone, with its full log
```

The 1.4 GB download dropped once with a connection reset; `curl -C -` resumed it.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-09-30: 871 sections of line (one version each), 778 points, 134 line ids, 3,148 km.
  Only ProRail (`0084_IM`) is registered.
- **OpenStreetMap**, Geofabrik `netherlands-260929.osm.pbf` (data to 2026-09-29), ODbL: track
  and stations. There are only 54 `route=railway`/`route=tracks` relations in the extract,
  none of them numbering ProRail lines, so OSM gives no names here.
- **Wikidata** (`wikidata.json`): 314 route numbers (P1671) for the Netherlands, but they are
  ProRail's numbered trajectories ("009 Meppel - Leeuwarden"), which do not join RINF's ids.
  Not used for names.
- **nl.wikipedia** "Lijst van spoorlijnen in Nederland" (`nlwiki_lijst.txt`) and every current
  line article it links (`nlwiki_lines.json`, via the MediaWiki API), retrieved 2026-09-30.
  The infobox `LENGTE` is the published length in `check_model.REGISTER["nl"]`.

## Names

ProRail's RINF id is its line's two end stations as station codes: "Asd-Rtd" is Amsterdam -
Rotterdam, the line nl.wikipedia calls "spoorlijn Amsterdam - Rotterdam". Each code is a RINF
point's uopid ("NLAsd"), so `nl_id_name` reads the name from RINF itself and drops
"Centraal". The id is the line's `ref`. Where a code is not a RINF point, the name falls back
to the line's first and last stations: Zp-Esg (Zutphen - Enschede grens) comes out as
"Enschede De Eschmarke - Enschede Grens". Junction-ended names such as "Barneveld Noord -
Ede-Wageningen" and "Lelystad Opstelterrein Aansl. - Zwolle" (the Hanzelijn) are ProRail's ends,
not the names riders use; nl.wikipedia's "Hanzelijn", "Flevolijn", "Kamperlijntje" would need a
hand-made map of ids to names.

No `colours/nl.csv`: ProRail's lines have no colours, and NS's service colours belong to
services, which stay OSM lines.

## Counts (2026-09-30)

- 96 register lines, 2,780 km, after build_model dropped junction-ended sections no OSM
  passenger route runs over (57 sections, 267 km: port and yard track, the freight-only
  Betuweroute).
- 286 lines in all with the OSM ones (NS and regional services, the Amsterdam and Rotterdam
  metros, trams, RandstadRail), 1,245 stations, 14,025 route-km.
- 408 RINF passenger-typed points, of which 404 are an OSM station (3 by distance alone, all
  right: Maastricht Randwijck/Randwyck and the like). The other four: Barneveld aansl. (a
  junction), Born and Oosterhout West stad (freight), Spekholzerheide (no station in OSM).
  Buitenpost is mapped only as stop positions and is matched through them.

## Check

`python check_model.py --region nl`: against RINF's own section lengths, 81 lines of 2 km or
more, median 0.996, 6 off by more than 5% (all yard and depot stubs of 2-6 km). Against
nl.wikipedia, 32 lines: 24 within 5%. The other 8, all in their REGISTER notes:

- **ProRail's line starts at a junction outside the town** the article starts at: Deventer -
  Almelo (Snippeling aansl.), Eindhoven - Weert (Tongelre aansl.), Apeldoorn - Deventer,
  Leeuwarden - Stavoren (Harinxmakanaal bridge), Meppel - Groningen.
- **Amsterdam - Rotterdam** (0.81): ProRail files Amsterdam - Haarlem under its own id, Ass-Rtd.
- **Utrecht - Boxtel, Gouda - Den Haag** (0.92, 0.94): RINF's own figure is that much shorter
  than the article's.
- **Breda - Rotterdam** (1.17): RINF's own Bd-Rtd is 59.0 against the article's 49.4; what the
  extra 10 km is has not been worked out.

## Still off, and why

- **OSM gap at Enschede**: the track at Enschede De Eschmarke is cut off from Enschede in OSM
  (an island of 188 nodes), so that section of Zp-Esg is left out rather than drawn straight.
- **Names are ProRail's ends**, as above.
- 18 sections kept although RINF's length disagrees ("length off" in the log), all short
  pieces around big stations and yards.
