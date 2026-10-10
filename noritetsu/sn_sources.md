# Senegal (sn): sources

## Build (2026-10-08)

    python extract.py --region sn --pbf data/raw/senegal-and-gambia-latest.osm.pbf --station-areas
    python wafrica_register.py --clip sn       # other countries' track, NOT_SERVICE routes,
                                               # track gaps joined, gauge breaks split
    python wafrica_register.py --fill sn
    python wafrica_register.py --convert sn
    python build_model.py --region sn --register wafrica_register:data/raw/rinf/sn
    python build_tiles.py --region sn
    python check_model.py --region sn

Built: **1 register line, 55 km, running**: TER Dakar – AIBD, its 14 stations
(OSM's TER route), listed, traced over the "TER Dakar-AIBD" named standard gauge track
(`OWN`, beside the metre gauge). Check: 55.4 against 36 + ~19 (1.01). OSM's TER routes
(the register line's twin under another name) and the Petit train de banlieue route are
dropped by --clip.

## Survey (2026-10-08)

Research only; nothing built. One line runs: the TER Dakar.

### What runs

- **TER Dakar (Train Express Régional), Dakar – Diamniadio – AIBD.** Standard gauge,
  25 kV, operated by SETER (SNCF's subsidiary) for the state's SENTER. Phase 1 Dakar –
  Diamniadio, 36 km, 14 stations, open since 27 Dec 2021. **Phase 2 Diamniadio – Aéroport
  Blaise Diagne (AIBD), about 19 km, opened to the public 28 Sept 2026** (free on that
  section until 4 Oct), so the line is now about 55 km. Every 8 minutes Dakar – Diamniadio,
  every 24 minutes on to AIBD; Dakar – AIBD about 55 min.
  Sources: dakaractu.com/TER-la-desserte-de-l-AIBD-officiellement-mise-en-service_a276475.html,
  senego.com (same news), au-senegal.com/ter-jusqu-a-l-aibd-ce-qui-va-changer-pour-les-voyageurs-au-senegal,18530.html,
  lesoleil.sn (June 2026: opening planned for September). fr.wikipedia "Train express
  régional Dakar-AIBD" for phase 1 (14 stations, 36 km), not yet updated for phase 2.
- **Not running**: Dakar – Bamako (Transrail; no passenger train since 2018, freight
  intermittent), the old Petit train de banlieue Dakar – Thiès (OSM still has its route
  relation 8530208; its Dakar end was rebuilt as the TER, and no PTB service has been
  reported since). No other passenger train in Senegal.

### OSM (Overpass, 2026-10-08, Dakar region bbox 14.4,-17.6,15.0,-17.0)

- Route relations **13645076 / 13645077 "Train Express Régional: AIBD - Diamniadio - Dakar"**
  (and the reverse), `network=Train Express Régional`, `ref=TER`. They already run to AIBD.
  Infrastructure relation 13607614 "Train Express Régional Dakar" (route=railway).
- Track: 210 `railway=rail` ways in the bbox; **103 named "TER Dakar-AIBD"** (all the
  standard gauge main line), 39 "Chemin de fer de Dakar au Niger" (metre gauge). 24
  station/halt objects, 22 named.
- Sample: data in this survey's scratch only (nothing saved under data/raw/sn).

### Timetables / GTFS

None found in Transitous (no `sn_` feed in api.transitous.org/gtfs/). The Mobility Database
catalogue CSV (share.mobilitydata.org/catalogs-csv) is behind a Cloudflare challenge for
scripts; Anita can open it in a browser and search "SN" if a feed matters. Not needed: the
TER is an interval service, and every station is a stop.

### Recipe

The smallest case in the project. Either:
- **OSM line only** (like Casablanca's trams in `nafrica_sources.md`): no register, the TER
  route relation becomes an OSM line. Simplest, but then Senegal has 0 register km.
- or (recommended, so the country has a register line) **one hand line in a shared West/
  Central Africa reader** (see `ng_sources.md`, "A shared reader"): `TER Dakar – AIBD`,
  stations from the route relation's stops, traced over the "TER Dakar-AIBD" named track by
  rinf.py, `kind=rail`, operator SETER. Expected ~55 km, 15-16 stations.

Extract: `africa/senegal-and-gambia-latest.osm.pbf`, **100 MB** (Geofabrik 2026-10-08).
Needs `--clip` against Senegal's outline (it includes The Gambia, which has no railway).

Check: Dakar – Diamniadio 36 km (WP, SETER), Diamniadio – AIBD ~19 km (press, Sept 2026).

### Licences

OSM ODbL. The press figures are facts only.

### Open questions

- Phase 2's intermediate stations (the press names none between Diamniadio and AIBD; OSM's
  route has the stops). Read them off the route relation at build time.
- Colour: the TER's livery/brand colour is not in OSM; `picked` if needed.
