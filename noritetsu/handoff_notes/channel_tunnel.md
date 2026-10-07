# Channel Tunnel: state at hand-off (2026-10-05, gb/fr agent)

Anita, 2026-10-05: "at some point we seem to have removed the train between uk and europe".
Nothing on the map ran UK - France; the tunnel drew as faint grey track. The full account is in
`fr_sources.md` "The Channel Tunnel" and `gb_sources.md` "The Channel Tunnel"; this file is the
short version plus what is left to do. **No real rebuild has been run** (managing session's
instruction: wait for the other agent's ownership.py / build_model.py change, then one combined
rebuild).

## Cause (it never drew right; no recent regression)

1. **No French line under the tunnel.** The tunnel is Getlink's concession, not RFN. SNCF's 216 000
   (Fretin - Fréthun) ends at Bif. Fréthun-lès-Calais, 400 m short of the portal, so the 22.7 km
   French half belonged to nobody. gb's High Speed 1 did own the UK half (both bores) to eEU00228.
2. **The border point was off the track.** RINF's eEU00228 lay 450 m off OSM's bores, and
   build_model.border_tails only finds a point within 60 m (borders.NEAR_M): no Eurostar or Le
   Shuttle route in gb or fr ran on to the border. fr built Eurostar only towards Belgium; gb
   built no Eurostar at all.
3. **Each extract held the other country's half.** OSM cuts both bores at the boundary and
   Geofabrik keeps whole ways, so fr.pmtiles drew the UK half grey (no fr line) and gb.pmtiles the
   French half grey, over or under the neighbour's line.

Found on the way: HS1 had no London station (its 0.3 km section into St Pancras International was
dropped as unridden), and Eurostar's London stop resolved to the Underground's "King's Cross St
Pancras".

## What changed (files)

- `borders.py` (landed by the managing session): `MOVE["eEU00228"] = (1.496109, 51.014814)`,
  midway between the bores' boundary nodes 1567644045 and 329586625, 25 m from each.
- `fr_register.py`: `channel_tunnel()` + `TUNNEL_*`, `_between()`: register line **"Tunnel sous la
  Manche"** (name_en Channel Tunnel, operator Eurotunnel, src "getlink", id
  line_id("tunnel-sous-la-manche") = fb32be2d155), one section eEU00228 - fj216000_1.7857_50.9209
  (Bif. Fréthun-lès-Calais, 216 000's own junction id), 23.11 km, geometry OSM's north bore and
  approach with each point moved halfway to the south bore (both bores 45-58 m apart; along one
  bore the other was outside the 40 m way buffer and unowned). `served_sections` set. Docstring
  bullet added. `--clip` added (calls gb_register.clip_channel("fr")).
- `gb_register.py`: `clip_channel(region)` + `CHANNEL_*` and `--clip`: drops the far half of the
  tunnel from data/proc/<gb|fr> (ways wholly east/west of the bores' cut in the Channel box).
  `SERVED_END`: HS1's sections to "London St. Pancras International" are served. Docstring and
  BORDER_M comment updated.
- `rules/fr.py`: Le Shuttle (service car_shuttle/car) is a named train; `extra_route_stops`: a
  Eurostar relation's `via` station becomes a stop (London - Brussels via Lille-Europe), also
  inserted into this build's copy of the relation so border_tails runs both ends to the borders.
- `rules/gb.py`: `extra_route_stops`: a train route's stop node resolved to a metro station by
  proximity only (the metro record's name lacks a word of the stop's name) moves to the rail
  station within 1.2 km whose name holds all the stop's words (Eurostar's "London St Pancras").
- `fr_sources.md`, `gb_sources.md`: sections "The Channel Tunnel"; `--clip` in both command lists.
- **Data written**: `gb_register.py --clip` and `fr_register.py --clip` have been run, so
  data/proc/gb (2 bores + 2 crossover ways out, 3 stops) and data/proc/fr (2 bores, 5 stops out)
  are already clipped. Backups of the unclipped ways.pkl/stops.pkl are in the session scratchpad
  (`.../scratchpad/procbak/`), temporary.

## Decisions (recorded in fr_sources.md)

- **Le Shuttle is a named train**, not a line: scheduled (up to 4/hour) but vehicles only, no foot
  passengers, one origin and destination. A ride on it credits the tunnel's two register lines;
  its terminal loops count for nobody.
- The French half is its own line (Getlink's, not RFN), not folded into 216 000; the UK half stays
  folded into High Speed 1 as before.
- Lille-Europe as a stop of the London - Brussels relations comes from their own `via` tag;
  London - Amsterdam relations say via Rotterdam and get nothing in France.

## Trial results (tools/ab.py, lines/stations only; foot/ways differ also for the other agent's
uncommitted ownership work)

- **fr** (ab3): 4 lines differ, stations 5482 -> 5484 (2 new: eEU00228 and the via stop's
  station record), none gone. New: register "Tunnel sous la Manche" 23.11 km; named trains
  "Eurostar : Paris ↔ London" Paris-Nord - border 350.81 km, "Eurostar : Bruxelles ↔ London"
  border - Lille-Europe - eEU00083 145.47 km, "Eurotunnel Le Shuttle" Coquelles - border 30.03 km.
  Tunnel line owns both bores and the approach (ways 143253048, 143253058, 1463709912) and
  credits itself whole; Eurostar's and Le Shuttle's tunnel stretch credits it.
- **gb** (ab3, with the first, too broad version of the rules/gb.py stop rule): HS1 148.89 ->
  149.18 km, sections 7 -> 8 (St Pancras International - junction, 0.47 km, now kept; the border
  section 49.05 -> 48.88 km, drawn to the moved point); new named trains, each St Pancras
  International - border: Eurostar Paris 138.90, Bruxelles 138.84, Amsterdam 138.84 km; Le
  Shuttle Folkestone - border 28.97 km. Stations 4084 -> 4085. That version also moved 25 stop
  nodes (Elizabeth line at Stratford/Paddington/Liverpool Street, Barking -> Barking Park...),
  so the rule was narrowed to name mismatches only. **Re-trial with the narrowed rule (ab4):
  gb 5 lines differ (1 register): HS1 as above and the 4 new named trains; stations 4084 ->
  4084, none gone or new; the only stop nodes moved are Eurostar's two "London St Pancras"
  nodes (King's Cross St Pancras -> London St. Pancras International).**
- **be** (ab, first round): 0 lines differ, stations unchanged. be needs no rebuild for this; its
  "Eurostar : Bruxelles ↔ London" and "Eurostar: Amsterdam ↔ London" pieces already end at
  eEU00083 / the Dutch side and join fr's by route-master id.
- Unit tests: 19 OK.

## Next steps

1. (done: the gb re-trial is clean.)
2. Once the ownership change lands: `python tools/compare_lines.py save gb fr` (snapshot for gb,
   fr, be already saved 2026-10-05 before any rebuild; re-save if dist changed since),
   `python tools/rebuild.py gb fr`, `compare_lines.py diff gb fr`, `check_model.py --region gb`
   and `--region fr`, `python -m unittest discover -s tests`, then **`tools/build_regions.py`**
   (managing session) so line_aliases/regions.json and closed.json pick up the new lines.
   **Rebuild gb and fr; be is not needed.**
3. Managing session: HANDOFF.md's "after every extract, clip" list should gain
   `python gb_register.py --clip` and `python fr_register.py --clip`.
4. Optional: `REGISTER["fr"]` in check_model.py has no row for the tunnel (Getlink publishes no
   per-line length; tunnel 50.45 km portal to portal, our halves 27.6 km of UK bore + 22.7 km).

## Still open

- "Eurostar: Amsterdam ↔ London" has no French piece (no stop in France in OSM), so it draws with
  a gap between the tunnel and Belgium. A ride still credits the register lines.
- Three HS1-named ways east of Ashford (158243849, 155331186, 1376434076; 4.5 km) are 40-170 m
  off HS1's drawn track and owned by no register line (Eurostar named-train track only).
- Le Shuttle's footprint in fr credits a short stretch of 216 000 beside the Coquelles terminal.
- Calais-Fréthun has no Eurostar stop in OSM.
