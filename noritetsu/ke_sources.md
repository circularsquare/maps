# Kenya (ke): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent.

### What runs (freshest evidence)

| service | track | status | evidence |
|---|---|---|---|
| Madaraka Express (SGR), Mombasa Terminus - Nairobi Terminus | SGR, 472 km | running, 3 a day each way (express, county/inter-county, night) | seat61.com/Kenya.htm, updated 6 June 2026; stops Mombasa, Mariakani, Miasenyi, Voi, Mtito Andei, Kibwezi, Emali, Athi River, Nairobi Terminus |
| Nairobi Commuter Rail: Nairobi - Embakasi Village (15 a day), - Syokimau (12), - Ruiru (4), - Limuru via Kikuyu (2), - Lukenya (2) | metre gauge | running | en.wikipedia "Nairobi Commuter Rail"; OSM has a route relation for each of the five (8489646-8, 13180873, 13180875) |
| Nairobi Central - Nairobi Terminus SGR connector (via Syokimau) | metre gauge plus the "SGR-MGR Passenger Rail Link" (6 km in OSM) | running, 4 a day timed to the SGR | seat61 |
| Mombasa Central - Mombasa Terminus (Miritini) link | metre gauge | running, several a day, 35 min | seat61 |
| Nairobi - Nanyuki "safari train" | metre gauge, via Thika, Sagana, Karatina | running, weekly (Fri out, Sun back) | seat61 June 2026, kenyarailway.com schedule |
| Nairobi - Kisumu "Kisumu safari train" | metre gauge via Naivasha, Nakuru | **greyed**: suspended July 2025 (washouts), ran only as a festive special 19 Dec 2025 - 12 Jan 2026, seat61 June 2026 "suspended as of January 2026". The Uplands - Longonot - Kijabe repair was test-run 19 Jan 2026 (rogerfarnworth.com Feb 2026 roundup) but no regular service since was found | kahawatungu.com, thekenyatimes.com, seat61 |
| SGR Nairobi - Suswa (Naivasha) | SGR phase 2A, 120 km | **greyed**: opened for passengers Oct 2019, absent from seat61's 2026 page and every 2026 schedule; OSM still has route 10167882 "Madaraka Express : Nairobi - Suswa" (stale, flag as named train or drop) | |
| Voi - Taveta, Nakuru - Malaba, Kisumu - Butere, Lukenya - Mombasa metre gauge | | no passenger service; not built | |

Decision: Kisumu greyed, not running. A festive-only train is not "about weekly"; flip `suspended=False` if Kenya Railways publishes a standing weekly timetable again.

### Line list (hand list, za/nafrica recipe)

Lines cut so each piece is wholly running or wholly not:

1. **Mombasa - Nairobi SGR** (Mombasa Terminus, Mariakani, Miasenyi, Voi, Mtito Andei, Kibwezi, Emali, Athi River, Nairobi Terminus). Published 472 km. OSM name "Mombasa-Nairobi Standard Gauge Railway" on 496 km of way (both tracks at stations), infra relation 7190306.
2. **Nairobi - Naivasha SGR** (Nairobi Terminus, Ongata Rongai, Ngong, Mai Mahiu, Suswa): greyed. Published ~120 km. Infra relation 9065596.
3. **Nairobi - Lukenya** (Nairobi Central, Makadara, Imara Daima, Syokimau, Athi River, Lukenya): running. Part of OSM "Nairobi - Mombasa Railway" / "Uganda Railway".
4. **Syokimau - Nairobi Terminus link** (the SGR-MGR passenger link): running, a few km.
5. **Makadara - Embakasi Village** branch: running.
6. **Nairobi - Ruiru - Thika - Nanyuki** (Nairobi Central, ... Ruiru, Thika, Makuyu, Sagana, Karatina, Kiganjo, Naro Moru, Nanyuki): running (commuter to Ruiru, weekly train beyond). Infra relation 2551184 "Nairobi-Nanyuki Railway". Published ~235 km (to confirm from Kenya Railways).
7. **Nairobi - Limuru** (via Dagoretti, Kikuyu): running commuter.
8. **Limuru - Nakuru - Kisumu**: greyed (relations 20511946 "Nairobi-Nakuru MGR", 1903732 "Nakuru-Kisumu MGR"). Nairobi - Kisumu about 400 km in OSM's named ways (181 + 215).
9. **Mombasa Central - Mombasa Terminus (Miritini)**: running, about 15 km of the old main line.

Expected: about 9 register lines, roughly 1,400 km, of which ~840 km running (SGR 472, Nanyuki ~235, the Nairobi commuter pieces ~100, Mombasa link ~15).

### Sources

- Station order and stops: OSM route relations (Madaraka Express 7329392, five NCR routes); seat61 for the Madaraka Express stops; kenyarailway.com/schedule (unofficial, but carries KR's timetables) for Nanyuki and Kisumu.
- Coordinates: OSM, 201 railway=station/halt objects, 180 named.
- Timetables/GTFS: none for rail. Digital Matatus (Mobility Database mdb-1815, CC BY-SA-ish University of Nairobi/MIT release) is matatus only in its current version: 136 routes, all route_type 3, no rail (sample in `data/raw/ke/survey/digitalmatatus_gtfs.zip`, 0.5 MB). The 2014 version was the one with commuter rail; not worth chasing, OSM's routes are newer.
- Wikidata: 32 station items, no line adjacency. Useless as a line source here.
- Licence: OSM ODbL; the rest is facts from timetables.

### OSM quality (Overpass, 2026-10-08)

- rail track 2,944 km, of which 1,860 km (63%) carries a `name`; `usage=main` on 2,194 km. Names are line names ("Mombasa-Nairobi Standard Gauge Railway" 496, "Nairobi - Mombasa Railway" 407, "Uganda Railway" 374, "Nakuru-Kisumu MGR" 215, "Nairobi-Nakuru MGR" 181, "SGR Nairobi Naivasha" 61...), so named track nearly works, but the SGR and old main line run side by side for 470 km with different names and the metre-gauge names are inconsistent; the hand list traced by rinf.py with `own` relations is safer.
- route=railway infra relations: Nakuru-Kisumu MGR, Kisumu-Butere, Nairobi-Nanyuki, Nakuru-Malaba, Nairobi-Naivasha SGR, Naivasha-Kisumu SGR (under construction/planned), Nairobi-Nakuru MGR, three unnamed.
- Geofabrik: `africa/kenya-latest.osm.pbf`, 335 MB.

### Recipe

Hand line list traced by rinf.py, like za and nafrica; stops from OSM passenger routes (`osm_stops`) where routes exist, `listed_only` with the timetable's stations for Nanyuki (no OSM route). Suggest one shared reader for the East/Southern African countries (`eafrica_register.py` with `eafrica_lines.py`, as nafrica_register.py does). NCR's five services stay OSM lines over the register pieces.

### Open questions

- Nairobi - Kisumu: watch for a standing timetable after the Kijabe repair.
- Nanyuki line km: no published figure found; check against the trace.
- The AFCON 2027 Nyayo - Talanta spur and the Riruta - Ngong line: not open.

## Build (2026-10-08)

Built by the eafrica agent with `eafrica_register.py` + `eafrica_lines.py` (nafrica's code, as
mideast_register does). Commands:

    python tools/slot.py 2 -- python extract.py --region ke --pbf data/raw/kenya-latest.osm.pbf --station-areas
    python eafrica_register.py --clip ke          # clip, tidy station names, name the NCR routes
    python eafrica_register.py --fill ke          # adds Mombasa Central (not in OSM)
    python build_model.py --region ke --register eafrica_register:data/raw/rinf/ke

**Result**: 9 register lines, 1,252 km: running 7 lines, 803 km (Mombasa - Nairobi SGR 461.7,
Nairobi - Thika - Nanyuki 230.2, Nairobi - Kikuyu - Limuru 46.9, Nairobi - Athi River - Lukenya
42.1, Mombasa - Miritini 13.8, Makadara - Embakasi Village 7.3, Nairobi Terminus - Syokimau
1.3); greyed 2 lines, 449 km (Limuru - Nakuru - Kisumu 349.1, Nairobi - Suswa SGR 99.9). Plus
NCR's five OSM routes as OSM lines over them. check_model: SGR 461.7 of 472 (0.98, station
to station). Nairobi - Suswa has no check: the 120 km published for phase 2A runs on to
Naivasha's container depot past the passenger station.

Decisions:
- Lines cut so each is wholly running or wholly greyed, as the survey listed. Kisumu greyed
  from Limuru (the managing session's call); Nairobi - Limuru runs (NCR's Limuru trains, 2 a
  day; OSM's route stops at Kikuyu).
- The weekly Nanyuki "safari train" counts as running (weekly = scheduled); the line starts at
  Makadara, where it leaves the Lukenya line, and carries every OSM station as a stop.
- Syokimau is on a 1.3 km spur past Nairobi Terminus, not on the main line (the trace from
  Imara Daima to Athi River does not pass it): its own line. The main line passes Nairobi
  Terminus, where the link trains call at its metre-gauge platform.
- Mombasa Central is not in OSM: placed where the metre-gauge main line ends by Mwembe Tayari
  (39.6617, -4.0578), `--fill`.
- The SGR's stops are seat61's (listed_only); the metre-gauge lines take every OSM station.
- OSM's two SGR routes (7190306, 7329392) duplicate the register line: rules/ke.py
  SKIP_ROUTES. The metre-gauge Mombasa - Nairobi routes and the Suswa route: NOT_SERVICE.
- NCR's routes are named "NCR : Nairobi <-> Syokimau", which build_model reads as "NCR" for all
  five; eafrica_register gives each a route_master "NCR Nairobi – Syokimau" etc.
- Station names lose a trailing "Railway Station"/"Train station"/"Station" (OSM maps many
  stations twice, "Kibera" and "Kibera Train station"), and same-named twins within 300 m
  that no route lists are dropped (eafrica_register.tidy).
- Colours: our own (colours/ke.csv, `picked`).
