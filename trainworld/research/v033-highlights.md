# v0.3.3 demand highlights, for the social post (2026-11-13, agent twstats)

Before = v0.3.2 as live (data 2026-11-11b: market rows from the release tree, which match the live site's
bundled `market-rows` asset byte for byte; pack files from R2 `data.anita.garden/trainworld/2026-11-11b/`).
After = master's freeze 20261113-013559 (what 8800 serves). No network in either: these are trips people want to
take (all modes), not rides. "Each way" for a pair is the game's own figure (the mean of the two directions), as
the trip explorer shows it. Every pair below has the same people and jobs before and after, so the gap-fill zones
play no part in them.

## Ready to post

1. **hakone gets about 8x the long trips it did.** about 1,100 a weekday each way before, about 9,100 now. japan's
   onsen towns count their real visitors now
2. **atami about doubles**, 1,700 to 3,500 long trips a weekday each way
3. **berlin to munich: 2.7x the trips** (1,430 to 3,900 a weekday each way). hamburg to munich 3x
4. **madrid to mallorca 2.6x, paris to nice 2.3x.** beach towns pull from far away now
5. **madrid to barcelona doubled**, 5,800 to 11,900 a weekday each way
6. **istanbul airport gets 13x the trips.** it sits just outside the istanbul zone, so nobody from the city went
   there before. about 8,600 a day then, about 111,000 now
7. **munich airport 4.7x**, about 12,600 trips a day to about 59,000. same story, it's outside the munich zone
8. **ubc's students used to sit in one spot 2 km from the bus loop.** now about 30,000 of them are within 800 m of
   it. and every university's students show up more often (attendance 60% -> 80%)

A shorter set if she wants 4: 1, 3, 6, 8.

## The figures and how each was measured

### 1-2. Japanese resort towns (T-3280)

79 Japanese resort towns' published visitors a year join the long-trip weight of their zones (4 people for each
visit a day). Trips a weekday each way, all modes, 25 km and more, between the town's zones and every Japanese
zone.

| Town | Before | After | x | Second check |
|---|---|---|---|---|
| Hakone (its five 6 km zones: Yumoto, Gora, Sengokuhara, Ashinoyu, Moto-Hakone) | 1,090 | 9,144 | 8.4 | about 8,990 (8.2x) |
| Atami | 1,746 | 3,498 | 2.0 | about 3,610 (2.1x) |

- First figure: T-3280's replica of the sim's long-trip build on the zone files (notes/T-3280.md "The fit", beta 4,
  the shipped value). Confirmed the shipped files carry it: `zone_tw` is in the new `tyo.market.json` and 30 other
  Japanese market files, absent from the v0.3.2 files, people and jobs per zone identical.
- Second check (independent arithmetic on the real files, old from R2 and new from the freeze): a zone's long-trip
  attraction `a` (build.rs) before and after. Hakone's five zones 12,455 -> 155,516 (x12.5); Atami's zone 17,741 ->
  53,667 (x3.0). Departures are kept, so only the arriving half scales; splitting each town's old trips into
  arriving and leaving by its jobs-to-people ratio and taking the ~1.3% normaliser rise gives the second-check
  column. Hakone agrees within 2%, Atami within 3%.
- "8x" holds whichever of the two you take (8.2-8.4). If she wants arrivals only: arrivals to Hakone grow about
  12x; that is the second method's estimate alone, so I would not post it.

### 3-5. The long-trip pull (T-3240, T-3254) and beach towns (market_resorts.json)

Big cities and resort places pull long trips more, faded in over 300-700 km (0.25 across a border). Trips a weekday
each way between the two places, all modes (car, plane, rail).

| Pair | km | Before | After | x |
|---|---|---|---|---|
| Hamburg - Munich | 612 | 477 | 1,432 | 3.00 |
| Berlin - Munich | 499 | 1,434 | 3,900 | 2.72 |
| Madrid - Mallorca | 570 | 652 | 1,710 | 2.62 |
| Paris - Bordeaux | 497 | 2,048 | 4,830 | 2.36 |
| Paris - Nice | 682 | 909 | 2,060 | 2.27 |
| Milan - Rome | 490 | 1,820 | 3,966 | 2.18 |
| Paris - Marseille | 644 | 1,645 | 3,577 | 2.18 |
| Madrid - Barcelona | 502 | 5,763 | 11,884 | 2.06 |
| Rome - Venice | 404 | 467 | 850 | 1.82 |
| Madrid - Malaga | 421 | 2,150 | 3,796 | 1.77 |
| Beijing - Shanghai | 1,074 | 9,838 | 15,651 | 1.59 |
| controls: Tokyo - Osaka, New York - Boston, Berlin - Hamburg, Rome - Naples | | | | 1.00, 1.00, 0.97, 0.97 |

- Mallorca, Nice and Malaga take the beach-town floor (pull excess 1.5); Barcelona, Munich, Paris their size.
- Method 1: a Python port of `placeTrips` (marketRule.ts) on each version's own market rows and rule (registry).
- Method 2: the sim itself (each tree's demand WASM, `MarketPacks` opened as checks/t1148 does, `corridor_trips` / 2)
  for each version. Both methods agree to the trip on every pair, before and after.
- Scripts: scratchpad `twstats/pairs.py`, `simpairs.mjs`, `pairs.txt`.

### 6-7. Airports across zone lines (T-2896)

An airport's passengers who live in a neighbouring zone now come from there; hub airports also count only their
non-connecting passengers (Istanbul 0.5, Munich 0.55). Trips a day to and from the airport (2 x the packs' outings).

| Airport | Before | After: in its own zone + from other zones | x |
|---|---|---|---|
| Istanbul (zone Arnavutkoy) | 8,616 | 4,308 + 106,504 = 110,812 | 12.9 |
| Munich (zone Rosenheim) | 12,634 | 6,948 + 52,242 = 59,190 | 4.7 |
| Ben Gurion, Tel Aviv (zone Ramla) | 4,140 | 4,972 + 44,474 = 49,446 | 11.9 |
| Dusseldorf | 12,434 | 19,734 + 43,608 = 63,342 | 5.1 |

- Read 1: the pack headers' `airport.outings` (old from R2, new from the freeze) and the new header's
  `airport.sites[].outside_outings` (planned for other zones: Istanbul 55,535, Munich 27,758 outings).
- Read 2: the airport rows actually placed in the neighbouring packs' market files (everyday kind 4 toward the
  airport's place, after the 40-80 km taper): Istanbul 53,252 outings (52,804 from the Istanbul zone), Munich
  26,121 (22,821 from the Munich zone). The table uses read 2. Published: Istanbul 84.5M a year (231k a day; at 0.5
  connecting about 116k), Munich 41.6M (2024; 114k a day, at 0.55 about 63k): both still under, as her rule asks.
- Scripts: `twstats/airscan.py`, `aircross.py`, `everyday.py`.

### 8. Universities (T-3214, T-3273, T-3218)

Vancouver pack (`van`), university students by cell, before (R2) and after (freeze), read with
pipeline/read_pack.py. The pack's total is the same both times (116,893): what changed is where they sit.

| Within this distance of the UBC bus exchange | 400 m | 800 m | 2 km |
|---|---|---|---|
| Before | 121 | 409 | 933 |
| After | 9,121 | 29,841 | 45,555 |

- Second read: T-3214's own proof run (a separate pipeline run, notes/T-3214.md) had 409 within 800 m before (the
  same) and 26,040 after the spread alone; the light pass then added the parent-count split, which is why the
  shipped figure is higher. "About 30,000" holds.
- Attendance: `everyday::RATE[3]` is 0.6 in the v0.3.2 tree and 0.8 on master (the old van header also says 0.6).
- Script: `twstats/campus.py`.

## Didn't hold up, or not worth posting

- **London Luton**: the header plans 21,911 outings from other zones, but only 2,754 survive the 40-80 km taper
  (central London is 40-80 km away), so Luton is 2.6x, not 11x. Skipped.
- **Kusatsu**: the replica says 3.6x (384 -> 1,388), my check says about 4.1x. Too loose to post a number.
- **Brussels Airport** 29x: mostly a new passenger figure (in its own zone 1,230 -> 20,665 outings), not the
  zone-line change. Skipped.
- **Narita, Haneda, Atlanta**: unchanged (already inside their zone).
- **Moscow - Sochi** unchanged: Russian (and Turkish, US, Japanese, British, Korean) travellers get no pull.
- **Trips abroad to beaches** barely move (Berlin - Mallorca 1.13x, London - Malaga 1.15x, Berlin - Antalya 0.97x):
  the pull across a border is small.
- **Under 300 km** nothing gains (Barcelona - Mallorca 0.89x, Lisbon - Algarve 0.96x, Lyon - Nice 0.92x): the pull
  starts at 300 km, and departures being kept moves a few trips farther.
- No university gets "x times more trips" in total: Vancouver's student count is unchanged, and the parent-count
  fix (T-3273) mostly cut overcounted campuses (Texas A&M Galveston 94,221 -> 0). The UBC line is about where the
  students are.

## Worth a screenshot

1. **Hakone, Tokyo game**: Destinations bubbles (D) over Odawara and Hakone-Yumoto at a zoom that shows Odawara
   and the lake; or the trip explorer from a Shinjuku zone to Hakone-Yumoto, which shows the weekday trips.
2. **Munich Airport, Munich game**: Destinations bubbles north-east of the city at the airport, which sits outside
   the Munich zone. Istanbul Airport in an Istanbul game works the same way and is bigger. (I have not opened
   either in the game; the figures are from the data files.)
3. **UBC, Vancouver game**: Destinations bubbles on the Point Grey campus, now spread over the campus around the bus
   exchange in place of one bubble at the south end.
