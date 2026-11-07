# How to talk about the demand speedups online

Agent framing, 2026-11-07, for Anita's ask: "how do you think i should frame the optimization we've
done, that sounds impressive and is honest?" Read-only research; every number below has its task id
in the last section. The game is called trainsworld here, as in SPEC.md (players see that name).

## Candidate framings

Pick one, or mix. All of them are written to her text rules; every claim traces to the numbers at
the bottom.

**1. Tweet length**

> trainsworld works out where 13 billion trips a day go, in 65 countries, in a browser tab. the far
> end of a long trip gets blurrier the farther away it is. north america's long trips now solve
> 1.94x faster and every big line moves under 0.6%.

**2. Exact speedups (my pick)**

> most of the speedups in my train game change no number at all.
> 81% of japan's stations are stops on one line, so the route search walks them along the line and
> skips the queue. 1.56x faster and every figure is the same to the bit. the far-zone
> table got 3x faster the same way. the one shortcut that does round things (long trips get coarser
> with distance) keeps every line over 100,000 riders a day within 0.6%.

**3. For game devs**

> demand gets level of detail too.
> games draw far mountains with fewer triangles. my train game does that with trips: the zone you
> look at is solved fine, zones you haven't looked at in 10 minutes go coarse, and a long trip's far
> end is dealt in bigger chunks the farther it goes (up to 16x). it's rust in webassembly workers,
> so the map never waits on it.

**4. For transit nerds**

> under the cute trains it's a four-step model.
> gravity commutes from census flows where a country has them, a logit against the car and walking,
> and frequency-based route choice (spiess and florian's optimal strategies) with three rounds of
> crowding. riders split over the lines at a stop by how often trains come. it runs over 3,545 zones
> in your browser.

**5. The honest one**

> i asked if long trips could be computed more cheaply, since a long line doesn't care about the
> exact streets at the far end. we tried two shortcuts first: folding far stations into hubs, and a
> few sample zones per city. both put 20-200% errors on lines, so they're gone. the one we kept deals
> far trips over coarser chunks and gets 1.94x on north america with big lines within 0.6%.

**6. Longer, with the other games**

> subway builder gives you one city at a time, with millions of commuters from census data. nimby
> rails gives you the whole world map with individual passengers. i wanted both: census-style
> demand everywhere, in one save. so trainsworld counts trips per hour between places, about 13
> billion a day across 65 countries, and only redoes the parts a new line touches. new york alone is 68 million trip legs a day. the zone you're looking at updates in
> seconds; all of japan, 10,000 stations, takes minutes.

Notes on these:
- 1 and 6 lead with "13 billion": that is the registry's sum of local trip legs a weekday (it leaves
  out the longer trips between cities), so it is a fair "about 13 billion trips a day".
- 4's "census flows where a country has them" matters: many places are tier 0 from global sources.
- 6's "all of japan takes minutes" is honest and still impressive; the last measured whole solve was
  21 minutes on 2026-11-04, before this week's three speedups. Re-measure before quoting a number.
- If she wants a word for the "fractal" idea, "level of detail" (game devs know it) or "coarser with
  distance" (anyone gets it).

## What not to claim

- **"Fractal."** Nothing here repeats itself at every scale. It is detail that falls off with
  distance, in a few fixed steps: 2 km squares, 4 km blocks in thin country (T-852), coarse zones
  nobody is looking at (T-1223), far ends of long trips in chunks 1-16x (T-2641). The plain names are
  "level of detail" or "multi-scale". Transport planners do the same thing by hand: fine zones in the
  study area, coarse "external zones" outside it (Toronto's zone guidance, below).
- **"Simulates every passenger."** It doesn't. Passengers are flows, trips per period between places
  (SPEC 2). The trains move, the riders on them are counts. Subway Builder, NIMBY Rails and Cities:
  Skylines all have individual people you can follow; this can't show one person's day (the trip
  explorer shows one pair of places instead).
- **"Realistic" or "accurate".** The spec puts sensible and fun over fitting real cities, and most
  of the world is a low-fidelity first pass. Real networks are a loose check only. Say "from census
  data" where true, "sensible" otherwise.
- **"First" or "only".** NIMBY Rails already has the whole world. Planning software has done
  frequency-based assignment since 1989. What is unusual is the combination (whole-world demand of
  this kind in a browser game), and even that I can't prove nobody has done.
- **"Fast" for whole countries.** The zone in view is seconds; a whole big save is minutes (Japan 21
  min, NA about 2 min). Say which one.
- **"Exact" for T-2641.** It rounds. Lines over 100,000 riders a day within 0.54%, lines over 10,000
  within 1.9%. T-2652 and T-2653 are exact.
- **"AI"** in any sense, and "the whole world": Russia, Belarus and Ukraine are held, and much of
  Africa and Latin America is still being built.
- **Beating other games on performance.** The comparisons below are from store pages, dev blogs and
  forum posts, and nobody has run them side by side. Cities: Skylines II slowing past ~150k people is
  player reports. Fine to say what trainsworld does; not fine to say it's faster than X.

## The other games

| Game | Map | Who travels | How they pick a route |
|---|---|---|---|
| [Subway Builder](https://www.subwaybuilder.com/) ([wiki](https://en.wikipedia.org/wiki/Subway_Builder)) | 55 real cities, one per map (36 US, 6 Japan, 5 UK, 3 France and a few more), October 2026 | "millions of commuters" from US census (LEHD) and Redistricter, homes to jobs by a gravity model; students and airport trips too | per commuter group, "the same pathfinding algorithms you use on your phone"; wait, transfers, income, delays |
| [NIMBY Rails](https://store.steampowered.com/app/1134710/NIMBY_Rails/) ([dev blog](https://carloscarrasco.com/page/18/)) | the whole real world, detailed | individual passengers spawned by population density, destinations weighted the same way and by distance | multi-train trips with transfers; pathfinding in worker threads with a cache ([2021 post](https://carloscarrasco.com/page/53/)) |
| Cities: Skylines I and II ([PC Gamer](https://pcgamer.com/cities-skylines-2-has-no-cap-on-the-number-of-people-it-can-track-and-its-basically-the-matrix), [traffic AI](https://paradoxinteractive.com/games/cities-skylines-ii/features/traffic-ai)) | one city you build | individual citizens with homes, jobs and lives; CS1 capped at 65,536 agents, CS2 has no cap | each agent by path cost; players report CS2 slowing badly past 100k-200k residents |
| Transport Fever 2 ([forum](https://steamcommunity.com/app/1066780/discussions/0/2639605437882690967)) | generated maps | individual persons, destinations hard-coded | fastest route |
| [OpenTTD cargodist](https://wiki.openttd.org/en/Manual/Passenger%20and%20cargo%20distribution) | generated maps | passengers with destinations between stations (symmetric demand) | a flow solver over the network's link graph |
| Railway Empire 2 ([forum](https://steamcommunity.com/app/1644320/discussions/1/6407003396125537537)) | regional maps | passengers between city pairs by the two populations, capped to some pairs | trains between cities; transfers new in 2 |
| [Mini Metro](https://en.wikipedia.org/wiki/Mini_Metro_(video_game)) | abstract, generated | a shape that wants any station of that shape | any line that gets there |
| Planning tools ([Spiess and Florian 1989](https://ideas.repec.org/a/eee/transb/v23y1989i2p83-102.html)) | a study area, finer inside ([zone guidance](https://tmg.utoronto.ca/files/Reports/Traffic-Zone-Guidance_March-2021_Final.pdf)) | trip tables between zones | "optimal strategies": riders share lines by frequency; trainsworld's route search is this |

Where the others are better: Subway Builder's cities are hand-finished and its riders are people;
NIMBY Rails has years of track and signal simulation (trainsworld has no signals, SPEC 2.1) and is a
finished commercial game; Cities: Skylines simulates lives, not trips.

## The numbers the framings rest on

Scale (registry as committed, 2026-11-07): 3,545 zones (405 city packs, 3,140 countryside zones) in
65 countries; their local trip legs add up to 13.6 billion a weekday (New York 68.4M; T-2690 checked
NY's 68.28M against it). Market places: 6,757. Uncommitted rows from tonight's runs add 20 more.
Test networks: all Japan 10,320 stations and 709 lines; her North America save 1,860 stations, 134
lines, 153 market packs (notes/T-2621.md).

This week's three (2026-11-06):
- T-2641 coarser far ends: chunk grows 1-16x per 50 km; NA zone pairs halved, solve 1.94x (4.3 ->
  2.2 s, period 0); lines over 100,000 riders within 0.54% (usually 0.3%), over 10,000 within 1.9%;
  Japan 1.2x.
- T-2652 leaf stations: 81% of Japan's stations, 77% of NA's; Japan's searches 1.56x, queue pops
  271M -> 124M a pass, whole pass about 1.35x; every figure the same to the bit.
- T-2653 far-zone table by zone: Japan's longer-trip spreading 4.4 -> 1.45 s (3x), exact.
- T-2621 the design behind them: hubs and sample zones gave 20-200% line errors for 1.3-2x and were
  dropped; on NA 72% of the long-trip zone pairs are over 100 km apart.

Earlier, for the "exact" story and the history:
- T-005 (2026-10-08): first kernel, New York solve 0.8-0.9 s in WASM; cell pairs number in billions,
  so commuting works between 2 km zones instead.
- T-036: eight pairs at a time in SIMD lanes, identical to the bit; NY edit 22-25 -> 19-20 s.
- T-149: only the cities a network reaches are loaded, numbers bit-identical; T-325 bare zones,
  panning never re-solves, equal to the bit.
- T-874: tiny commute pairs dealt in chunks, NY pairs -49%, stations at most 0.1%.
- T-1222 / T-1223: coarse zones 25-35% less memory, rail +0.05%; all 723 Chinese zones load.
- T-1545: Japan's peak memory 2,694 -> 1,334 MB a worker, load 1,627 -> 1,107 s. T-1594: WASM memory
  grown 32 MB ahead, all-China 253-293 -> 97 s, figures identical.
- T-2386 (2026-11-04): Japan's whole solve 21 minutes. T-2596: NA whole solve 120-127 s.
  T-2605: the city in view 6.6-7.0 s on the NA save, an edit 3.1-3.3 s.
- Method (docs/demand-kernel.md): doubly constrained gravity between 2 km zones, logit mode choice
  against car and walk, frequency-based strategy search, at most three crowding rounds by successive
  averages, then riders held to the trains' room.
