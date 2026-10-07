# Lines in pieces: the worklist (started 2026-10-05, maps-33)

Anita, 2026-10-05: "in general we should try to eliminate lines that are in multiple
disconnected segments i think. unless we like specifically think that they reflect reality.
but we should maybe make a list and go one by one and try to find why they are multiple
segments."

`python tools/pieces_report.py` is the list (reads dist/data only; `--list [cc ...]` gives every
line with its pieces, the gap between them and a first guess). A line is joined across
countries by id first, as the app does, so a line whose pieces meet at a shared border point
counts as one.

## Where it stood on 2026-10-05 (after the RINF pieces fix, before the fixes below)

| kind | in pieces | km outside the biggest piece |
|---|---|---|
| register lines | 323 | 9,418 |
| OSM lines (metros, trams, operating patterns) | 259 | 3,021 |
| named trains | 32 | 12,282 |

By first guess (`pieces_report.GUESSES`):

| kind | crossing | near (< 1 km) | hole (1-25 km) | far (> 25 km) |
|---|---|---|---|---|
| register | - | 57 (478 km) | 203 (5,022 km) | 63 (3,919 km) |
| OSM line | 94 (1,852 km) | 107 (374 km) | 54 (585 km) | 4 (210 km) |
| named | 31 (12,152 km) | 1 | - | - |

The worst register countries: de 44, ru 36, ua 29, au 56, in 19, cz 18, fr 15, at 11 (10
after the Kamptalbahn fix), it 11, pl 10. pl has 53 OSM lines in pieces (trams).

## Fixed on 2026-10-05 (trialled on every country it can touch, then rebuilt)

- **Transit sections** (`build_model.transit_tail`): a route with no stop in a country but
  whose track there runs between two of that country's border points gets one section from
  border to border. Eurostar Amsterdam - London had no French stop in OSM, so France built
  nothing of it and the merged "Eurostar" line had a hole from the tunnel to Belgium. Also
  the railjets Wien - Zürich / Bregenz across Germany (Salzburg - Kufstein, 112 km), four
  Russian trains across Kazakhstan (66 km), four Kazakh trains across Kyrgyzstan (15 km).
- **fill_holes retried pairs** (`rinf.fill_holes`): a pair that passed every check but lost to
  a shorter fill in the same round was never asked again (Austria's Kamptalbahn, Stiefern -
  Schönberg am Kamp). Austria: 113 fills (was 108); Kamptalbahn, Lindau - Innsbruck,
  Salzkammergutbahn and Wien Meidling - Linz each one piece fewer.
- **Out-and-back spurs** (`build_model.fold_spurs`): a line's two sections at a middle
  junction that leave it along the same rails become one section with the spur cut out; where
  the line already has a direct section between the same two stations of the same length, the
  detour goes instead (most of gb's: the bridged "borrowed" sections ran up a spur and back
  beside the line's own track). A switchback, whose legs leave the fork in a V, stays
  (Czechia's Chodov-úvrať). 74 spurs found (gb 56); 16 left, 6 km, all under 0.8 km. gb 41
  lines shorter and closer to their published lengths (ECML 642 -> 630 km against 633; Durham
  Coast 109 -> 94), tr 4 (Menemen - Aliağa 39 -> 25 km), ru 1. 48 junction ids gone, no
  stops. These were the UK's "branches that lead to no station" (South Wales Main Line up
  the Westerleigh curve to "Junction near Yate" and back).
- **App: continuation track on the map** (dist/index.html `paintSelection`): the track from a
  junction end to a stop past it was cut from the wrong half of the section when the
  section's points run against its key (Hell Gate Line -> New York Penn Station drew the
  half under the Hell Gate Line itself).

## The order to go through the rest

1. **crossing** (125 lines, nearly all named trains and OSM lines). Usually one cause shared
   by many lines: a crossing with no shared border point (`borders.EXTRA`), or a country in
   between not built. Known ones: Liechtenstein (railjet Wien - Zürich and both EuroNights
   via Buchs, 12.5 km gap: build `li`, which transit_tail would then cross);
   Iletsk on Russia - Kazakhstan (the five Tashkent / Kazakh trains 4.4 km off; HANDOFF
   thread 0's ex-USSR list); Dublin - Belfast (Enterprise, 13.4 km off: no RINF point at
   Newry / Dundalk); Kaliningrad trains through Lithuania (not transit: they stop in
   Lithuania; to look at); Київ-Експрес (pl+ua, 65 km off at Yahodyn).
2. **near** (164 lines). Likely a few systematic causes (a junction 100-500 m from where
   another line's section ends; a station split in two). Sample ten, find the causes, fix in
   code, re-run the report.
3. **hole** (257 lines). Per country: fill (RINF countries with `fill_holes`), bridge
   (`pieces.bridge_gaps`, as gb, kr, cn, tr do; rinf.py's `bridge_pieces` is written and
   untried), or track missing from OSM (leave, note it).
4. **far** (67 lines). **Done 2026-10-05**, Anita: "ok we can split lines". Passenger
   trains run on parts of a register line and the middle is closed or freight (fr: Chartres
   - Bordeaux in four pieces up to 229 km apart). `build_model.split_far_pieces`: a register
   line's piece over FAR_PIECE_KM (25) from its biggest piece, with two stops or more,
   becomes a line of its own, named in name_en by its end stops; the register's KEEP_WHOLE
   is honoured (cn's 青荣城际线). 54 lines made 75 more in 15 countries (au 17, fr 8, kz 6,
   de 6, pl 5, jp 4, in 4...). Aliases (`pieces`) carry saved rides. The 4 OSM lines that
   were "far" (RE8, TER 11, ARST, Tehran - Parand) are routes and stay as they are.

**Anita, 2026-10-05: stop here.** "lets not work through rest of list then. we can stop after
splitting far apart line pieces and wrap up." Items 1-3 above are parked, not started; pick
them up only if she asks again.

Each country's findings go in its `<cc>_sources.md`; shared fixes are trialled with ab.py on
every country before landing, as usual.
