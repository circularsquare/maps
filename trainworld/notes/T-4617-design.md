# T-4617: countries in the Cities tab, with border multipliers (design)

Design only; nothing is built. Agent borders, 2026-11-28.

Her ask, 11-28: "could we also build a new feature for seeing / adjusting cross border damping multipliers?
accessible via the cities panel. i guess lets also add countries to the city panel. there should be a toggle button
on the topleft that lets you switch between showing cities and countries, kinda like the 'show stations separately'
button. countries should show most of the stats that we show for cities, just aggregated. but yeah we should also
show the top 10 or so touching / closest to touching countries, with an option to expand to see more countries, and
each of their border demand mults with the selected country. in sandbox mode we should be able to adjust these. on
hovering the demand mult we can say like 'multiplier applied to international travel with [country]. adjustable in
sandbox mode' (or if in sandbox mode, we just show it with + - buttons and just say 'multiplier applied to
international travel with [country]')"

Earlier asks: T-3857 (P4, her TODO 11-20 from players: "sandbox demand scaling, sandbox border damping controls");
this design is its border half. Nothing in notes/repeats.md or ARCHIVE.md. Related open border work: T-2984 (named
pack pairs), T-4053 (a common language factor), T-810 and T-429 (EU commutes and long trips too low), T-2578
(Mexico-US), T-2713 (twin towns), T-3489 (commutes by direction).

## 1. What she sees

**The toggle.** Cities tab, top left, right of the network diagram button: a flag icon (`Icon.flag`), lit while on,
as "Show stations separately" is. Hover "Show countries". Off at each load, as Show stations separately.

**The countries list.** The cities list's columns and sort headers (Name, Train share, Trips, Profit, Points). First
row "All countries": the All cities sums, and a click opens the all cities view. Then one row per country with a
built stop, its figures summed over those cities the way the All cities row sums every city, so the rows add up to
the first row. Hovers as a city row's, with the country's name (the name's: "France", then "4 cities, 212
stations"). A click opens the country view, a double click also moves the camera to its built stations. Sorted by
trips, as the cities. Narrow docks hide columns as the cities list does.

**The country view** (the inspector; beside the tab on a phone, as the city view):

```
France                                            [go to]
Train share        [===|=====|==========]
Cities             4
Stations           212
Lines              31
Train trips        1.2M/day
Points             4.56M
Profit             $389M a day
By purpose         [bar]
Top lines          ...                    boardings today
Top stations       ...
Border multipliers
  Germany          396k/day      0.22x
  Belgium          ...           ...
  (10 rows)
  Show all countries
```

In a sandbox the multiplier is a stepper, `[-] 0.22x [+]`; a click on the value lets her type one, and a changed
pair also has the back icon (hover "Back to 0.22x"), as Advanced fares' country rows do. A row's name opens that
country's view. A country with no built stop (opened from such a row) shows its title and Border multipliers only.

Hovers: the trips figure "396,431 trips per day between France and Germany" (following the day shown, as the Cities
tab does); the multiplier in a normal game "Multiplier applied to international travel with Germany", then on a
second line "Adjustable in sandbox mode"; in a sandbox the first line alone.

## 2. Border damping today

Three factors, and none of them is one number per country pair:

| Trips | Factor | Where it lives |
|---|---|---|
| Long trips (25 km and up, the market) | 1 at home, 0.5 between two countries of the open-border area, else 0.26 x (origin place's income / US)^1.5, so it differs by direction and by place | sim `intercity.rs` `border_factor` into `PairRule.b`, applied in the game for pairs 250 km and over; pairs under 250 km come from the rows (`near`), which carry it from `pipeline/market_zones.py`, as the normalisers do; the corridor file (T-4199); the app's copy `game/marketRule.ts` `border` |
| Commutes between zones | `pipeline/borders.json` `national` (14 pairs), else 0.05; 0 for a closed border | `joint_gravity`, offline; fixed lists in the market packs, and per place pair in the rows' `commute` |
| Everyday outings and airport trips across a zone line | the commutes' national factor | `cross_everyday`, offline; fixed lists in the market packs |

No pack holds two countries, so the local layer meets no national border.

Today's figures from `app/src/game/market-rows.json` (a weekday, both ways; commutes as two trips a commuter;
"shown" is section 3's number):

| Pair | Long trips | Long factor | Commute trips | Commute factor | Shown |
|---|---|---|---|---|---|
| France-Germany | 341,000 | 0.5 | 55,000 | 0.05 | 0.22 |
| Spain-Portugal | 112,000 | 0.5 | 15,000 | 0.05 | 0.24 |
| Germany-Poland | 167,000 | 0.5 | 45,000 | 0.05 | 0.17 |
| US-Canada | 329,000 | 0.20 | 104,000 | 0.05 | 0.12 |
| US-Mexico | 155,000 | 0.068 | 313,000 | 0.05 | 0.055 |
| China-Hong Kong | 852,000 | 0.18 | 224,000 | 0.05 | 0.12 |
| Malaysia-Singapore | 160,000 | 0.22 | 793,000 | 0.5 | 0.41 |
| Japan-Korea | 171,000 | 0.14 | 300 | 0.05 | 0.14 |
| China-Vietnam | 59,000 | 0.026 | 143,000 | 0.05 | 0.039 |
| India-Bangladesh | 88,000 | 0.010 | 1,523,000 | 0.05 | 0.041 |
| Brazil-Argentina | 14,000 | 0.05 | 44,000 | 0.05 | 0.05 |
| Israel-Palestine | 9,000 | 0.011 | 15,000 | 0.01 | 0.01 |
| South Korea-North Korea | 5,000 | 0.006 | 0 | 0 (closed) | 0.006 |
| India-Pakistan | 24,000 | 0.006 | 0 | 0 (closed) | 0.006 |
| Russia-Ukraine | 18,000 | 0.044 | 0 | 0 (closed) | 0.044 |

The long factor here is the share of the no-border long trips that happen, weighed by trips; near pairs split by the
rule's own direction ratio, so it is an estimate.

**Found on the way: closed borders are open to long trips.** `borders.json`'s zeros (the Koreas, India-Pakistan,
Russia and Belarus with Ukraine, Armenia-Azerbaijan, Armenia-Turkey, Azerbaijan with Georgia and Russia,
Algeria-Morocco) reach commutes and everyday outings only. `border_factor` and market_zones.py never read them, so
long trips cross at the income rule: Seoul-Pyongyang 213 a day each way, Amritsar-Lahore 223, India-Pakistan about
24,000 a day in all, Russia-Ukraine 18,000. docs/zones.md calls these borders closed. Call 2.

## 3. The number shown

One number per pair, the same from either side: the share of the trips there would be with no border that still
happen, long trips and commutes together, both ways. Worked out in the page from the rows: long trips with and without
their factor per place pair, commutes divided by their factor. Everyday outings across are not in the rows; they move
with the number but are not in it. Shown as fare multiples are ("0.22x"), two significant figures under 0.1
("0.055x"). The trips beside it are the same rows' long trips and commute trips, both ways.

## 4. A change in a sandbox

- **Saved** in the header as `borders`, `{"DE|FR": 0.5}` (alphabetical codes, as borders.json), only pairs she
  changed, each the value she chose. A new top-level header field, so an older build refuses the save as newer
  (`headerUnread`) instead of dropping it; no save level change. A data rebuild that moves a pair's own number keeps
  her value.
- **Applied** as the ratio of her value to the pair's own, to every trip across that border: long trips both ways
  (`PairRule.b` and the near rows), commutes and everyday outings (their lists, as their pairs are built). Each factor
  stays at most 1, and 1 means no border at all (every factor 1). A closed border scales the ordinary rule instead
  (0.05 for commutes, the income rule for long trips).
- **Steps**: 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7 and 1, plus the pair's own number; any value from 0 to 1
  can be typed. A press is an edit as a fare change is: one Ctrl+Z, saved with the game. Only a sandbox; a normal
  game made a sandbox gets the steppers then.
- **What it gets wrong** (for DECISIONS when built): the normalisers stay, so opening a border adds trips abroad and
  takes none from trips at home; commutes are not balanced again, so more commuters across take none from local
  commutes; commute and outing pairs the offline solve dropped (under 1 a day) stay missing, and a closed border opened
  has long trips but no commuters, since none were ever solved; where a factor reaches 1 before the others, the true
  mix sits a little under the value shown.

## 5. What a change costs

- No offline solve runs again; everything happens in the game.
- Each demand worker updates the pairs of the two countries and works out the corridors and their book again: at
  most about 5 s a worker on NYC's market (2.7 s to compose and 2.3 s for the book without the corridor file, against
  0.6 s with it; Node, notes/T-4199.md), less if only the two countries' pairs are redone. A save with changed borders
  would load that much slower and use about 45 MB more a worker (174 against 129 MB), unless stage 3 finds the file's
  corridors can be scaled in place (their trips are linear in the factor).
- Trips are calculated again only for a network that serves zones in both countries: the market makes pairs only
  between zones a station serves. A network inside one country calculates nothing; only the figures for trips no station
  carries move (To other places, Top destinations, the trips beside the multipliers), with no solve to wait for.
  Stage 1 lists which figures those are. A network across the border waits as for an edit there: NYC-size about a minute (loads 61-95 s on
  perf_bench), Belgrade's Balkan network about 13 minutes (784-817 s), her Asia save about 7.5 minutes to the final
  (455 s, notes/T-3917.md).
- The kept trips beside each slot and the corridor file take the table into their keys, so a stale result never
  shows.

## 6. Touching or closest to touching

From `packs/outlines.json` (14 MB, already loaded in the page for realms and shades): 4,880 zones in 136
countries, neighbours sharing exact vertices. A coastal outline takes 5 km of sea, so a strait under about 10 km counts
as touching (Oresund, Johor, Hong Kong-Shenzhen); wider ones (Gibraltar, the Channel) are near, not touching.

- Touching: a zone of each country shares an outline edge.
- Gap: the shortest distance between the two countries' outline vertices, on a grid, once per country opened.
- Order: touching countries first, most trips between them first, then the rest by gap. Ten shown; "Show all
  countries" lists every other country with zones.
- Worked out in the page rather than shipped as a table: it always matches the zones, and adds no step a join can
  forget (T-2112's seams.py).

## 7. Which city figures add up to a country

Add up cleanly (over the country's built cities, the All cities way): train share and train trips (`sumModes`),
profit (the region ledgers), points, stations, cities; lines with a stop there, counted once; By purpose, Top lines
and Top stations over its cities' stations.

Left out: Trips within and To other places (would need trips between its cities from the market, and would cover only
built cities), By trip length (a trip between two of its cities would count twice), Top destinations, Goals reached
(a star is a city's), the home icon (home is a zone or all cities). Call 4.

## 8. Calls for her

1. **What the number is.** (a) One number for all travel between the two, both ways (France-Germany 0.22x: long
   trips 0.5, commutes 0.05), and + and - move every kind together. (b) Long trips only (France-Germany 0.5x, US-Canada
   0.2x); commutes still move with it, unseen. (c) A plain 1x on top of today's rules, the same on every pair until
   changed, so a normal game shows nothing worth seeing. **I'd take (a).**
2. **Closed borders let long trips through** (section 2). Close them to long trips too: a separate small task, a
   demand rule change, before stage 4 so the panel reads 0x there. **I'd do it.**
3. **Range and steps**: 0 to 1, 1 meaning no border, the steps in section 4, typing for anything between. Nothing
   above 1. **I'd keep it there.**
4. **Country view**: leave out Trips within, To other places, By trip length, Top destinations and Goals reached for
   now. **I'd leave them out**; "Trips within France" and "To other countries" could be a later stage.
5. **Trips beside each multiplier** ("396k/day"), so a change shows its effect. **I'd show them.**

## 9. Player text

New:

| Where | Text |
|---|---|
| Toggle, hover and label | Show countries |
| Countries list, first row | All countries |
| Country view, heading | Border multipliers |
| Under the ten rows | Show all countries |
| Hover of a row's trips | {count} trips per day between {a} and {b} |
| Multiplier hover, normal game (her wording, two lines) | Multiplier applied to international travel with {country} / Adjustable in sandbox mode |
| Multiplier hover, sandbox (her wording) | Multiplier applied to international travel with {country} |
| Stepper label, read as "Less border multiplier" and "More border multiplier" | border multiplier |

Reused: "Go to {city}" (the go-to icon), "Back to {value}" (the back icon), the name hover (`cities.allTip`),
the split, trips and points hovers with the country's name (`cities.where`, `cities.tripsTip`,
`cities.pointsPast`), the city view's row labels (Train share, Cities, Stations, Lines, Train trips, Points, Profit,
By purpose, Top lines, Top stations, "boardings today").

Unchanged: the sandbox hover "Unlimited money"; the multipliers are found where they show.

Changelog, at the last stage: "added countries to the cities tab, with border multipliers you can change in sandbox".

## 10. Stages

One agent each. Estimates from `tools/agent_tokens.py stats` (median minutes: app logic 14, trip figures and sim 38,
all agents 23).

1. **Read-only plan** (Opus, about 200k, about 25 min). From the code, settle: every reader of trips across a border
   in sim and app (`PairRule.b`, `place_eachway`'s near rows, `apart`, selection.rs, market_trip.rs, the corridor
   file's corridors and sums, the commute and everyday pair builders, the airports' outside outings, marketRule.ts
   `border`, `placeTrips`, `rowsAnswer`, the zone hover's `marketKinds`); how the table reaches each worker (a
   Service field checked each solve, as fares are, or a market call such as `set_fare_factors`) and whether the
   shared `Market` can change in place or is composed again; where demandClient decides which clusters a change
   dirties (only those serving both countries); the corridor file, scaled or refused; the kept trips key. Time the
   adjacency over outlines.json in Node. Writes notes/T-4617.md with stages 2-4 file by file.
2. **Countries in the Cities tab** (Sonnet, about 250k, about 20 min; can run beside stage 3). The toggle, the
   countries list and the country view without its border section. Budget: UI, typecheck and 1 screenshot.
3. **The table in the demand WASM and workers** (Opus, about 400k, about 45 min; after call 2's task if she says
   yes). Rust tests: no table gives today's bits, a pair's trips scale by the ratio, 1 equals no border, a closed pair
   opens on the ordinary rule. Budget: trip figures, 1 figure diff on the real NYC save, which must not move.
4. **Border multipliers** (Opus, or Sonnet if stage 1's plan leaves no open questions; about 350k, about 30 min).
   Adjacency, the shown numbers and trips from the rows, the sandbox stepper and back icon, the header field, one
   undo, the kept trips key; the docs below, DECISIONS (section 4's lies), the changelog item. Budget: app logic,
   typecheck and 1 game run: a sandbox game near a border, change one, see the trips move, reload, Ctrl+Z.

About two hours of agent time in all.

## 11. Docs to change when built

docs/ui-panels.md "Cities tab" and "A new game" (its sentence that a sandbox works demand out as a normal game);
docs/ui-inspectors.md, a new "The country view"; docs/regional.md 5.1 and docs/zones.md 4 and 4a (the sandbox
ratio); SPEC.md 11 and docs/architecture.md 11 (the header's `borders`, the cache key); wiki/between-cities.md (a line
on sandbox multipliers); docs/zones.md's closed borders if call 2 lands.
