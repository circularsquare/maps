# Real rail ridership worldwide, as a check on demand (T-2086, 2026-11-03)

A table of published passenger figures for 165 rail systems in 110 cities and 49 countries
(`research/ridership.csv`), with three ratios to hold the game against: riders a weekday per km of
line, per station, and rail trips per resident. It is a loose check that demand looks believable in
every kind of city, not a calibration target (CLAUDE.md "Working here": demand is simple on purpose,
real cities are loose checks).

The seven real networks already have a careful one-off comparison, `notes/T-1742.md` (game rail trips
a weekday against published entries: London 0.96, Paris 1.18, Toronto 1.0, Sendai 0.78, Seoul 1.53,
New York 2.4-2.7, Tokyo 0.4-0.5). For those seven, T-1742's bases (station entries, gate entries,
validations, inside each pack) are the better comparator than the sums here. This table adds the
rest of the world, so the same kind of check can be run where there is no real save yet, and gives
the bands by region that a standing check would need.

## The table

One row per system; metro, commuter or suburban rail, light rail and trams as separate rows where a
figure exists. Columns:

- `annual_m`, `year`: passengers a year in millions and the year (or fiscal year) they are for.
- `count_basis`: what one passenger is. **entries** (or validations, or journeys): counted once on
  entering the system, changes between its lines free (MTA, TfL, IDFM, most metros). **unlinked**:
  each boarding counted, a change counts again (NTD and APTA in North America, China's CAMET
  "keyunliang", Delhi Metro's ridership, Seoul Metro's figure). **linked**: whole trips by the mode
  (Singapore LTA, Sydney Opal, ORR). Unlinked runs about 1.3-1.9 times entries (New York's subway:
  2,207M unlinked in NTD against the MTA's 1,195M entries for 2024). The game's `railTripsPerDay`
  counts a journey once, so it sets against entries.
- `avg_day_k`: annual / 365, thousands.
- `weekday_k`, `weekday_basis`: an average weekday where the source gives one; otherwise estimated
  as annual / 300 (the Americas, Europe, Africa, Oceania) or annual / 340 (Asia), marked
  "estimated". The factors come from New York's subway (320), London's Underground (about 300) and
  Asian systems' smaller weekend drop. Some rows only have a daily figure from a list, with the
  year unstated; those say "daily as listed".
- `length_km`, `stations`, `riders_per_km_weekday`, `riders_per_station_weekday`.
- `metro_pop_m`, `pop_basis`: **demographia** is the built-up urban area from Demographia World
  Urban Areas, 20th edition, August 2025 (http://www.demographia.com/db-worldua.pdf), one definition
  everywhere. It joins some neighbours (Washington with Baltimore, Boston with Providence, Guangzhou
  with Shenzhen at 69.6M, Shanghai with Suzhou at 45M), so Beijing, Shanghai, Guangzhou, Shenzhen
  and Chengdu use their municipal resident counts for 2024 instead (**municipal**: 21.8M, 24.8M,
  19.0M, 18.0M, 21.5M; checked against the cities' statistics bureaus as reported, Shanghai's
  given as 24.80M or 24.87M). Chengdu's municipality is much larger than its city.
- `source`, `source_kind`: **primary** where the figure came from the operator, a statistics office
  or a national dataset; **via list** where it came from a Wikipedia list that cites one (as the
  brief allowed, a pointer; each list cites its own source row by row). 53 rows are primary.

### Where the figures came from

- **US**: FTA's National Transit Database, 2024 calendar year, summed by agency and mode through the
  open API (data.transportation.gov `5ti2-5uiv` for metro and light rail, `8bui-9xvu` for commuter
  rail). APTA's own report refused the download (403); its weekday figures came through Wikipedia's
  lists of North American rapid transit and light rail by ridership (APTA, Q1 or Q2 2026 weekdays, a
  year after NTD's). New York's subway is the MTA's 2024 figure (1,194.9M entries, 3.736M an
  average weekday).
- **Canada, Mexico**: APTA (Canada) and INEGI (Mexico) through the same lists.
- **London**: TfL quarterly performance report Q4 2024/25 (Underground 1,216M, Elizabeth line
  231M, Overground with DLR and trams 299M); ORR for National Rail (about 300M a quarter in the
  London and South East sector, which includes the Elizabeth line and the Overground, so about 780M
  a year for the rest; that sector runs far past London).
- **Paris**: IDFM via INSEE, 2024 (Metro 1,476M, RER 1,037M, Transilien 328M, RATP trams 321M).
- **Japan**: MLIT's station passenger file S12 (FY2019, the copy in `riders/japanriders/data/S12-22/`),
  boardings plus alightings halved and summed within 50 km of Tokyo station, 40 km of Osaka, 30 km
  of Nagoya and 20 km of Fukuoka, Sapporo, Sendai and Hiroshima, split into JR, municipal subways and
  private railways. Each operator counts its own passengers, so a change between operators counts
  twice; FY2019 is before telework (2024 runs perhaps 5-10% below). T-1742's 42.7M for Tokyo is the
  same file.
- **Seoul**: Seoul Metro's 2024 statistics as reported (2,417.5M on Lines 1-8, 6.605M an average
  day); Korail's metropolitan lines and AREX from the list.
- **China**: the China Association of Metros (CAMET) 2024 statistics report. Its PDF
  (infosharingp2-oss.camet.org.cn) did not answer from here (connection reset), so the per-city
  figures are the Wikipedia list's, which cite it.
- **Hong Kong**: MTR's 2025 annual report (domestic weekday 4.71M, up 0.6%, so about 4.68M in 2024).
- **Singapore**: LTA's yearly ridership table (MRT 3.412M, LRT 0.210M an average day in 2024).
- **Jakarta**: BPS via GoodStats (328.15M on Jabodetabek rail in 2024) and KAI Commuter's 973k a
  weekday in Q3 2024.
- **Mumbai**: Central and Western Railway's 2025-26 average days as reported (3.87M and 2.95M).
- **Sydney**: TfNSW Opal trips FY2024-25 through Wikipedia's patronage template.
- Everything else: Wikipedia's "List of metro systems" and "List of suburban and commuter rail
  systems", latest year shown on each.

Not used, and why: APTA's PDF (403), the CAMET PDF (unreachable), the Tokyo person trip survey's
mode split (its results pages did not give totals; T-1742 used MLIT's 2015 census, about 34M rail
trips a day for the whole Tokyo area). No sign-ups, nothing behind a login.

## Caveats

- **Bases differ.** Read entries, unlinked and linked rows apart; a city's sum mixes them where its
  operators count differently (New York's subway is entries, its railroads NTD unlinked).
- **Years differ**, 2019 to 2025; several commuter figures from the lists have no year at all and
  some look old (Kolkata's 6.1M a day, Cape Town's 0.5M, Buenos Aires 2015).
- **The population is the denominator that moves most.** A built-up area can be smaller than what
  a commuter network serves (Zurich's S-Bahn against 0.92M, Munich's S-Bahn) or much larger than the
  city its metro serves (Bangkok 20.3M, Jakarta 36.9M, Manila 25.5M). Per-resident figures are good
  to about a factor of 1.5.
- **Coverage per city is uneven.** Where only the metro row exists (Rome, Lisbon, Istanbul without
  Marmaray, Delhi without its suburban trains) the city's per-resident figure is low.

## The ratios

Weekday riders per km of line, metro rows (median, with the range):

| Region | Metro rows | Per km | Per station |
|---|---|---|---|
| North America | 14 | 3,200 (1,200-15,400) | 5,600 (1,300-16,200) |
| Europe (with Turkey, Russia) | 25 | 11,900 (7,800-32,500) | 13,000 (7,300-34,700) |
| East Asia | 20 | 8,700 (2,600-33,600) | 17,000 (2,800-47,300) |
| Southeast Asia | 9 | 8,500 (2,800-22,500) | 10,300 (3,000-29,200) |
| South Asia | 7 | 8,500 (4,800-18,400) | 9,900 (6,500-28,500) |
| Latin America | 9 | 14,300 (9,300-36,100) | 16,900 (10,100-40,900) |
| Middle East | 4 | 11,600 (2,800-18,300) | 19,600 (5,900-29,800) |
| Africa | 3 | 8,300 (900-45,600) | 8,100 (2,300-57,900) |

Commuter rail per km is an order of magnitude lower: North America 20-410 (median 180), Europe
570-6,600 (median 1,800), Seoul's Korail lines 5,400, Mumbai's suburban 14,700 (the world's most
crowded), Jakarta's KRL 3,300.

Some single values worth having to hand: Tokyo's subways 33,600 a km (FY2019), Hong Kong's MTR
22,400, Paris Metro 20,000, Mexico City 18,500, Delhi 18,400, Toronto's subway 15,400, Singapore
14,900, Shanghai 12,500 and Beijing 11,700 (both unlinked), London Underground 10,000, New York's
subway 9,400, Chicago's L 2,300, Houston's light rail 1,000, Dallas's 340.

**Rail trips per resident a weekday**, all of a city's rows summed (median, with the range):

| Region | Cities | Per resident |
|---|---|---|
| East Asia | 25 | 0.41 (0.02-1.13) |
| Europe | 26 | 0.46 (0.09-1.05) |
| Latin America | 10 | 0.17 (0.05-0.31) |
| Middle East | 4 | 0.19 (0.06-0.24) |
| South Asia | 7 | 0.15 (0.01-0.33) |
| Oceania | 4 | 0.10 (0.05-0.25) |
| North America | 21 | 0.06 (0.006-0.27) |
| Southeast Asia | 7 | 0.04 (0.004-0.64) |
| Africa | 5 | 0.04 (0.001-0.22) |

The top: Tokyo 1.13 (operator boardings, so about 0.9 on journeys), Munich 1.05, Prague 1.01, Paris
0.93, Stockholm 0.84, Vienna 0.79, Berlin 0.77, London 0.74, Osaka 0.73, Hong Kong 0.66,
Singapore 0.64. The big Chinese cities sit at 0.45-0.51 on unlinked counts (about 0.3 on entries),
Seoul 0.47 (unlinked), New York 0.23, Toronto 0.26, Montreal 0.27, Mexico City 0.21, Delhi 0.21,
Mumbai 0.28. Car-centred US metros: Los Angeles 0.015, Dallas 0.009, Miami 0.008, Phoenix 0.007,
Houston 0.006. Cities whose networks are new or tiny for their size: Jakarta 0.03, Manila 0.04,
Dhaka 0.012, Ho Chi Minh City 0.004, Lagos 0.001.

## Against the game

T-1742's finding carries over and the table makes its shape clearer. The game's rail trips per
resident on a real network come out in a narrow band (0.41 Tokyo, 0.50 Seoul, 0.56 New York, 0.68
London, 0.73 Paris; 0.2 for the small networks), while real cities run from 0.006 to over 1. The
game has no term for how much a place uses rail, on purpose, so the network and the global mode
rules set the volume. That makes the game high where real rail use is low for the network's size
and low where it is highest.

Cities with real networks in `data/packs/real/` to compare directly: New York (`nyc`), London
(`lon`), Paris (`par`), Tokyo (`tyo`), Seoul (`seo`), Toronto (`tor`), Sendai (`sdj`). `realnet.py`'s
`jpn7` network (all of Japan's lines from N02, cut to the seven Japanese city packs) also covers
Osaka, Nagoya, Fukuoka, Sapporo and Hiroshima, whose S12 sums are in this table. Every other row
needs a real save first; `realnet.py`'s GTFS adapter would do most North American, European and
Australian cities, whose operators publish GTFS.

## A standing check (proposed as T-2088)

Run each real save through the weekday solve and set its rail trips per resident and per km of line
against this table's band for its kind of city, flagging a city only when it falls outside about
0.5-2 times the published figure; it is a band for "would a player see something obviously wrong",
never a target to fit.

In more detail:

1. A small script reads a real save's solved weekday (as `notes/T-1742/run_real.mjs` already dumps it:
   `railTripsPerDay`, line riders, line km, pack population) and prints, per city, rail trips per
   resident and riders a weekday per km of metro and per km of commuter line, next to this table's
   row (on the matching basis: entries against `railTripsPerDay`).
2. The band is wide on purpose: within 0.5-2 times published is fine; outside it is a note, not a
   failure, unless a player would see it (a commuter line busier than Mumbai's, a US light rail line
   at Tokyo subway loads).
3. Grow the set of real saves for the cities where the model is most likely off (below), a few at a
   time: Chicago or Houston for a car-centred US metro, Hong Kong or Osaka for a very dense one,
   Mexico City or Mumbai for a big city with heavy informal transport, Madrid for a European metro
   outside the seven.
4. Rerun after any demand change that moves rail volume, beside the T-636 ceilings.

## Where the simple model should be furthest off

- **Car-centred US metros (Houston, Dallas, Phoenix, Atlanta, Los Angeles): game high.** Real use
  is 0.006-0.025 trips a resident and 300-1,600 riders a km on light rail. New York is already 2.5
  times its published figure in the game, and New York is the US's densest. In a sprawling metro the
  game's car is priced well (free parking in suburbs), but people there also own cars outright, live
  far from stations and have schedules built around driving; the model sees only time and money
  per trip, and the same rail share rules as London. A player's line in Houston could carry several
  times what METRORail does. Worth a real save to measure, since US cities are where many players will
  build.
- **Very dense Asian megacities (Tokyo, Osaka, Hong Kong, Seoul's core, Shanghai): game low.** Real
  use is 0.5-1.1 a resident and 20,000-34,000 riders a km; the game's Tokyo is 0.4-0.5 of published
  (T-1742). Long Japanese commuter trips do not show, rail gets no weight for being the majority
  habit, and the cap at the trains' room (crowding up to 5.4 at 120% full) may stop the game reaching
  real Tokyo or Mumbai loads at all: Mumbai's suburban trains run far past rated capacity
  (14,700 riders a km a day, 44,000 a station).
- **Cities with heavy informal or bus transport (Lagos, Dhaka, Jakarta, Manila, Ho Chi Minh City,
  Cairo, Mexico City): either way, mostly game high on new lines.** Real metros there are young and
  short, and many ramped up slowly (Jakarta MRT 134k a weekday on 15.7 km; Ho Chi Minh City's line
  1 about 60k; Lagos's Blue Line about 10k). Minibuses, motorbike taxis and walking carry most
  trips; the game's "car" stands for driving, taxis and longer bus trips, priced like a private
  car, and fares scale with income. On the other hand where an old, cheap network exists (Cairo,
  Mexico City, Mumbai, Sao Paulo) real loads are among the world's highest and the game may run low
  for the capacity reason above.
- **Small European cities with strong S-Bahn and metro (Munich, Prague, Zurich, Vienna): probably
  close.** These are near London and Paris on per-resident use, where the game already sits near 1.

## Next

- T-2088 (filed): the standing check above, starting with a script over the seven saves and one new
  car-centred US save.
- The CAMET PDF and APTA's report could replace the list-sourced rows for China and the US weekday
  figures when either answers; neither needs an account.
- A JR East or Tokyo person trip journey count would firm up Tokyo's basis (T-1742 used 34M from
  MLIT's 2015 census).
