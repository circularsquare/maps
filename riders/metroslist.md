# Metro Systems with Publicly Available OD or Near-OD Data

## Tier 1 — Real hourly-or-finer OD, openly downloadable

| City | Operator | Granularity | Coverage | Source |
|---|---|---|---|---|
| New York | MTA | Hourly (monthly avg) | Full system, ongoing 2020+ | https://data.ny.gov (search "Subway Origin-Destination Ridership") |
| San Francisco | BART | Hourly, actual measured | Full system, ongoing 2001+ | https://www.bart.gov/about/reports/ridership |
| London | TfL | 15-min bands | Full Underground, rolling survey | https://data.london.gov.uk (NUMBAT dataset, free registration) |
| Shanghai | Shanghai Metro | 10-min | 302 stations, May–Aug 2017 only | figshare DOI 10.6084/m9.figshare.28844942 |

## Tier 1.5 — Has OD but needs station-level disaggregation for hourly

| City | Operator | OD granularity | Station-level hourly? | Source |
|---|---|---|---|---|
| Seoul | Seoul Metro + Korail | Monthly | Yes (entry + exit split) | data.seoul.go.kr OA-20501 + hourly entry/exit dataset |
| Shenzhen | Shenzhen Metro | Varies (research release) | Varies | University of Leeds repository DOI 10.5518/599 |

## Tier 2 — Has OD, coarser time dimension

| City | Operator | Time granularity | Notes | Source |
|---|---|---|---|---|
| Singapore | LTA | Weekday vs weekend | Hourly station-level sold separately | https://datamall.lta.gov.sg |

## Tier 3 — Station-level only, no OD pairs

| City | Operator | Time granularity | Scope | Source |
|---|---|---|---|---|
| Paris | Île-de-France Mobilités | Hourly | One day: Nov 20, 2025 (expanding) | https://prim.iledefrance-mobilites.fr |
| Bengaluru | BMRCL | Hourly | Aug 2025 onward; no OD publicly | https://data.opencity.in |

## Dead Ends (no public OD data)

Hong Kong MTR, Tokyo (JR East / Tokyo Metro / Toei), Beijing Subway, Berlin BVG, Munich MVG, Madrid Metro, Mexico City Metro, Amtrak, SNCF passenger OD, Deutsche Bahn passenger OD.

## Notes for trip-allocation visualization

- All Tier 1 and 1.5 systems have GTFS feeds for schedule matching.
- Real-time GTFS-RT available for NYC, BART, TfL — usable for a day-of recording pass.
- Shanghai 2017 dataset predates Lines 14, 15, 17, 18 and major suburban extensions; inner city network mostly unchanged.
- Seoul disaggregation method: take monthly OD, divide by ~22 weekdays, redistribute across hours using that day's hourly entry/exit share at the origin station.
- BART is the only system with true measured OD (entry + exit gates). NYC infers via next-swipe. London, Shanghai, Seoul, Shenzhen, Singapore all have entry + exit gates and measured data.