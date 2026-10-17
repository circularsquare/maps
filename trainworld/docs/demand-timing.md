[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: time of day (4.2)

Periods, kinds of day and the average day.

### 4.2 Time of day

Five periods: morning peak 6-10, midday 10-16, evening peak 16-20, evening 20-24, night 0-6. They
are the player's scheduling periods too (section 6.3: high demand in the peaks, medium midday and
evening, low at night), defined once in `sim/src/track/params.rs` (`PERIOD_HOURS`,
`PERIOD_LEVEL`) for schedules and demand alike. The hours are local time (T-309,
docs/world.md "Local time"): each line's periods, boarding hours and kind of day are its first
stop's region's, so London's morning peak is London's 6-10. The solve itself has no clock, so
demand's period q is every region's own period q.

Each commuter makes a trip to work and a trip home a day. Of the day's trips to work, 70 / 12 / 6 /
4 / 8% start in the five periods; of the trips home, 2 / 18 / 58 / 16 / 6% (judgement for now). A
trip home goes back by the station pair the morning's trip used, so a period's rail trips are its
share of trips to work on the morning's station pairs plus its share of trips home on the same
pairs reversed. Crowding is judged at each period's busiest hour: 1.4 times the period's mean hour
in the peaks, 1.15 midday, 1.3 in the evening, 1.5 at night. Everyday outings share these combined-
load busiest-hour factors; their five-period departure shares are provisional:

| Purpose/direction | 6–10 | 10–16 | 16–20 | 20–24 | 0–6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Errands outward | 10% | 55% | 30% | 4% | 1% |
| Errands return | 6% | 49% | 36% | 8% | 1% |
| Social outward | 5% | 35% | 42% | 17% | 1% |
| Social return | 2% | 25% | 43% | 27% | 3% |
| School outward | 94% | 4% | 1.5% | 0.4% | 0.1% |
| School return | 1% | 82% | 15% | 1.8% | 0.2% |
| University outward | 65% | 27% | 7% | 0.9% | 0.1% |
| University return | 4% | 42% | 42% | 11% | 1% |
| To the airport | 30% | 35% | 22% | 6% | 7% |
| From the airport | 12% | 33% | 28% | 24% | 3% |
| Medical outward | 30% | 55% | 12% | 2% | 1% |
| Medical return | 8% | 60% | 25% | 5% | 2% |

The inspector uses actual purpose/mode/period selection results. Bars show departures per hour,
dividing period counts by 4/6/4/4/6 hours; outward and return rows remain separate.

**Kinds of day** (T-260, notes/T-170.md). Two: weekday and weekend. Game day 1, a new game's
first, is a Monday, so days 6 and 7 of each week are the weekend (`day_kind_at` in
`sim/src/track/params.rs`, `app/src/game/days.ts`); no holidays yet. The tables above and the
solve are a weekday's. **A weekend is worked out from the weekday's solve, not solved**: without
crowding a trip's mode and route depend on its period's service only, and the solve keeps each
period's loads, boardings and alightings and each city's trips, rail and walking by purpose
(T-158). So the weekend's riders of a purpose in period q are the weekday's times
`rate x (weekend outward + return share) / (weekday outward + return share)`
(`sim/src/demand/days.rs` `load_factor`); the views are built from those on the main thread and
every query (bubbles, catchments, selections, longer trips, transfers) weighs its periods the
same way in the workers, with the everyday purposes' per-direction weekend shares where it reads
directions. The weekend keeps the weekday's crowded route and mode choices (fewer riders would
crowd less), and the same trains run every day (Anita: no separate weekend schedules for now;
T-261 would need its own solve).

Weekend trips per weekday trip, from NHTS 2017 (person-weighted, Mon-Fri against Saturday and
Sunday; England's NTS agrees for work, shopping and leisure) and, for airports, TSA checkpoint
throughput 2025 by day of week: work 0.30, shopping and errands 1.22, social and leisure 1.60,
school 0.08, university 0.22, airport 1.01, medical 0.24, the regional market's longer trips 1.58
(NHTS trips of 50 miles or more). Weekend period shares are the weekday's above times NHTS's
weekend/weekday ratio in each period, renormalised; school, university, airport and medical keep
their weekday timing (too few weekend trips in NHTS to shape them):

| Weekend, purpose/direction | 6–10 | 10–16 | 16–20 | 20–24 | 0–6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| To work | 54.6% | 16.1% | 11.6% | 8.7% | 9.0% |
| Home from work | 4.5% | 17.5% | 36.6% | 28.9% | 12.5% |
| Errands outward | 9.8% | 63.2% | 23.0% | 3.2% | 0.8% |
| Errands return | 6.6% | 58.6% | 27.7% | 6.6% | 0.5% |
| Social outward | 7.6% | 47.8% | 30.7% | 13.4% | 0.5% |
| Social return | 1.5% | 38.8% | 38.1% | 18.2% | 3.4% |
| Longer trips | 15.1% | 34.9% | 23.5% | 23.9% | 2.6% |

**The average day** (T-464) is a third day the views and queries can be asked for: 5 weekdays and
2 weekend days over 7, taken on each purpose's rate and period shares (`days.rs` `AVERAGE`), so
every figure linear in them is the same blend of the weekday's and the weekend's. It is the demand
views' default; the clock, the money and the boarding account know only the two kinds.

On New York's real network a weekend day has 0.77 of a weekday's local rail legs (6.39M against
8.29M; commutes 1.04M against 3.46M, social 2.60M against 1.62M), its busiest period is midday,
and each purpose keeps its weekday rail share. Costs: no extra solve or opening; the weekend's
views about 7 ms on the main thread a publish; the boarding account twice the work (one account
per kind of day); weekend bubbles are asked for only while shown (about 0.6 s in each solving
worker). Weekday results are bit-identical to before (notes/T-170.md).
