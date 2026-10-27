[Back to the spec](../SPEC.md) | [Demand index](demand.md)

# Demand: time of day (4.2)

Periods, kinds of day and the average day.

### 4.2 Time of day

Five periods: morning peak 6-10, midday 10-16, evening peak 16-20, evening 20-24, night 0-6. They
are the player's scheduling periods too (section 6.3: high demand 6-20, the morning peak, midday
and evening peak; medium 20-24; low 0-6), defined once in `sim/src/track/params.rs` (`PERIOD_HOURS`,
`PERIOD_LEVEL`) for schedules and demand alike. A level is a set of whole periods: each period is
solved on its own level's trains, with its own share tables and busiest-hour factor, so the three
periods of high demand are three solves on the same trains. The hours are local time (T-309,
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
| Attractions outward (as social) | 5% | 35% | 42% | 17% | 1% |
| Attractions return | 2% | 25% | 43% | 27% | 3% |

The inspector uses actual purpose/mode/period selection results. Bars show departures per hour,
dividing period counts by 4/6/4/4/6 hours; outward and return rows remain separate.

**Kinds of day** (T-260, T-262, T-264, T-1373, notes/T-170.md). Weekday, Saturday, Sunday, and a
superholiday's day away, going-out day and coming-back day (below). Game day 1,
a new game's first, is a Monday, so days 6 and 7 of each week are Saturday and Sunday
(`day_kind_at` in `sim/src/track/params.rs`, `app/src/game/days.ts`) where a line's country says
nothing else: its own weekend days and public holidays decide its kinds of day (a holiday runs as
a Sunday; docs/world.md "Calendar"). The tables above and the solve are a weekday's. **A Saturday
and a Sunday are worked out from the weekday's solve, not solved**: without crowding a trip's mode
and route depend on its period's service only, and the solve keeps each period's loads, boardings
and alightings and each city's trips, rail and walking by purpose (T-158). So a Saturday's riders
of a purpose in period q are the weekday's times
`rate x (Saturday outward + return share) / (weekday outward + return share)`
(`sim/src/demand/days.rs` `load_factor`), and a Sunday's likewise; the views are built from those
on the main thread and every query (bubbles, catchments, selections, longer trips, transfers)
weighs its periods the same way in the workers, with the everyday purposes' per-direction shares
where it reads directions. Both keep the weekday's crowded route and mode choices (fewer riders
would crowd less), and the same trains run every day (Anita: no separate weekend schedules for
now; T-261 would need its own solve). **A superholiday's day away** (T-264) is a fourth kind,
worked out the same way: a Sunday's load factor times `AWAY_RATE` (local outings 0.7, attractions
1.5, the rest 1), not in the average day. **Its going-out and coming-back days** (T-1373) are a
weekday and a Sunday with the longer trips `RUSH_LONGER` (2) times a weekday's, at a weekday's and
a Sunday's timing, and leaning by direction: going out, each place pair's longer trips from the
larger place to the smaller are times `1 + lean`, the other way `1 - lean`, coming back the
opposite, so a pair's trips in all are kept ([demand-market.md](demand-market.md) has the lean).
The market's loading pass fills one more column for that, the longer trips' loads times their
pair's lean, on the same routes and modes (`IntercityApi::lean_loads`); a going-out day's longer
trips are theirs plus it, a coming-back day's theirs less it, before the day's factor. Queries
(selections, a station's far zones, the trip explorer) weigh each pair by its places' lean. Neither
day is in the average. Demand days are weekday 0, Saturday 1, Sunday 2, away 3, going out 4, coming
back 5 and the average 6.

Trips per weekday trip, from NHTS 2017 (person-weighted, Mon-Fri against Saturday and against
Sunday, the mean of the outward and return ratios; England's NTS agrees for the weekend as a whole
for work, shopping and leisure) and, for airports, TSA checkpoint throughput 2025 by day of week;
longer trips are NHTS trips of 50 miles or more:

| Purpose | Saturday | Sunday |
| --- | ---: | ---: |
| Work | 0.37 | 0.24 |
| Shopping and errands | 1.39 | 1.05 |
| Social and leisure | 1.54 | 1.67 |
| School | 0.11 | 0.06 |
| University | 0.30 | 0.14 |
| Airport | 0.92 | 1.11 |
| Medical | 0.30 | 0.18 |
| Attractions (as social) | 1.54 | 1.67 |
| Longer trips | 1.69 | 1.48 |

A day's period shares are the weekday's above times NHTS's day/weekday ratio in each period,
renormalised (`research/weekend/tables.py`); school, university, airport and medical keep their
weekday timing (too few weekend trips in NHTS to shape them), and attractions take social's:

| Purpose/direction, Saturday / Sunday | 6–10 | 10–16 | 16–20 | 20–24 | 0–6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| To work | 58.6 / 48.3% | 15.0 / 17.7% | 11.5 / 11.7% | 6.3 / 12.7% | 8.6 / 9.6% |
| Home from work | 4.0 / 5.1% | 18.4 / 16.3% | 37.1 / 35.9% | 30.4 / 26.7% | 10.1 / 16.0% |
| Errands outward | 11.0 / 8.2% | 62.3 / 64.3% | 22.8 / 23.2% | 3.1 / 3.5% | 0.8 / 0.8% |
| Errands return | 7.0 / 6.0% | 57.2 / 59.9% | 28.3 / 27.1% | 6.7 / 6.5% | 0.8 / 0.5% |
| Social outward | 5.9 / 9.6% | 44.0 / 51.6% | 33.5 / 27.6% | 16.0 / 10.8% | 0.6 / 0.4% |
| Social return | 1.6 / 1.4% | 31.0 / 45.6% | 39.5 / 36.8% | 22.1 / 14.7% | 5.8 / 1.5% |
| Longer trips | 16.6 / 13.3% | 31.4 / 39.4% | 19.6 / 28.5% | 29.5 / 16.6% | 2.9 / 2.2% |

**The average day** (T-464) is a fourth day the views and queries can be asked for: 5 weekdays, a
Saturday and a Sunday over 7, taken on each purpose's rate and period shares (`days.rs`
`AVERAGE`), so every figure linear in them is the same blend. It is the demand views' default;
the clock, the money and the boarding account know only the three kinds. Saturday and Sunday
bubbles are asked for only while shown; New York's weekend figures and the costs are in
notes/T-170.md and notes/T-263.md.
