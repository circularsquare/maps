# Rush hours

A day is cut into five periods, each solved on its own.

| Period | Hours | Trains |
|---|---|---|
| Morning peak | 6 to 10 | high |
| Midday | 10 to 16 | high |
| Evening peak | 16 to 20 | high |
| Evening | 20 to 24 | medium |
| Night | 0 to 6 | low |

Your high, medium and low train settings run in those hours.

Commutes bunch up. 70% of the day's trips to work start in the morning peak, and 58% of the trips home in the evening peak. Errands and visits spread more evenly through the day.

Inside a period the busiest hour is busier than the average hour, up to 1.5 times, and [crowding](full-trains.md) is judged at that hour.

Weekends and public holidays have their own demand, with less commuting and more outings. Long holidays bring people going out one way and coming back the other. Each country decides which days those are. Trains run the same on every day.

The trips themselves are in [who travels](who-travels.md).

Exact rule: `docs/demand-timing.md` "Time of day". The commute shares here are commutes only. Everyday trips have a table for each purpose.
In the code: **main** `sim/src/track/params.rs` `PERIOD_HOURS`, also `PERIOD_LEVEL`, `PEAK_HOUR_FACTOR` `sim/src/demand/api.rs` `TO_WORK`, `TO_HOME` `sim/src/demand/days.rs` `day_rate` `sim/src/demand/everyday.rs` `OUTWARD`
