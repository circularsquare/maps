# Rush hours

A day is cut into five periods, each solved on its own.

| Period | Hours | Trains |
|---|---|---|
| Morning peak | {{sim/src/track/params.rs:PERIOD_HOURS[0][0]|0}} to {{sim/src/track/params.rs:PERIOD_HOURS[0][1]|0}} | high |
| Midday | {{sim/src/track/params.rs:PERIOD_HOURS[1][0]|0}} to {{sim/src/track/params.rs:PERIOD_HOURS[1][1]|0}} | high |
| Evening peak | {{sim/src/track/params.rs:PERIOD_HOURS[2][0]|0}} to {{sim/src/track/params.rs:PERIOD_HOURS[2][1]|0}} | high |
| Evening | {{sim/src/track/params.rs:PERIOD_HOURS[3][0]|0}} to {{sim/src/track/params.rs:PERIOD_HOURS[3][1]|0}} | medium |
| Night | {{sim/src/track/params.rs:PERIOD_HOURS[4][0]|0}} to {{sim/src/track/params.rs:PERIOD_HOURS[4][1]|0}} | low |

Your high, medium and low train settings run in those hours.

Commutes bunch up. {{sim/src/demand/api.rs:TO_WORK[0]*100|0}}% of the day's trips to work start in the morning peak, and {{sim/src/demand/api.rs:TO_HOME[2]*100|0}}% of the trips home in the evening peak. Errands and visits spread more evenly through the day.

Inside a period the busiest hour is busier than the average hour, up to {{sim/src/track/params.rs:max(PEAK_HOUR_FACTOR)}} times, and [[full-trains|crowding]] is judged at that hour.

Weekends and public holidays have their own demand, with less commuting and more outings. Long holidays bring people going out one way and coming back the other. Each country decides which days those are. Trains run the same on every day.

The trips themselves are in [[who-travels]].

Exact rule: `docs/demand-timing.md` "Time of day". The commute shares here are commutes only. Everyday trips have a table for each purpose.
In the code: `sim/src/track/params.rs` `PERIOD_HOURS`, `PERIOD_LEVEL`, `PEAK_HOUR_FACTOR` `sim/src/demand/api.rs` `TO_WORK`, `TO_HOME` `sim/src/demand/days.rs` `day_rate` `sim/src/demand/everyday.rs` `OUTWARD`
