# Perceived minutes

People don't count minutes evenly. A minute spent walking feels longer than a minute sitting on a train. So the game adds up each trip in perceived minutes: real minutes, some counted heavier than others. The trip with the fewest perceived minutes is the one [people lean towards](how-people-choose.md).

| A minute of | Counts |
|---|---|
| riding, with room to breathe | one minute |
| waiting on the platform | one minute |
| walking to a station | 1.5 |
| walking between platforms | 1.39 |
| driving the whole way | 1.33 |
| driving to a station | 2.66 |
| walking the whole way | 1.5 |

A trip's total is each part's real minutes times its weight, added up, plus fares and fuel turned into minutes.

Riding gets heavier as the train fills up: see [full trains](full-trains.md).

**Money is minutes too.** Fares and fuel go on the same pile, turned into minutes. A dollar of fare counts about 4.4 minutes to someone on a rich country's wage, and more where people earn less. Default fares are lower in poorer places to match. A dollar of driving costs counts 3 minutes.

Some things are a fixed price in minutes wherever they happen, like [parking](parking.md) and changing trains ([waiting and changing](waiting-and-changing.md)).

Exact rule: `docs/demand-mode.md` "Weights, ferry fans, driving and parking". Between cities a dollar counts less, and differently for each income group: `docs/demand-plain.md` "Between cities (the car and the plane against the train)".
In the code: **main** `sim/src/demand/kernel.rs` `Params`, where every weight is a field, used by `access_cost_min`, `car_drive` `sim/src/demand/kernel/search.rs` `board_cost` `sim/src/demand/api.rs` `FARE_MIN_PER_USD` `app/src/game/income.ts` `fareWeight`
