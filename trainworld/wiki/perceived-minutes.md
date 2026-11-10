# Perceived minutes

People don't count minutes evenly. A minute spent walking feels longer than a minute sitting on a train. So the game adds up each trip in perceived minutes: real minutes, some counted heavier than others. The trip with the fewest perceived minutes is the one [[how-people-choose|people lean towards]].

| A minute of | Counts |
|---|---|
| riding, with room to breathe | one minute |
| waiting on the platform | one minute |
| walking to a station | {{sim/src/demand/kernel.rs:Params.w_walk}} |
| walking between platforms | {{sim/src/demand/kernel.rs:Params.w_xfer}} |
| driving the whole way | {{sim/src/demand/kernel.rs:Params.w_drive}} |
| driving to a station | {{sim/src/demand/kernel.rs:Params.ride_drive_f*Params.w_drive}} |
| walking the whole way | {{sim/src/demand/kernel.rs:Params.walk_trip_w}} |

A trip's total is each part's real minutes times its weight, added up, plus fares and fuel turned into minutes.

Riding gets heavier as the train fills up: see [[full-trains]].

**Money is minutes too.** Fares and fuel go on the same pile, turned into minutes. A dollar of fare counts about {{app/src/game/income.ts:FARE_W_REF}} minutes to someone on a rich country's wage, and more where people earn less. Default fares are lower in poorer places to match. A dollar of driving costs counts {{sim/src/demand/api.rs:FARE_MIN_PER_USD|0}} minutes.

Some things are a fixed price in minutes wherever they happen, like [[parking]] and changing trains ([[waiting-and-changing]]).

Exact rule: `docs/demand-mode.md` "Weights, ferry fans, driving and parking". Between cities a dollar counts less, and differently for each income group: `docs/demand-plain.md` "Between cities (the car and the plane against the train)".
In the code: **main** `sim/src/demand/kernel.rs` `Params`, where every weight is a field, used by `access_cost_min`, `car_drive` `sim/src/demand/kernel/search.rs` `board_cost` `sim/src/demand/api.rs` `FARE_MIN_PER_USD` `app/src/game/income.ts` `fareWeight`
