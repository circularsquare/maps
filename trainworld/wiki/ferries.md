# Ferries

A ferry line runs boats on water between ports. The route across the water is free. A port costs US${{sim/src/track/params.rs:PORT_COST|0}}M to construct.

**Three sizes of boat.** You pick one on the line's panel, where a train line picks its train type. Switching buys the boats the line needs beyond the spare ones of that size, and the old boats wait in the depot for another line.

- **Small**: {{sim/src/track/boarding.rs:RATED_PER_CAR|0}} riders, up to {{sim/src/track/ferry.rs:SMALL_TOP*3.6|0}} km/h. US${{sim/src/track/params.rs:SMALL_BOAT_PRICE|0}}M a boat, US${{sim/src/track/params.rs:SMALL_RUN_COST_BOAT_KM|0}} a km to run. Fares are {{app/src/game/fareCurve.ts:FERRY_FARES[0]}} times the usual to start, since fast boats charge more.
- **Medium**: {{sim/src/track/boarding.rs:2*RATED_PER_CAR|0}} riders, up to {{sim/src/track/ferry.rs:BOAT_TOP*3.6|0}} km/h. US${{sim/src/track/params.rs:BOAT_PRICE|0}}M a boat, US${{sim/src/track/params.rs:RUN_COST_BOAT_KM|0}} a km.
- **Large**: {{sim/src/track/boarding.rs:8*RATED_PER_CAR|0}} riders, up to {{sim/src/track/ferry.rs:BOAT_TOP*3.6|0}} km/h. US${{sim/src/track/params.rs:LARGE_BOAT_PRICE|0}}M a boat, US${{sim/src/track/params.rs:LARGE_RUN_COST_BOAT_KM|0}} a km. It stands longer at each port and at the ends, so it needs more boats for the same headway.

A boat never goes faster than the water allows: slow in a narrow channel, full speed only well away from land.

**Long crossings.** A ferry line longer than {{app/src/game/newLine.ts:LONG_FERRY_KM}} km starts with a boat an hour and none at night, unless you set your own default for new ferry lines. A big boat every hour or two usually earns more on a long crossing than smaller ones running often.

**Riders** walk to a port as to a station, wait half the headway, and change to and from trains as anywhere. A crowded boat puts people off as a crowded train does ([[full-trains]]). Fares follow the usual curve, the part above the base times the size's multiplier ([[fares]]).

**Ferry fans.** Some people just like boats. On most everyday trips {{sim/src/demand/everyday.rs:FAN_SHARES[0]*100|0}}% of people are fans, and half as many on airport and medical trips and commutes. For a fan a minute aboard counts as {{sim/src/demand/kernel/fans.rs:FAN_RIDE_W}} of a minute, so a ferry can win their trip even when it's slower.

Exact rule: `docs/track-and-service.md` "6.9 Ferry lines" for boats, money and saves; `docs/demand-mode.md` "Weights, ferry fans, driving and parking" for fans. Fans are on everyday trips and commutes inside a zone only; trips between cities ride ferries as any line.
In the code: **main** `sim/src/track/ferry.rs` `Boat`, `BOAT_TOP`, `SMALL_TOP`, also `sim/src/track/params.rs` `BOAT_PRICE`, `SMALL_BOAT_PRICE`, `LARGE_BOAT_PRICE`, `RUN_COST_BOAT_KM`, `SMALL_RUN_COST_BOAT_KM`, `LARGE_RUN_COST_BOAT_KM`, `PORT_COST` `app/src/game/fareCurve.ts` `FERRY_FARES` `app/src/game/newLine.ts` `LONG_FERRY_KM` `sim/src/demand/everyday.rs` `FAN_SHARES` `sim/src/demand/kernel/fans.rs` `FAN_RIDE_W`, `COMMUTE_FAN_SHARE`
