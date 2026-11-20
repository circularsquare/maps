# Ferries

A ferry line runs boats on water between ports. The route across the water is free. A port costs US$10M to construct.

**Three sizes of boat.** You pick one on the line's panel, where a train line picks its train type. Switching buys the boats the line needs beyond the spare ones of that size, and the old boats wait in the depot for another line.

- **Small**: 160 riders, up to 65 km/h. US$15M a boat, US$30 a km to run. Fares are 1.5 times the usual to start, since fast boats charge more.
- **Medium**: 320 riders, up to 45 km/h. US$8M a boat, US$30 a km.
- **Large**: 1,280 riders, up to 45 km/h. US$50M a boat, US$75 a km. It stands longer at each port and at the ends, so it needs more boats for the same headway.

A boat never goes faster than the water allows: slow in a narrow channel, full speed only well away from land.

**Long crossings.** A ferry line longer than 40 km starts with a boat an hour and none at night, unless you set your own default for new ferry lines. A big boat every hour or two usually earns more on a long crossing than smaller ones running often.

**Riders** walk to a port as to a station, wait half the headway, and change to and from trains as anywhere. A crowded boat puts people off as a crowded train does ([full trains](full-trains.md)). Fares follow the usual curve, the part above the base times the size's multiplier ([fares](fares.md)).

**Ferry fans.** Some people just like boats. On most everyday trips 15% of people are fans, and half as many on airport and medical trips. For a fan a minute aboard counts as 0.1 of a minute, so a ferry can win their trip even when it's slower. Commutes have no fans.

Exact rule: `docs/track-and-service.md` "6.9 Ferry lines" for boats, money and saves; `docs/demand-mode.md` "Weights, ferry fans, driving and parking" for fans. Fans are on everyday trips inside a zone only; commutes and trips between cities ride ferries as any line.
In the code: **main** `sim/src/track/ferry.rs` `Boat`, `BOAT_TOP`, `SMALL_TOP`, also `sim/src/track/params.rs` `BOAT_PRICE`, `SMALL_BOAT_PRICE`, `LARGE_BOAT_PRICE`, `RUN_COST_BOAT_KM`, `SMALL_RUN_COST_BOAT_KM`, `LARGE_RUN_COST_BOAT_KM`, `PORT_COST` `app/src/game/fareCurve.ts` `FERRY_FARES` `app/src/game/newLine.ts` `LONG_FERRY_KM` `sim/src/demand/everyday.rs` `FAN_SHARES`, `FAN_RIDE_W`
