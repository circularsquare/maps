# Trips between cities

Longer trips, from one city to another, start the same way as every trip: someone wants to go. The game works out how many people want to go where before it asks how they get there. Your network gets a share of them, and the car and the plane take the rest.

**Who goes.** Bigger places send and draw more. Richer countries make more long trips: a country's trips scale with its income per head, up to the US level, to the power {{pipeline/longhaul.json:eps}}. A resident makes {{pipeline/longhaul.json:R|3}} ground trips a weekday at the US level, and the chance of going to a place falls with distance to the power -{{pipeline/longhaul.json:A}}. That ground trip phases out between {{pipeline/longhaul.json:tail_onset_km[0]|0}} and {{pipeline/longhaul.json:tail_onset_km[1]|0}} km. From there a thin tail of far trips takes over, mostly between big places, and falls off much more slowly with distance, to the power -{{pipeline/longhaul.json:G}}.

Crossing a border cuts the trips. Between two countries outside the open-border area (the EU, the UK and a few more) a trip keeps {{pipeline/longhaul.json:border}} of its chance at the US income level, and less for a poorer origin. Inside the open-border area it keeps half.

Places closer than {{sim/src/demand/intercity.rs:NEAR_KM|0}} km pair up zone by zone. Farther ones pair as whole places, spread over their zones.

**Three ways to go:**

- **Car.** A fixed bit of getting going, then the road at a steady pace. The road is {{sim/src/demand/intercity/costs.rs:ROAD_DETOUR}} times the straight line unless the game has a routed drive for the pair. The trip costs fuel and tolls per km of road, about US${{sim/src/demand/intercity/costs.rs:CAR_USD_PER_KM}} where a country has no figure of its own, and [[parking]] at the far end. Two people share the cost. Ferries and the Channel add their own fare and time.
- **Plane.** Only between far places that both have an airport. Door to door it counts the drive to the airport, {{sim/src/demand/intercity/costs.rs:AIR_TERMINAL_MIN|0}} minutes in the terminal at both ends together, a wait for the flight, the flight at {{sim/src/demand/intercity/costs.rs:AIR_KMH|0}} km/h plus {{sim/src/demand/intercity/costs.rs:AIR_FIXED_MIN|0}} minutes, and {{sim/src/demand/intercity/costs.rs:AIR_NO_DIRECT_MIN|0}} more with no direct flight. The fare is US${{sim/src/demand/intercity/costs.rs:AIR_FARE_USD|0}} plus US${{sim/src/demand/intercity/costs.rs:AIR_USD_PER_KM}} a km. You don't build planes and they don't fill up.
- **Train.** Your network, with its fares, timetable, changes and crowding. A long trip is planned, so its first wait is shorter than a local one, at most {{sim/src/demand/kernel/search.rs:PLANNED_WAIT_CAP_MIN|0}} minutes ([[waiting-and-changing]]).

Walking is not on the menu. The three are compared in the usual way ([[how-people-choose]]), and a long trip softens the comparison: ten minutes matter less on a long trip.

**Money counts less here.** People take longer trips knowing the price, so a dollar is worth fewer minutes than on a local trip. It also depends on income: richer people mind a dollar less. Each country's people are split into three income groups, and each group chooses on its own between the car, the plane and your trains. Which of your trains they take is picked once for all three groups, at their average, so a high-speed line's extra fare counts less on a long trip than on a commute.

**Commutes and outings across a zone line** are not part of this. They choose the way local trips do ([[how-people-choose]]).

**When.** Most long trips happen in the day and few at night. The weekend has more of them than a weekday.

Exact rule: `docs/regional.md` "The total travel market, then rail's share", "Mode choice" and "Where a long-distance trip starts and ends". , the figure lives in `pipeline/longhaul.json`. Commutes and everyday outings between zones are in `docs/zones.md` "Commutes between zones" and "Everyday outings across a zone line".
In the code: **main** `pipeline/longhaul.json` for how many trips, copied into the registry's market rule. `sim/src/demand/intercity/costs.rs` `air_cost`, `routed_car`, `ROAD_DETOUR` `sim/src/demand/intercity.rs` `SHARES`, `NEAR_KM`, `ONSET_KM`, `DECAY_A`
