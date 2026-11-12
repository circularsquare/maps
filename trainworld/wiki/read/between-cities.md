# Trips between cities

Longer trips, from one city to another, start the same way as every trip: someone wants to go. The game works out how many people want to go where before it asks how they get there. Your network gets a share of them, and the car and the plane take the rest.

**Who goes.** Bigger places send and draw more. Richer countries make more long trips: a country's trips scale with its income per head, up to the US level, to the power 0.7. A resident makes 0.107 ground trips a weekday at the US level, and the chance of going to a place falls with distance to the power -2.37. That ground trip phases out between 300 and 700 km. From there a thin tail of far trips takes over, mostly between big places, and falls off much more slowly with distance, to the power -0.7.

**Big cities draw more than their size.** From the next world update, a capital or a big regional centre pulls more of the long trips than its people alone would. The game compares a place's rank on Natural Earth's map ranking of cities with the rank a place of its size usually has; each step above that, up to 3, multiplies its pull by e to the power 0.6. Berlin ranks three steps above its size, so trips of 700 km or more from elsewhere in Germany go there about six times as readily. Remote centres that the ranking lifts for map balance get less, and beach and resort towns get some too. The extra pull fades in between 300 and 700 km, so trips shorter than that keep their pattern, and it is weaker on trips from abroad. People don't travel more because of it: each place sends the same number of trips, just more of them to the big cities. Trips from the US, Japan, the UK and Korea keep the old pattern, which already matched their surveys. So do trips from Russia and Turkey, which already had as many long trips as their published counts.

Crossing a border cuts the trips. Between two countries outside the open-border area (the EU, the UK and a few more) a trip keeps 0.26 of its chance at the US income level, and less for a poorer origin. Inside the open-border area it keeps half.

Places closer than 250 km pair up zone by zone. Farther ones pair as whole places, spread over their zones.

**Three ways to go:**

- **Car.** A fixed bit of getting going, then the road at a steady pace. The road is 1.25 times the straight line unless the game has a routed drive for the pair. The trip costs fuel and tolls per km of road, about US$0.15 where a country has no figure of its own, and [parking](parking.md) at the far end. Two people share the cost. Ferries and the Channel add their own fare and time.
- **Plane.** Only between far places that both have an airport. Door to door it counts the drive to the airport, 135 minutes in the terminal at both ends together, a wait for the flight, the flight at 750 km/h plus 30 minutes, and 60 more with no direct flight. The fare is US$100 plus US$0.05 a km. You don't build planes and they don't fill up.
- **Train.** Your network, with its fares, timetable, changes and crowding. A long trip is planned, so its first wait is shorter than a local one, at most 30 minutes ([waiting and changing](waiting-and-changing.md)).

Walking is not on the menu. The three are compared in the usual way ([how people choose](how-people-choose.md)), and a long trip softens the comparison: ten minutes matter less on a long trip.

**Money counts less here.** People take longer trips knowing the price, so a dollar is worth fewer minutes than on a local trip. It also depends on income: richer people mind a dollar less. Each country's people are split into three income groups, and each group chooses on its own between the car, the plane and your trains. Which of your trains they take is picked once for all three groups, at their average, so a high-speed line's extra fare counts less on a long trip than on a commute.

**Commutes and outings across a zone line** are not part of this. They choose the way local trips do ([how people choose](how-people-choose.md)).

**When.** Most long trips happen in the day and few at night. The weekend has more of them than a weekday.

Exact rule: `docs/regional.md` "The total travel market, then rail's share", "Mode choice" and "Where a long-distance trip starts and ends". , the figure lives in `pipeline/longhaul.json`. Commutes and everyday outings between zones are in `docs/zones.md` "Commutes between zones" and "Everyday outings across a zone line".
In the code: **main** `pipeline/longhaul.json` for how many trips, copied into the registry's market rule. `pipeline/market_pull.json` for the big cities' pull, `pipeline/market_pull.py` `table`. `sim/src/demand/intercity/costs.rs` `air_cost`, `routed_car`, `ROAD_DETOUR` `sim/src/demand/intercity.rs` `SHARES`, `NEAR_KM`, `ONSET_KM`, `DECAY_A`, `pull_gamma_of` `sim/src/demand/intercity/market.rs` `pull_w`
