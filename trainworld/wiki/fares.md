# Fares

One fare curve covers the whole network. It says what a trip of a given length costs, and you set it in the Money tab.

**The curve** is three points: a base fare for a trip of no length, then a fare at {{app/src/game/fareCurve.ts:NODE_KM[1]|0}} km and one at {{app/src/game/fareCurve.ts:NODE_KM[2]|0}} km. Between points it runs in straight lines, and past the last it keeps the same slope. A point can't be cheaper than the one before, so a fare never falls as a trip gets longer and never goes under the base.

A new game starts with a base of US${{app/src/game/fareDefault.ts:START_BASE_US}} where you start in the US. The fare at {{app/src/game/fareCurve.ts:NODE_KM[1]|0}} km is about {{app/src/game/fareCurve.ts:START_NODES[1][1]/START_NODES[0][1]|1}} times the base, and at {{app/src/game/fareCurve.ts:NODE_KM[2]|0}} km about {{app/src/game/fareCurve.ts:START_NODES[2][1]/START_NODES[0][1]|1}} times.

**Fares by country.** Each country charges the curve times its own scale. The default scale follows income: the US's scale times a country's income per head over the US's, raised to the power {{app/src/game/income.ts:FARE_E}}. Poorer places charge less, and a dollar of fare weighs more in minutes there ([[perceived-minutes]]), so a default fare feels about the same everywhere. A new game's curve is set once for the country it starts in. You can set your own scale for a country as a multiple of a reference country's fares.

A trip pays the scale of the country it starts in, on its whole length, even if it changes lines.

**Faster lines.** A line with high-speed trains charges the part of the curve above the base times {{app/src/game/fareCurve.ts:HSR_FARE}}, unless you set another. A ferry line has its own multiplier, {{app/src/game/fareCurve.ts:FERRY_FARE}} to start. The base is paid once a trip, whatever it rides.

**Changing trains** can cost a transfer fare, paid on every boarding after the first. It starts at nothing.

**Who gets the money.** The base fares collected on the network are shared between lines, half by how many people board each line and half by passenger-km, and no line takes more than one base for a boarding. A short feeder gets less than an even share and a long line more. The per-km part goes to the line that carried it.

**Higher fares lose riders.** A fare is money, and money is minutes ([[perceived-minutes]]), so raising it changes how many trips pick the train ([[how-people-choose]]). The Money tab shows a new fare at once, but a line's income keeps the old fares until riders solved with the new ones appear.

Exact rule: `docs/track-and-service.md` "Fares and the ledgers" for the curve, the scales, the multipliers and the split between lines. Trips between cities use the curve itself, not a country's scale, and change lines free.
In the code: **main** `app/src/game/fareCurve.ts` `faresFor`, also `fareAt`, `lineFareMul`, `START_NODES`, `HSR_FARE` `app/src/game/income.ts` `incomeScale`, `FARE_E` `app/src/game/fareDefault.ts` `startNodesAt`
