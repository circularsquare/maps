# Fares

One fare curve covers the whole network. It says what a trip of a given length costs, and you set it in the Money tab.

**The curve** is three points in the Money tab: a base fare for a trip of no length, then a fare at 20 km and one at 100 km. The Advanced fares panel, opened from the Money tab, can give it anywhere from one point (one flat fare for every trip) to 12; two points 0.1 km apart make a step, so a curve can be a staircase of fare bands. Between points it runs in straight lines, and past the last it keeps the same slope. A point can't be cheaper than the one before, so a fare never falls as a trip gets longer and never goes under the base. No point can go over 1,000 (US$, before a country's scale).

A new game starts with a base of US$2.5 where you start in the US. The fare at 20 km is about 2.6 times the base, and at 100 km about 9.0 times.

**Fares by country.** Each country charges the curve times its own scale. The default scale follows income: the US's scale times a country's income per head over the US's, raised to the power 0.5. Poorer places charge less, and a dollar of fare weighs more in minutes there ([perceived minutes](perceived-minutes.md)), so a default fare feels about the same everywhere. A new game's curve is set once for the country it starts in. You can set your own scale for a country as a multiple of a reference country's fares.

A trip that changes lines still pays the curve once, on its whole length. Each part of it pays the scale of the country it leaves from: the base fare at the stop where it boards, each stretch at the stop where that stretch starts.

**Faster lines.** A line with high-speed trains charges the part of the curve above the base times 1.25, unless you set another. A ferry line has its own multiplier, 1 to start. Any one line can also get its own fare multiple, from nothing up to 3, at the bottom of its panel; it works the same way, on top of the others, and it scales the base too. The base is paid once a trip, times the highest multiple among the lines the trip rides, so a trip that changes onto a dearer line pays that line's base.

**Making a long high-speed line pay.** Most of what a high-speed line spends goes on running its trains, and that grows with every car and every train an hour. Fares barely move until the trains fill. A line from Tokyo to Osaka with sixteen cars every ten minutes loses money every day. The same line with five or eight cars makes a profit with the same riders. Start with five to eight cars and a train every twenty to thirty minutes, and add cars or trains when the line panel shows full trains. "Losing money" here is the daily profit in the profitability map and the Finances tab, fares minus running and upkeep. Paying back the building cost takes far longer.

**Fare zones.** In the Advanced fares panel you can make a fare area and paint its stations into numbered fare zones, one click a station or a loop around many. A trip that boards and ends at stations of one fare area pays that fare area's price for the fare zones it spans, counted from the lower number to the higher with both included: fare zone one to fare zone three is three fare zones, and two stations in fare zone two are one fare zone even when the trip passes through fare zone one. That price takes the curve's place, so the base share, a line's multiple, high-speed lines and transfer fares work as before. Each fare area has a price for one fare zone up to 12, and a new one starts at the curve's fares for 5 km a fare zone. A trip with an end outside the fare area, or in another fare area, pays the curve for the whole trip.

**Changing trains** can cost a transfer fare, paid on every boarding after the first. It starts at nothing and can't be set above the base fare.

**Who gets the money.** The base fares collected on the network are shared between lines, half by how many people board each line and half by passenger-km, and no line takes more than one base for a boarding. A short feeder gets less than an even share and a long line more. The per-km part goes to the line that carried it.

**Higher fares lose riders.** A fare is money, and money is minutes ([perceived minutes](perceived-minutes.md)), so raising it changes how many trips pick the train ([how people choose](how-people-choose.md)). The Money tab shows a new fare at once, but a line's income keeps the old fares until riders solved with the new ones appear.

Exact rule: `docs/track-and-service.md` "Fares and the ledgers" for the curve, the scales, the multipliers, the fare zones and the split between lines. Trips between cities change lines free.
In the code: **main** `app/src/game/fareCurve.ts` `faresFor`, also `fareAt`, `lineFareMul`, `START_NODES`, `HSR_FARE` `app/src/game/income.ts` `incomeScale`, `FARE_E` `app/src/game/fareDefault.ts` `startNodesAt` `app/src/game/fareZones.ts` `demandZones`, `chargedPrices`
