# Parking

Parking is what makes people give up the car in a dense city. Every drive pays for it, in [perceived minutes](perceived-minutes.md).

**Finding a space** takes 7 minutes on every trip, everywhere.

**Paying for it** only happens where jobs are packed in. Above 3,000 jobs a square km, the price climbs with density, up to 50 perceived minutes in the densest centres. Some drivers park free anyway, fewer the denser it gets.

**Congestion charges** (Manhattan, London) are added on top for drivers arriving there.

**Parking at home** counts too. Where people live densely the car costs something to keep there, 40% of what parking costs at work by the same rule, counted on people living there. This is why dense neighbourhoods take the train more.

**Poorer countries** pay less of both. Below $35,000 a head the prices shrink with income. Above that they're full price.

If you drive to a station instead, see [walking to stations](walking-to-stations.md).

Exact rule: `docs/demand-mode.md` "Weights, ferry fans, driving and parking". Parking where you arrive applies to commutes, everyday trips and trips between cities. Parking at home applies to commutes and everyday trips only.
In the code: `sim/src/demand/kernel.rs` `park_parts`, `home_park_at`, `park_income_of`, `ride_station_end`
