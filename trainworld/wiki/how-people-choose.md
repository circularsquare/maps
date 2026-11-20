# How people choose

Every trip can go on foot, by car, or on your network. Each way gets a cost in [[perceived-minutes]]. Cheaper wins more of the trips, smoothly, with no cliff edge.

**Walking goes first.** A walk's share depends on how many minutes it saves against the car and your network, with no head start. Walks over {{sim/src/demand/kernel.rs:WALK_REACH_KM|0}} km aren't on the menu.

**Some people won't walk far.** {{sim/src/demand/kernel.rs:Params.walk_tier * 100|0}}% of people never walk the whole way once it's over {{sim/src/demand/kernel.rs:Params.walk_tier_m|0}} m. They pick between your network and the car, and they'll take a train or bus even when walking would be quicker.

**Then train against car.** What's left is split by how many times cheaper one is than the other, so ten minutes matter a lot on a short trip and hardly at all on a long one. The train's share is one over one plus the cube of its cost over the car's, so equal costs split the trips in half and the share drops fast on either side. If the train costs twice as much as driving, only a small slice of people take it.

Past {{sim/src/demand/kernel.rs:Params.ratio_fade_lo|0}} times as dear, the train fades away, and from {{sim/src/demand/kernel.rs:Params.ratio_fade_hi|0}} times nobody takes it, so a pointless detour line gets nobody. The car fades the same way when it's the dear one.

**Nobody has a soft spot for trains.** Rail and the car are compared on time and money alone.

**Trains never lose to walking.** A train that takes longer than walking there isn't offered at all, counting real minutes on both sides. The people who won't walk far are the exception.

**Islands.** Where no road reaches, the car and the walk are off the table. Your ferry or train takes the trips, unless it costs far more than a drive over a bridge that isn't there.

Parts of the cost live in [[walking-to-stations]], [[waiting-and-changing]], [[full-trains]], [[driving]] and [[parking]].

Exact rule: `docs/demand-mode.md` "Cost and the rail-against-car curve" "Walking and land parts". This page is commutes and everyday trips. Between cities the plane joins in and walking is out: `docs/regional.md` "Mode choice".
In the code: **main** `sim/src/demand/kernel.rs` `car_choice`, which calls `ratio_share` for the train against car split; also `walk_trip_cost`, `bound_share` `sim/src/demand/kernel/mode.rs` `mode_choice_part`
