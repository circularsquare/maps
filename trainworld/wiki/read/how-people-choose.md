# How people choose

Every trip can go on foot, by car, or on your network. Each way gets a cost in [perceived minutes](perceived-minutes.md). Cheaper wins more of the trips, smoothly, with no cliff edge.

**Walking goes first.** A walk's share depends on how many minutes it saves. It starts 10 minutes ahead of the car, which stands for getting to the car and walking from the space. Walks over 10 km aren't on the menu.

**Then train against car.** What's left is split by how many times cheaper one is than the other, so ten minutes matter a lot on a short trip and hardly at all on a long one. If the train costs twice as much as driving, only a small slice of people take it.

Past 3 times as dear, the train fades away, and from 4 times nobody takes it, so a pointless detour line gets nobody. The car fades the same way when it's the dear one.

**Nobody has a soft spot for trains.** Rail and the car are compared on time and money alone.

**Trains never lose to walking.** A train that takes longer than walking there isn't offered at all, counting real minutes on both sides.

**Islands.** Where no road reaches, the car and the walk are off the table. Your ferry or train takes the trips, unless it costs far more than a drive over a bridge that isn't there.

Parts of the cost live in [walking to stations](walking-to-stations.md), [waiting and changing](waiting-and-changing.md), [full trains](full-trains.md), [driving](driving.md) and [parking](parking.md).

Exact rule: `docs/demand-mode.md` "Cost and the rail-against-car curve" "Walking and land parts". This page is commutes and everyday trips. Between cities the plane joins in and walking is out: `docs/regional.md` "Mode choice".
In the code: `sim/src/demand/kernel.rs` `ratio_share`, `car_choice`, `walk_trip_cost`, `bound_share` `sim/src/demand/kernel/mode.rs` `mode_choice_part`
