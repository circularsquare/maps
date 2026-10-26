"""T-1275: does a New York - London high-speed line pay, on the game's rules?
Rail against the plane in the market's logit (intercity.rs: share = v / (rest + v), v =
exp(-bt (rail - plane)), rest 1 where no car goes, bt = 0.10 sqrt(60 / car minutes))."""
import math

PLANE = 1788.0          # perceived minutes New York - London (atlantic.py)
BT = 0.0096             # the scale, from the car's minutes as though a road crossed the sea
L = 5600.0              # route km of track (great circle 5,562 between the places' centres)
RUN_MIN = L / 300 * 60  # 320 km/h track, about 300 average with the ends
ACCESS = 60.0           # perceived minutes to and from the two stations (walks, rides, a change)
AVG_DAY = (5 + 1.69 + 1.48) / 7
MARKETS = {"New York - London places": 1965, "places within 200 km of each": 3875, "within 400 km (a feeder network)": 14459}
CRUSH, START = 160, 0.7
ECON = 100


def best(trips_each_way_weekday, hw):
    t = trips_each_way_weekday * AVG_DAY * 2
    rail0 = RUN_MIN + ACCESS + hw / 2
    top = (0, 0, 0)
    for f in range(0, 1500, 5):
        rail = rail0 + 3 * f
        v = math.exp(max(-80, min(80, -BT * (rail - PLANE))))
        s = v / (1 + v)
        rev = t * s * f
        if rev > top[0]:
            top = (rev, f, s)
    return top, t


def costs(riders_day, hw, level_mult=1.0, water=2.0):
    n = 1440 / hw                       # trains a day each way
    per_train = riders_day / 2 / n
    cars = max(4, min(20, math.ceil(per_train / (CRUSH * START))))
    run = 2 * n * L * (5 + 3 * cars)    # real US$ a day
    upkeep = L * 300 * 1.5 + 2 * 500
    trains = math.ceil(2 * (RUN_MIN + 10) / hw)
    fleet = trains * cars * 3.75e6
    build = L * 90e6 * level_mult * water * 1.5
    return run, upkeep, fleet, build, cars, per_train


print(f"run {RUN_MIN:.0f} min one way; plane {PLANE:.0f} perceived min")
for name, w in MARKETS.items():
    print(f"\n{name}: {w} trips a weekday each way")
    for hw in (60, 120, 240, 720, 1440):
        (rev, f, s), t = best(w, hw)
        riders = t * s
        run, upkeep, fleet, build, cars, per_train = costs(riders, hw)
        net = rev - run - upkeep
        print(f"  every {hw:4d} min: best fare ${f}, rail {100*s:.0f}% of {t:.0f} trips a day -> {riders:.0f} riders, "
              f"fares ${rev/1e6:.2f}M/day, running ${run/1e6:.2f}M ({cars} cars, {per_train:.0f} a train), upkeep ${upkeep/1e6:.2f}M, "
              f"net ${net/1e6:.2f}M a real day = ${ECON*net/1e9:.2f}B a game day; fleet ${fleet/1e9:.2f}B; build ${build/1e12:.2f}T")
# what the line costs at the ocean's prices, and at ground level where no pack's water mask reaches
for lab, lm, wa in (("bridge or tunnel at +1/-1 (2x over water)", 1.0, 2.0), ("tunnel at -3", 1.2, 2.0), ("level 0 priced as land (no mask mid-ocean)", 0.5, 1.0)):
    print(f"build {lab}: ${L*90e6*lm*wa*1.5/1e12:.2f}T")
# break-even: the best headway (trains of at most 20 cars, filled to 70% of crush) for a market
for w in (5000, 10000, 20000, 30000, 40000, 80000, 134435):
    top = None
    for hw in (10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720):
        (rev, f, s), t = best(w, hw)
        riders = t * s
        if riders / 2 / (1440 / hw) > CRUSH * START * 20:
            continue
        run, upkeep, fleet, build, cars, per_train = costs(riders, hw)
        net = rev - run - upkeep
        if top is None or net > top[0]:
            top = (net, hw, f, rev, run, upkeep, cars, riders)
    net, hw, f, rev, run, upkeep, cars, riders = top
    pay = 1.51e12 / (ECON * net) if net > 0 else float("inf")
    print(f"{w} each way weekday: every {hw} min, {cars} cars, fare ${f}, {riders:.0f} riders, fares ${rev/1e6:.1f}M, running ${run/1e6:.1f}M, upkeep ${upkeep/1e6:.1f}M, net ${net/1e6:.1f}M a real day; $1.51T back in {pay:.0f} game days ({pay*90/3600:.0f} h at the top speed)")
