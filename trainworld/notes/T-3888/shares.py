"""T-3888: bus share of a short trip under today's rules and the walk-tier options.

A hand copy of Params::car_choice / ratio_share / walk_no_rail for one trip with one bus option
(no station-pair logsum, one parking case per row). Not the game; numbers to compare options with.
Assumed bus inputs: stops every 400 m at 19 km/h (docs: about 19 km/h with stops every 400 m),
250 m walk at each end (a home 150 m from the route plus 100 m), a bus every 10 minutes.
"""
import math

BETA, BONUS = 0.10, 10.0
W_WALK = 1.5
WALK_MIN_KM = 1.3 / 6.0 * 60  # 13 real minutes a straight km
FARE_W = 4.4                  # US minutes a dollar
OWN_TRIP, OWN_KM = 2.1 * 3, 0.1365 * 3


def ratio(rail, car):
    q = max(rail, 0.01) / car
    s = 1 / (1 + q ** 3)
    fr = min(max(4 - q, 0), 1)
    fc = min(max(4 - 1 / q, 0), 1)
    return 1 - (1 - s * fr) * fc


def car_cost(d, place):
    """'dense': 16.5 km/h streets, every driver pays a space (15) and home parking (7);
    'suburb': 30 km/h streets, parks free."""
    road = 1.3 * d
    kmh = {"dense": 16.5, "suburb": 30.25, "plain": 16.5}[place]
    park = {"dense": 7 + 7 + 15, "suburb": 7, "plain": 7}[place]
    return road / kmh * 60 * 1.33 + road * OWN_KM + OWN_TRIP + park


def walk_cost(d, far_w=W_WALK):
    t = d * WALK_MIN_KM
    return min(t, 10) * W_WALK + max(t - 10, 0) * far_w


def bus(d, board, fare_mult, headway=10.0, w=W_WALK, access_m=250.0):
    walk_real = access_m / 1000 * WALK_MIN_KM
    ride = 1.3 * d / 19.0 * 60
    fare = (2.50 + 0.20 * fare_mult * 1.3 * d) * FARE_W
    return 2 * walk_real * w + headway / 2 + ride + board + fare, 2 * walk_real + headway / 2 + ride


def shares(d, place, board=0.0, fare_mult=0.8, w_access=W_WALK, walks=True, rule=True,
           asc_rail=0.0, far_w=W_WALK, headway=10.0, rail=None):
    """(bus share, walk share) of all trips."""
    car = car_cost(d, place)
    if rail is None:
        rail, real = bus(d, board, fare_mult, headway, w_access)
    else:
        rail, real = rail, 0.0
    walk = walk_cost(d, far_w)
    allowed = (real <= d * WALK_MIN_KM) or not rule
    bt = BETA * min(1.0, math.sqrt(60 / car))
    uw = bt * (car - walk) + BONUS * BETA
    if not allowed:
        return 0.0, (1 / (1 + math.exp(-uw)) if walks else 0.0)
    if not walks:
        return ratio(rail, car), 0.0
    util = bt * (car - rail) + asc_rail
    e = math.exp(uw - util)
    pw = e / (1 + math.exp(-util) + e)
    return (1 - pw) * ratio(rail, car), pw


def mix(s, a, b):
    return tuple((1 - s) * x + s * y for x, y in zip(a, b))


def tier_a(d, place, s=0.10, limit=0.8, **kw):
    """Option A: a share s walks the whole way only up to `limit` km; past it they don't walk and
    the walking rule doesn't refuse them."""
    if d <= limit:
        return shares(d, place, **kw)
    return mix(s, shares(d, place, **kw), shares(d, place, walks=False, rule=False, **kw))


def tier_b(d, place, s=0.10, w2=3.0, **kw):
    """Option B: a share s weighs every walk minute, to the stop and the whole way, at w2."""
    def sh(w):
        car = car_cost(d, place)
        rail, real = bus(d, 0.0, 0.8, 10.0, w)
        walk = d * WALK_MIN_KM * w
        bt = BETA * min(1.0, math.sqrt(60 / car))
        uw = bt * (car - walk) + BONUS * BETA
        if real > d * WALK_MIN_KM:
            return 0.0, 1 / (1 + math.exp(-uw))
        util = bt * (car - rail)
        e = math.exp(uw - util)
        pw = e / (1 + math.exp(-util) + e)
        return (1 - pw) * ratio(rail, car), pw
    return mix(s, sh(W_WALK), sh(w2))


print("check against demand-plain 'Walking the whole way' (84 / 74 / 44 / 18% at 0.5/1/2/3 km, free parking, 16.5 km/h):")
for d in (0.5, 1.0, 2.0, 3.0):
    car = car_cost(d, "plain")
    bt = BETA * min(1.0, math.sqrt(60 / car))
    print(f"  {d} km: walk {100 / (1 + math.exp(-(bt * (car - walk_cost(d)) + 1.0))):.0f}%")

DS = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
CASES = [
    ("today (boarding 4, fare 1.0x)", lambda d, p: shares(d, p, board=4.0, fare_mult=1.0)),
    ("T-3856 (no boarding, 0.8x)", lambda d, p: shares(d, p)),
    ("A 10% stop walking past 0.8 km", lambda d, p: tier_a(d, p)),
    ("A 10% past 0.5 km", lambda d, p: tier_a(d, p, limit=0.5)),
    ("A 20% past 0.8 km", lambda d, p: tier_a(d, p, s=0.20)),
    ("B 10% every walk minute 3.0", lambda d, p: tier_b(d, p)),
    ("C far walk 2.5 past 10 min", lambda d, p: shares(d, p, far_w=2.5)),
    ("D head start vs car only", lambda d, p: shares(d, p, asc_rail=1.0)),
    ("A 10% + D", lambda d, p: tier_a(d, p, asc_rail=1.0)),
]

for place in ("dense", "suburb"):
    print(f"\n{place}: bus % (walk %) of all trips by straight km; car {', '.join(f'{car_cost(d, place):.0f}' for d in DS)} perceived")
    print("  bus perceived/real (T-3856):", ", ".join(f"{bus(d, 0, 0.8)[0]:.0f}/{bus(d, 0, 0.8)[1]:.0f}" for d in DS),
          "| walk:", ", ".join(f"{walk_cost(d):.0f}/{d * WALK_MIN_KM:.0f}" for d in DS))
    print(f"{'case':34s}" + "".join(f"{d:>12.1f}" for d in DS))
    for name, f in CASES:
        print(f"{name:34s}" + "".join(f"{100 * f(d, place)[0]:5.1f} ({100 * f(d, place)[1]:4.1f})" for d in DS))

# Test A's made-up train (3-minute walks at 1.5 each end, 2.5 wait, 140 km/h, free) against the car:
# rail of rail + car for a tier whose walk minutes weigh 3.0 instead (option B's risk).
print("\ntest A rail % of rail + car, suburb car, by straight km: everyone / a tier at walk weight 3.0")
for d in (0.5, 1, 2, 3, 5, 10, 20):
    car = car_cost(d, "suburb")
    ride = d / 140 * 60
    r1 = ratio(2 * 3 * 1.5 + 2.5 + ride, car)
    r2 = ratio(2 * 3 * 3.0 + 2.5 + ride, car)
    print(f"  {d:>4} km: {100 * r1:5.1f} / {100 * r2:5.1f}")
