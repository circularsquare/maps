"""T-2228: what passing tracks at stations would give, by a timetable sweep (the rule the design
proposes for the sim). One direction of a line: locals stop everywhere, expresses at a few stations,
both on one double track. Locals step aside only at a passing station they stop at (or the first
station, where the order of departure is free) and wait there for every express that would otherwise
come within the spacing s of them before their next such place; elsewhere an express follows the
local ahead, s behind it. Free run times from the game's own rules (accel 1.0/0.5/0.3, brake 0.9,
dwell 30 s), checked against the sim's measured stop times (today.json).
"""
import math, json, sys

DWELL = 30.0
# at a passing station: the express may pass once the local is off the main line, and the local
# may leave once the express is past the far turnout (s)
CLEAR = 60.0

def accel(v):
    return 1.0 if v < 60 / 3.6 else 0.5 if v < 200 / 3.6 else 0.3

def run(L, vmax, step=2.0):
    """Times at positions 0..L (every `step` m) for a run from rest at 0 to rest at L."""
    n = int(L // step) + 1
    xs = [min(i * step, L) for i in range(n)] + ([L] if (n - 1) * step < L else [])
    # accel curve
    vacc = [0.0]
    for i in range(1, len(xs)):
        v = vacc[-1]
        dx = xs[i] - xs[i - 1]
        v2 = math.sqrt(v * v + 2 * accel(v) * dx)
        vacc.append(min(v2, vmax))
    ts = [0.0]
    for i in range(1, len(xs)):
        vb = lambda x: math.sqrt(max(0.0, 2 * 0.9 * (L - x)))
        va = min(vacc[i - 1], vb(xs[i - 1]))
        vb2 = min(vacc[i], vb(xs[i]))
        vm = max((va + vb2) / 2, 0.05)
        ts.append(ts[-1] + (xs[i] - xs[i - 1]) / vm)
    return xs, ts

def t_at(prof, x):
    xs, ts = prof
    if x <= 0:
        return 0.0
    if x >= xs[-1]:
        return ts[-1]
    i = int(x / (xs[1] - xs[0]))
    i = min(i, len(xs) - 2)
    f = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ts[i] + f * (ts[i + 1] - ts[i])

_CACHE = {}

def schedule(pos, stops, vmax):
    key = (tuple(pos), tuple(stops), vmax)
    if key not in _CACHE:
        _CACHE[key] = _schedule(pos, stops, vmax)
    return _CACHE[key]

def _schedule(pos, stops, vmax):
    """Free times at every station: (arrive, depart) for stops, (t, t) for passing."""
    out = {}
    t = 0.0
    for a, b in zip(stops, stops[1:]):
        prof = run(pos[b] - pos[a], vmax)
        dep = t
        out.setdefault(a, (t - DWELL if a != stops[0] else 0.0, dep))
        for k in range(a + 1, b):
            tk = dep + t_at(prof, pos[k] - pos[a])
            out[k] = (tk, tk)
        t = dep + prof[1][-1]
        out[b] = (t, t + DWELL if b != stops[-1] else t)
        t = out[b][1]
    out[stops[0]] = (0.0, 0.0)
    return out

def spacing(v):
    return 70.0 + v / 0.6

def sweep(pos, xstops, vmax, HL, HX, phase, passing, hours=4):
    """One direction over `hours`: returns mean local one-way time, mean express one-way time,
    overtakes per local trip and mean hold per overtake, over trains in the middle hours."""
    n = len(pos)
    lstops = list(range(n))
    L = schedule(pos, lstops, vmax)
    X = schedule(pos, xstops, vmax)
    s = spacing(vmax)
    span = hours * 3600
    trains = []
    for j in range(int(span // HL)):
        trains.append({"kind": "L", "t0": j * HL, "free": L, "stops": set(lstops)})
    for i in range(int(span // HX)):
        trains.append({"kind": "X", "t0": i * HX + phase, "free": X, "stops": set(xstops)})
    for tr in trains:
        tr["arr"] = {}
        tr["dep"] = {}
        tr["holds"] = 0
        tr["held"] = 0.0
    # where locals may step aside: the first station (any departure order) and passing stations
    points = [0] + [k for k in sorted(passing) if 0 < k < n - 1] + [n - 1]
    # times at the first station
    for tr in trains:
        tr["dep"][0] = tr["t0"]
        tr["arr"][0] = tr["t0"]
    for pi in range(len(points) - 1):
        a, b = points[pi], points[pi + 1]
        # at a: locals wait for expresses that would catch them before b
        locals_ = sorted([t for t in trains if t["kind"] == "L"], key=lambda t: t["dep"][a])
        exps = sorted([t for t in trains if t["kind"] == "X"], key=lambda t: t["dep"][a])
        def free_ahead(tr, frm, k):
            f = tr["free"]
            return tr["dep"][frm] + (f[k][0] - f[frm][1])
        if a > 0:
            for lo in locals_:
                normal = lo["dep"][a]
                for ex in exps:
                    xa = ex["dep"][a]
                    if xa + CLEAR <= lo["dep"][a]:
                        continue  # passed already
                    la_b = free_ahead(lo, a, b)
                    xb = free_ahead(ex, a, b)
                    if xb >= la_b + s:
                        break  # never comes within s before b: later expresses even less
                    new = xa + CLEAR
                    if new - normal > HL - s:
                        break  # would block the next local: the express follows instead
                    lo["dep"][a] = new
                if lo["dep"][a] > normal:
                    lo["holds"] += 1
                    lo["held"] += lo["dep"][a] - normal
        # run a -> b with following (order by departure from a), station by station
        order = sorted(trains, key=lambda t: t["dep"][a])
        for k in range(a + 1, b + 1):
            ahead = None
            for tr in order:
                f = tr["free"]
                prev = k - 1
                t_free = tr["dep"][prev] + (f[k][0] - f[prev][1])
                if ahead is None:
                    t_arr = t_free
                else:
                    # behind a local that steps aside here, only the clearance of its turnout
                    aside = k == b and k in passing and ahead["kind"] == "L" and tr["kind"] == "X"
                    t_arr = max(t_free, ahead["arr"][k] + (CLEAR if aside else s))
                tr["arr"][k] = t_arr
                stops_here = k in tr["stops"]
                tr["dep"][k] = t_arr + (DWELL if stops_here and k != n - 1 else 0.0)
                ahead = tr
            order = sorted(order, key=lambda t: t["dep"][k])
    mid = [t for t in trains if span * 0.25 <= t["t0"] <= span * 0.6]
    lo = [t for t in mid if t["kind"] == "L"]
    ex = [t for t in mid if t["kind"] == "X"]
    lt = sum(t["arr"][n - 1] - t["dep"][0] for t in lo) / len(lo)
    xt = sum(t["arr"][n - 1] - t["dep"][0] for t in ex) / len(ex)
    ov = sum(t["holds"] for t in lo) / len(lo)
    hold = sum(t["held"] for t in lo) / max(1, sum(t["holds"] for t in lo))
    global LAST
    LAST = trains
    return lt, xt, ov, hold, L[n - 1][0], X[n - 1][0]

CASES = {
    "shinkansen": dict(pos=[k * 30000.0 for k in range(7)], xstops=[0, 3, 6], vmax=320 / 3.6,
                       heads=[(600, 600), (1800, 600), (1200, 300)],
                       sets={"none": [], "middle (3)": [3], "two (2, 4)": [2, 4], "all five": [1, 2, 3, 4, 5]}),
    "commuter": dict(pos=[k * 2000.0 for k in range(13)], xstops=[0, 4, 8, 12], vmax=160 / 3.6,
                     heads=[(600, 600), (300, 600), (900, 900)],
                     sets={"none": [], "express stops (4, 8)": [4, 8], "every third (3, 6, 9)": [3, 6, 9],
                           "every other": [2, 4, 6, 8, 10], "all eleven": list(range(1, 12))}),
}

def main():
    res = {}
    for name, C in CASES.items():
        print("==", name)
        for HL, HX in C["heads"]:
            for label, ps in C["sets"].items():
                rows = [sweep(C["pos"], C["xstops"], C["vmax"], HL, HX, ph, ps) for ph in range(0, HX, 30)]
                best = min(rows, key=lambda r: r[0] + r[1])
                mean = [sum(r[i] for r in rows) / len(rows) for i in range(4)]
                free_l, free_x = rows[0][4], rows[0][5]
                print(f"  L{HL//60:>2} X{HX//60:>2} {label:22} free L {free_l/60:5.1f} X {free_x/60:5.1f} | best phase: L {best[0]/60:5.1f} X {best[1]/60:5.1f} overtakes {best[2]:.2f} hold {best[3]/60:.1f} | mean: L {mean[0]/60:5.1f} X {mean[1]/60:5.1f}")
                res.setdefault(name, []).append(dict(HL=HL, HX=HX, set=label, free_l=free_l, free_x=free_x, best=best[:4], mean=mean))
    json.dump(res, open(r"C:\Users\anita\projects\maps\artifacts\trainworld\t2228\overtake.json", "w"), indent=1)

if __name__ == "__main__":
    main()
