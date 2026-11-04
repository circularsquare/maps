"""T-1508: the longer-trip rule's shape under a few changes, from the market rows alone, domestic
pairs of places only (each country's places cover it). Reproduces t1508_dump's place pairs.

    python t1508_scen.py
A scenario: the ground curve times (1 - onset(lo, hi)) (it fades where the tail starts), the tail's
onset moved to (lo, hi); the tail's normaliser rescaled by the change in its domestic weight sum
(its departures conserved, `TAIL_KEEP`), the ground's kept (its trips past the fade are dropped);
then one world Rt factor set so the US's trips of 300 km and more match NHTS 2017.
"""
import json, math, collections
TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
rule = json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["market"]["rule"]
R = list(rows.values())
n = len(R)
lat = [math.radians(r["lat"]) for r in R]
lon = [math.radians(r["lon"]) for r in R]
YEAR = 425


def km(i, j):
    h = math.sin((lat[j] - lat[i]) / 2) ** 2 + math.cos(lat[i]) * math.cos(lat[j]) * math.sin((lon[j] - lon[i]) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(1, h)))


def onset(d, lo, hi):
    return min(1., max(0., (d - lo) / (hi - lo)))


def ground(d):
    on = onset(d, *rule["onset_km"])
    return 0. if on == 0 else on * max(d, rule["d0_km"]) ** -rule["A"]


def tail(d, lo, hi):
    on = onset(d, lo, hi)
    return 0. if on == 0 else on * max(d, rule["D1_km"]) ** -rule["G"]


by_c = collections.defaultdict(list)
for k, r in enumerate(R):
    by_c[r["country"]].append(k)
D = {}
for c, ids in by_c.items():
    for i in ids:
        for j in ids:
            if i != j:
                D[(i, j)] = km(i, j)

WATCH = [("sha", "bei"), ("sha", "gua"), ("bei", "gua"), ("ber", "mun"), ("ber", "fra"), ("ber", "ham"), ("seo", "bus"), ("seo", "jej"),
         ("tyo", "osa"), ("tyo", "fuk"), ("tyo", "spk"), ("nyc", "wdc"), ("nyc", "bos"), ("nyc", "cic"), ("nyc", "lsn"), ("par", "lyo"), ("par", "mar"), ("mad", "bar"), ("lon", "cit")]
idx = {r["zone"]: k for k, r in enumerate(R)}
# NHTS 2017, US, person-trips a person a day by band (T-376's targets), 300 km and over
NHTS = {300: 0.00256 + 0.00117 + 0.00150, 700: 0.00123 + 0.00128, 1500: 0.00135 + 0.00071}


def run(name, glo, ghi, tlo, thi, fit=True):
    # tail normaliser change per origin: domestic weight sum new over old (abroad held)
    scale = [1.] * n
    for c, ids in by_c.items():
        for i in ids:
            old = sum(R[j]["ta"] * tail(D[(i, j)], *rule["tail_onset_km"]) for j in ids if j != i)
            new = sum(R[j]["ta"] * tail(D[(i, j)], tlo, thi) for j in ids if j != i)
            # the share abroad of i's tail weight is not known here; the domestic sum stands for it
            scale[i] = old / new if new > 0 and old > 0 else 1.
    band = collections.defaultdict(lambda: [0., 0., 0., 0.])
    pair = collections.defaultdict(lambda: [0., 0.])
    for (i, j), d in D.items():
        if d < 100:
            continue
        g = R[i]["g"] * ground(d) * (1 - (onset(d, glo, ghi) if glo else 0.)) * R[j]["a"]
        rel = min(R[j]["inc"] / R[i]["inc"], 1.) ** rule["eps_t"]
        t = R[i]["t"] * scale[i] * tail(d, tlo, thi) * R[j]["ta"] * rel
        b = 0 if d < 300 else 1 if d < 700 else 2 if d < 1500 else 3
        x = band[(R[i]["country"], b)]
        x[0] += g; x[1] += t
        p = pair[(min(i, j), max(i, j))]
        p[0] += g; p[1] += t
    us = {k: band[("US", b)] for k, b in ((300, 1), (700, 2), (1500, 3))}
    pop_us = sum(R[k]["pop"] for k in by_c["US"])
    # fit: one factor on Rt so US 300 km+ (all of it, ground + tail) matches NHTS's 300 km+
    target = sum(NHTS.values()) * pop_us
    g300 = sum(v[0] for v in us.values())
    t300 = sum(v[1] for v in us.values())
    k = max((target - g300) / t300, 0.) if fit else 1.
    print(f"## {name}: ground fades {glo}-{ghi}, tail onset {tlo}-{thi}; Rt x {k:.2f}")
    print("US vs NHTS, trips a person a day: " + ", ".join(f"{b}+ {(us[b][0] + k * us[b][1]) / pop_us:.5f} ({NHTS[b]:.5f})" for b in (300, 700, 1500)))
    for c in ("CN", "JP", "KR", "DE", "FR", "GB", "ES"):
        pop = sum(R[q]["pop"] for q in by_c[c])
        v = [band[(c, b)][0] + k * band[(c, b)][1] for b in range(4)]
        print(f"  {c}: a year 100-300 {v[0]*YEAR/1e6:.0f}M, 300-700 {v[1]*YEAR/1e6:.0f}M, 700-1500 {v[2]*YEAR/1e6:.0f}M, 1500+ {v[3]*YEAR/1e6:.0f}M; 300+ a person a weekday {sum(v[1:])/pop:.4f}")
    out = []
    for a, b in WATCH:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            p = pair[(min(i, j), max(i, j))]
            out.append(f"{a}-{b} {(p[0] + k * p[1]) / 1e3:.1f}k")
    print("  pairs, both ways a weekday: " + "; ".join(out))
    return k


run("as built", None, None, 300, 700, fit=False)
run("as built, Rt refit to NHTS", None, None, 300, 700)
run("ground fades 300-700", 300, 700, 300, 700)
run("both over 150-400", 150, 400, 150, 400)
run("both over 100-300", 100, 300, 100, 300)
