"""T-1546: rebuild the midday choice of every zone pair of each focus corridor from the dumps
(t1546_test.rs, ../artifacts/trainworld/T-1546/<case>.json) under rules for the plane and rail, without
a new solve. The built rule reproduces `demand` (PairRule::alt, rail_logsum_x, mode_beta).

    python trainworld/notes/T-1546/t1546_eval.py table <case,case,...> <variant,variant,...>
    python trainworld/notes/T-1546/t1546_eval.py parts <case> <pair> <variant>
    python trainworld/notes/T-1546/t1546_eval.py rings <case> <pair> <variant>
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "artifacts", "trainworld", "T-1546"))

PUB = {"sha-bei": 86, "bei-gua": 25, "tyo-osa": 85, "tyo-fuk": 11, "par-lyo": 90, "par-mar": 77, "mad-bar": 78,
       "lon-cit": 45, "nyc-wdc": 79, "nyc-bos": 53}
PUB_ALL = {"tyo-osa": (64, 11, 25), "mad-bar": (64, 18, 18), "nyc-wdc": (20, 5, 64), "nyc-bos": (5, 4, 87)}
NAMES = {"sha": "Shanghai", "bei": "Beijing", "gua": "Guangzhou", "tyo": "Tokyo", "osa": "Osaka", "fuk": "Fukuoka", "ngo": "Nagoya",
         "par": "Paris", "lyo": "Lyon", "mar": "Marseille", "mad": "Madrid", "bar": "Barcelona", "lon": "London", "cit": "Edinburgh",
         "nyc": "New York", "wdc": "Washington", "bos": "Boston"}

# built constants (intercity.rs)
TERM, WAIT, KMH, FIXED, NODIRECT = 90., 25., 750., 30., 60.

# A variant: changes to the plane's minutes and to rail. Keys:
#   acc: "place" (built: the place's people-weighted mean drive per airport) or "zone" (the zone's own)
#   acc_f, acc_add: the drive at the two ends counted acc_f x minutes + acc_add per end
#   term: minutes at the airports (both ends); wait: the wait for a flight; fixed: the flight's fixed minutes
#   air_k: the plane's minutes beyond flying (access, airports, wait) all times this
#   rail_ivt: rail's ride on the focus line counted this (1 built)
VARIANTS = {
    "built": {},
    "zone": {"acc": "zone"},
    "term135": {"term": 135.},
    "term135z": {"term": 135., "acc": "zone"},
    "ovt15": {"air_k": 1.5},
    "ovt15z": {"air_k": 1.5, "acc": "zone"},
    "acc2": {"acc_f": 2.},
    "acc2z": {"acc_f": 2., "acc": "zone"},
    "ride_z": {"acc": "zone", "acc_f": 2.66, "acc_add": 25.},
    "wait10": {"wait": 10.},
    "rivt08": {"rail_ivt": 0.8},
    "rfare134": {"rail_fare": 1.34},
    "rfare134z": {"rail_fare": 1.34, "acc": "zone"},
    "t135z_rf": {"term": 135., "acc": "zone", "rail_fare": 1.34},
    "t135z_w10": {"term": 135., "acc": "zone", "wait": 10.},
    "t135_rf": {"term": 135., "rail_fare": 1.34},
    "t120": {"term": 120.},
    "t120z": {"term": 120., "acc": "zone"},
}


def sat(x):
    return max(-80., min(80., x))


def load(case):
    return json.load(open(os.path.join(OUT, f"{case}.json")))


def plane(v, rule, pf, pt, zf, zt):
    """The plane's (minutes, usd) for a zone pair, rule `rule`, from place pf to pt, zones zf, zt."""
    if not math.isfinite(rule["fl"][0]):
        return None
    km = rule["km"]
    best = None
    for k, (x, xd) in enumerate(pf["air"]):
        for l, (y, yd) in enumerate(pt["air"]):
            direct = xd or yd
            if v.get("acc", "place") == "zone":
                a = zf["air_min"][k] + zt["air_min"][l]
            else:
                a = x + y
            # the airport pair is chosen on plain minutes, as built
            plain = a + TERM + WAIT + km / KMH * 60 + FIXED + (0 if direct else NODIRECT)
            acc = v.get("acc_f", 1.) * a + 2 * v.get("acc_add", 0.)
            over = acc + v.get("term", TERM) + v.get("wait", WAIT)
            over *= v.get("air_k", 1.)
            m = over + km / KMH * 60 + v.get("fixed", FIXED) + (0 if direct else NODIRECT)
            if best is None or plain < best[0]:
                best = (plain, m, acc, direct)
    return best[1], rule["fl"][1], best[2]


def eval_pair(d, q, v, detail=False):
    """Rail, air and car trips of a corridor (midday) under variant v; optionally per row."""
    mp, pp = d["mp"], d["p"]
    mb = lambda c: mp["beta"] * math.sqrt(min(1., mp["damp_min"] / c))
    rb = lambda c: pp["beta"] * math.sqrt(min(1., pp["damp_min"] / c))
    zones = q["zones"]
    A, B = q["places"]["A"], q["places"]["B"]
    tot = [0., 0., 0., 0.]
    rows = []
    for za, zb, allt, car_min, car_usd, rail, parts in q["rows"]:
        zf, zt = zones[str(za)], zones[str(zb)]
        ab = zf["place"] == q["pa"]
        pf, pt = (A, B) if ab else (B, A)
        rule = q["rules"]["AB" if ab else "BA"]
        pl = plane(v, rule, pf, pt, zf, zt)
        rr = [0., 0., 0., 0.]
        for k, (share, fgf) in enumerate(pf["groups"]):
            w = pf["w"] * fgf
            trips = allt * share
            car_m = car_min + car_usd * w
            if rule["car"] and pl:
                s = rule["split"][0] + rule["split"][1] * w
                fl = pl[0] + pl[1] * w
                vv = math.exp(sat(-mb(s) * (fl - s)))
                drive, rest, air = car_m, 1 + vv, vv / (1 + vv)
            elif pl:
                drive, rest, air = pl[0] + pl[1] * w, 1., 1.
            else:
                drive, rest, air = car_m, 1., 0.
            rc = rail[k]
            if rc is not None:
                rc += (v.get("rail_ivt", 1.) - 1.) * q["ride_min"]
                if parts:
                    # rail's fare times rail_fare: the middle group's fare part, scaled to this group
                    mid = pf["groups"][len(pf["groups"]) // 2][1]
                    rc += (v.get("rail_fare", 1.) - 1.) * parts[3] * fgf / mid
                vr = math.exp(sat(-rb(car_m) * (rc - drive)))
            else:
                vr = 0.
            f = vr / (rest + vr)
            rr[0] += trips
            rr[1] += trips * f
            rr[2] += trips * (1 - f) * air
            rr[3] += trips * (1 - f) * (1 - air)
        for i in range(4):
            tot[i] += rr[i]
        if detail:
            rows.append((za, zb, rr, pl, parts, rail))
    return tot, rows


def lo(p):
    p = min(max(p, .005), .995)
    return math.log(p / (1 - p))


def table(cases, variants):
    # results cached per case and variant definition (the dumps are large and slow to rebuild)
    cpath = os.path.join(OUT, "eval_cache.json")
    try:
        cache = json.load(open(cpath))
    except Exception:
        cache = {}
    res, last, keys = {}, {}, []
    for c in cases:
        todo = [vn for vn in variants if f"{c}|{json.dumps(VARIANTS[vn], sort_keys=True)}" not in cache]
        if todo:
            d = load(c)
            for vn in todo:
                cache[f"{c}|{json.dumps(VARIANTS[vn], sort_keys=True)}"] = {"-".join(q["pair"]): eval_pair(d, q, VARIANTS[vn])[0] for q in d["pairs"] if not q.get("missing")}
            del d
            json.dump(cache, open(cpath, "w"))
        for vn in variants:
            for key, t in cache[f"{c}|{json.dumps(VARIANTS[vn], sort_keys=True)}"].items():
                res[(key, vn, c)] = t
                last[(key, vn)] = t
                if (key, c) not in keys:
                    keys.append((key, c))
    print("| Pair | case | " + " | ".join(variants) + " | Published |")
    print("|---|---|" + "---|" * (len(variants) + 1))
    for k, c in keys:
        cells = []
        for vn in variants:
            t = res[(k, vn, c)]
            r, a, cc = (x / t[0] for x in t[1:])
            cells.append(f"{100*r:.0f}/{100*a:.0f}/{100*cc:.0f}, {100*r/(r+a) if r+a else 0:.0f}%")
        a, b = k.split("-")
        pub = f"{PUB[k]}%" + (f" ({'/'.join(map(str, PUB_ALL[k]))})" if k in PUB_ALL else "") if k in PUB else ""
        print(f"| {NAMES[a]} - {NAMES[b]} | {c} | " + " | ".join(cells) + f" | {pub} |")
    for vn in variants:
        e = []
        for k in PUB:
            if (k, vn) in last:
                t = last[(k, vn)]
                e.append(abs(lo(t[1] / (t[1] + t[2]) if t[1] + t[2] else 0) - lo(PUB[k] / 100)))
        print(f"{vn}: error {sum(e) / max(len(e), 1):.2f} over {len(e)} pairs (the last case given for each)")


def parts(case, pair, vn):
    d = load(case)
    q = next(x for x in d["pairs"] if "-".join(x["pair"]) == pair)
    v = VARIANTS[vn]
    zones = q["zones"]
    A, B = q["places"]["A"], q["places"]["B"]
    print(f"{case} {pair} {vn}; ride {q['ride_min']:.0f}; places' airport drives A {[round(x) for x, _ in A['air']]} B {[round(x) for x, _ in B['air']]}")
    _, rows = eval_pair(d, q, v, detail=True)
    # the station zone of each end, then the most populous zones near 7.5, 12.5, 20, 32 km, each to the other end's station zone
    zs = {e: sorted([(int(z), zz) for z, zz in zones.items() if zz["place"] == q["pa" if e == "A" else "pb"]], key=lambda t: t[1]["st_km"]) for e in "AB"}
    pick = {}
    for e in "AB":
        L = [zs[e][0]]
        for want in [7.5, 12.5, 20, 32]:
            c = [t for t in zs[e] if abs(t[1]["st_km"] - want) < 2.5]
            if c:
                L.append(max(c, key=lambda t: t[1]["pop"]))
        pick[e] = L
    by = {(r[0], r[1]): r for r in rows}
    w = A["w"] * A["groups"][len(A["groups"]) // 2][1]
    print(f"middle group: {w:.2f} minutes a dollar")
    print("| Zone | own airport drives | access | egress | wait + ride | fare | rail | plane: drive / time / money | rail / air / car |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e, other in (("A", "B"), ("B", "A")):
        for z, zz in pick[e]:
            o = pick[other][0][0]
            key = (z, o) if e == "A" else (o, z)
            za, zb, rr, pl, pa_, rail = by[key]
            ww = (A if e == "A" else B)["w"]
            lab = f"{e} {zz['st_km']:.0f} km, {zz['pop']/1e3:.0f}k"
            dr = "/".join(f"{x:.0f}" for x in zz["air_min"])
            if pa_:
                rs = f"{pa_[0]:.0f} | {pa_[1]:.0f} | {pa_[2]:.0f} | {pa_[3]:.0f} | {rail[1]:.0f}"
            else:
                rs = "| | | | no rail"
            pls = f"{pl[2]:.0f} / {pl[0]:.0f} / {pl[1]*w:.0f}" if pl else "-"
            t = rr[0]
            print(f"| {lab} | {dr} | {rs} | {pls} | {100*rr[1]/t:.0f} / {100*rr[2]/t:.0f} / {100*rr[3]/t:.0f} |")


def rings(case, pair, vn):
    d = load(case)
    q = next(x for x in d["pairs"] if "-".join(x["pair"]) == pair)
    _, rows = eval_pair(d, q, VARIANTS[vn], detail=True)
    zones = q["zones"]
    edges = [0, 5, 10, 15, 25, 40, 1e9]
    ring = lambda k: max(i for i, e in enumerate(edges) if k >= e)
    g = {}
    for za, zb, rr, *_ in rows:
        ka, kb = zones[str(za)], zones[str(zb)]
        a, b = (ka, kb) if ka["place"] == q["pa"] else (kb, ka)
        key = (ring(a["st_km"]), ring(b["st_km"]))
        x = g.setdefault(key, [0., 0., 0., 0.])
        for i in range(4):
            x[i] += rr[i]
    T = sum(x[0] for x in g.values())
    for key in sorted(g):
        x = g[key]
        print(f"A {edges[key[0]]}-{edges[key[0]+1]} B {edges[key[1]]}-{edges[key[1]+1]}: {100*x[0]/T:.0f}% of trips, rail of rail+air {100*x[1]/(x[1]+x[2]) if x[1]+x[2] else 0:.0f}%, r/a/c {100*x[1]/x[0]:.0f}/{100*x[2]/x[0]:.0f}/{100*x[3]/x[0]:.0f}")


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "table":
        table(sys.argv[2].split(","), sys.argv[3].split(","))
    elif w == "parts":
        parts(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        rings(sys.argv[2], sys.argv[3], sys.argv[4])
