"""T-1541: tables from the access diagnosis outputs (../artifacts/trainworld/T-1541/<case>_<variant>.json).

    python trainworld/notes/T-1541/t1541_table.py shares case:variant ...    corridor shares, a row each
    python trainworld/notes/T-1541/t1541_table.py rings case:variant pair    rail by the ends' distance
    python trainworld/notes/T-1541/t1541_table.py parts case:variant pair    sample zones' perceived minutes
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "artifacts", "trainworld", "T-1541"))


def load(cv):
    c, v = cv.split(":")
    return json.load(open(os.path.join(OUT, f"{c}_{v}.json")))


def pct(x):
    return f"{100 * x:.0f}%"


def shares(jobs):
    print("| Case | Pair | rail / air / car of all (day) | rail of rail + air | midday: trips on served zone pairs | rail of rail + air on served pairs | midday rebuilt: rail / air / car, rail of rail + air |")
    print("|---|---|---|---|---|---|---|")
    for cv in jobs:
        for p in load(cv)["pairs"]:
            if p.get("missing"):
                continue
            d, md = p["day"], p["midday"]
            r = d["rail"] / d["trips"]
            air, car = (1 - r) * d["air_of_nonrail"], (1 - r) * d["car_of_nonrail"]
            served = md["served_trips"] / md["trips"]
            # rail and air on served pairs: rail all there; air on served pairs is the place split
            rs = md["rail"] / md["served_trips"] if md["served_trips"] else 0
            ars = (1 - rs) * d["air_of_nonrail"]
            t = md["trips"]
            mr, ma, mc = md["rail"] / t, md["air"] / t, md["car"] / t
            mid = f"{pct(mr)} / {pct(ma)} / {pct(mc)}, {pct(mr / (mr + ma)) if mr + ma else '-'}"
            print(f"| {cv} | {'-'.join(p['pair'])} | {pct(r)} / {pct(air)} / {pct(car)} | {pct(r / (r + air)) if r + air else '-'} | {pct(served)} | {pct(rs / (rs + ars)) if rs + ars else '-'} | {mid} |")


def rings(cv, pair):
    p = next(x for x in load(cv)["pairs"] if "-".join(x["pair"]) == pair)
    rs = sorted({r["ra"] for r in p["rings"]} | {r["rb"] for r in p["rings"]})
    edges = [0, 5, 10, 15, 25, 40]
    lab = lambda e: f"{e:g}-{edges[edges.index(e) + 1]:g} km" if e != 40 else "40+ km"
    tot = sum(r["trips"] for r in p["rings"])
    print(f"{cv} {pair}: rail of rail + air (share of the corridor's trips) by the A end's (rows) and B end's (columns) distance from the line's station")
    print("| A \\ B | " + " | ".join(lab(e) for e in rs) + " | all |")
    print("|---" * (len(rs) + 2) + "|")
    for ra in rs + ["all"]:
        cells = []
        for rb in rs + ["all"]:
            sel = [r for r in p["rings"] if (ra == "all" or r["ra"] == ra) and (rb == "all" or r["rb"] == rb)]
            t = sum(r["trips"] for r in sel)
            rl, ai = sum(r["rail"] for r in sel), sum(r["air"] for r in sel)
            cells.append(f"{pct(rl / (rl + ai)) if rl + ai > 0 else '-'} ({pct(t / tot)})" if t > 0 else "")
        print(f"| {lab(ra) if ra != 'all' else 'all'} | " + " | ".join(cells) + " |")
    pr = p["pop_by_ring"]
    print("people by ring, A:", [f"{x / 1e6:.1f}M" for x in pr["A"]], "B:", [f"{x / 1e6:.1f}M" for x in pr["B"]])


def parts(cv, pair):
    p = next(x for x in load(cv)["pairs"] if "-".join(x["pair"]) == pair)
    print(f"{cv} {pair} (ride {p['ride_min']:.0f} min on the line); perceived minutes, middle income group, midday")
    print("| Zone (end, km from station, people) | access | egress | wait + ride | fare | rail best pair (logsum) | car: time + money | plane: time + money | rail / air / car |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in p["samples"]:
        d = s["detail"]
        car, air = d["car"], d["air"]
        g = d["group_shares"]
        lab = f"{s['end']} {s['zone_km']:.0f} km, {s['pop'] / 1e3:.0f}k"
        if "rail" in d:
            r = d["rail"]
            rail = f"| {r['access']:.0f} | {r['egress']:.0f} | {r['wait_ride']:.0f} | {r['fare']:.0f} | {r['best_total']:.0f} ({r['logsum']:.0f})"
        else:
            rail = "| no rail | | | | " + ("not served" if not d["served"] else "no route")
        print(f"| {lab} {rail} | {car['min']:.0f} + {car['money']:.0f} = {car['total']:.0f} | {air['min']:.0f} + {air['money']:.0f} = {air['total']:.0f} | {pct(g[0])} / {pct(g[1])} / {pct(g[2])} |")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "shares":
        shares(sys.argv[2:])
    elif what == "rings":
        rings(sys.argv[2], sys.argv[3])
    else:
        parts(sys.argv[2], sys.argv[3])
