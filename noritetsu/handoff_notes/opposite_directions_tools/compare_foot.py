"""Compare a trial build's foot.json with dist's, per country: what each line's sections
credit (owner line -> km), and what each line owns (the app's ownTrack), old vs new.

    python compare_foot.py <trial dir> cc [cc ...] [--rows N]
"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
sys.path.insert(0, str(ROOT))
import ownership


def merge(iv, km):
    tol = min(0.15 / km, 0.1) if km > 0 else 0
    iv = sorted([min(a, b), max(a, b)] for a, b in iv)
    out = []
    for lo, hi in iv:
        if out and lo <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    if out and out[0][0] <= tol:
        out[0][0] = 0
    if out and out[-1][1] >= 1 - tol:
        out[-1][1] = 1
    return out


def model(d):
    lines = json.loads((d / "lines.json").read_text(encoding="utf-8"))["lines"]
    foot = ownership.read(d / "foot.json")
    sec = {}
    for l in lines:
        shut = set(l.get("closed") or [])
        for a, b, km, g in l["sections"]:
            sec[g] = (l, a, b, km, f"{a}|{b}" in shut)
    credit = {}     # (line id, a, b) -> {owner line id: km}
    own = defaultdict(list)
    for g, (l, a, b, km, closed) in sec.items():
        f = foot.get(g, [[g, 0, 1, 0, 1]])
        cr = defaultdict(float)
        for t, fr, to, x, y in f:
            if t not in sec:
                continue
            cr[sec[t][0]["id"]] += abs(y - x) * km
            if not l.get("service") and not closed and sec[t][0]["id"] == l["id"] and not sec[t][4]:
                own[t].append((fr, to))
        credit[(l["id"], a, b)] = cr
    owned = defaultdict(float)
    for t, iv in own.items():
        l, a, b, km, _c = sec[t]
        owned[l["id"]] += km * min(1, sum(hi - lo for lo, hi in merge(iv, km)))
    return {l["id"]: l for l in lines}, credit, owned


def main():
    args = sys.argv[1:]
    rows = 30
    if "--rows" in args:
        rows = int(args[args.index("--rows") + 1])
        del args[args.index("--rows"):args.index("--rows") + 2]
    trial = Path(args[0])
    for cc in args[1:]:
        if not (trial / cc / "foot.json").exists():
            print(f"{cc}: no trial")
            continue
        L0, C0, O0 = model(ROOT / "dist" / "data" / cc)
        L1, C1, O1 = model(trial / cc)
        pair = defaultdict(lambda: [0.0, 0.0])
        cov0 = cov1 = 0.0
        for k in set(C0) | set(C1):
            a, b = C0.get(k, {}), C1.get(k, {})
            l = L1.get(k[0]) or L0.get(k[0])
            if l.get("src", "osm") == "osm" and not l.get("service"):
                cov0 += sum(a.values())
                cov1 += sum(b.values())
            for o in set(a) | set(b):
                pair[(k[0], o)][0] += a.get(o, 0.0)
                pair[(k[0], o)][1] += b.get(o, 0.0)
        own_tot0, own_tot1 = sum(O0.values()), sum(O1.values())
        ch = [(k, v) for k, v in pair.items() if abs(v[1] - v[0]) >= 0.05]
        new = [(k, v) for k, v in ch if v[0] < 0.01 and k[0] != k[1]]
        print(f"{cc}: OSM line km credited {cov0:,.1f} -> {cov1:,.1f}; owned km {own_tot0:,.1f} -> "
              f"{own_tot1:,.1f}; {len(ch)} (line, owner) pairs move >=50 m, {len(new)} owners new to a line "
              f"({sum(v[1] for k, v in new):.1f} km)")
        nm = lambda i: ((L1.get(i) or L0.get(i) or {}).get("ref") or "") + " " + ((L1.get(i) or L0.get(i) or {}).get("name") or i)
        for (x, o), (p, q) in sorted(ch, key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:rows]:
            tag = "NEW " if p < 0.01 and x != o else ("self" if x == o else "    ")
            print(f"    {tag} {nm(x)[:45]:45} -> {nm(o)[:40]:40} {p:7.2f} -> {q:7.2f} km")
        oc = [(i, O0.get(i, 0), O1.get(i, 0)) for i in set(O0) | set(O1) if abs(O1.get(i, 0) - O0.get(i, 0)) >= 0.2]
        oc.sort(key=lambda r: -abs(r[2] - r[1]))
        for i, p, q in oc[:12]:
            print(f"    owns {nm(i)[:50]:50} {p:7.2f} -> {q:7.2f} km")


if __name__ == "__main__":
    main()
