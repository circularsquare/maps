"""T-551: tables from the ceiling test's output (notes/T-551.md).

    venv\\Scripts\\python.exe trainworld/notes/t551_tables.py artifacts/trainworld/T-551-nyc.json artifacts/trainworld/T-551-lon.json > artifacts/trainworld/T-551-tables.txt
"""
import json
import sys

import numpy as np

BANDS = ["0-2", "2-5", "5-10", "10-20", "20-50", "50+"]
SC = [("ceiling_a", "a: 140 km/h, free"), ("ceiling_b", "b: 100 km/h, walks, wait, fare"), ("real", "c: real network")]


def arr(x):
    return np.array(x, dtype=float)  # layers x bands x [trips, walk, rail]


def pct(a, b):
    return f"{100 * a / b:5.1f}" if b > 0 else "    -"


def summary(r, name):
    print(f"\n### {name}: shares of all trips (walk / rail / road), and rail of trips that do not walk")
    print(f"{'scenario':34s} {'layer':9s} {'trips M':>8s} {'walk':>6s} {'rail':>6s} {'road':>6s} {'rail/nw':>8s}")
    lev = r["levers"]["base"]
    for key, lab in SC:
        a = lev[key]
        if a is None:
            continue
        a = arr(a)
        for ln, sel in [("commute", a[:1]), ("everyday", a[1:]), ("all", a)]:
            t, w, rl = sel[..., 0].sum(), sel[..., 1].sum(), sel[..., 2].sum()
            print(f"{lab:34s} {ln:9s} {t / 1e6:8.2f} {pct(w, t)} {pct(rl, t)} {pct(t - w - rl, t)} {pct(rl, t - w):>8s}")


def by_band(r, name):
    print(f"\n### {name}: by straight-line distance (km), all purposes: share of trips in the band, walk, then rail of all / of non-walk")
    lev = r["levers"]["base"]
    hdr = f"{'scenario':34s}" + "".join(f"{b:>16s}" for b in BANDS)
    print(hdr)
    base = arr(lev["ceiling_a"]).sum(axis=0)
    tt = base[:, 0].sum()
    print(f"{'trips in band %':34s}" + "".join(f"{pct(base[b, 0], tt):>16s}" for b in range(6)))
    print(f"{'walk %':34s}" + "".join(f"{pct(base[b, 1], base[b, 0]):>16s}" for b in range(6)))
    for key, lab in SC:
        a = lev[key]
        if a is None:
            continue
        a = arr(a).sum(axis=0)
        print(f"{lab:34s}" + "".join(f"{pct(a[b, 2], a[b, 0])}/{pct(a[b, 2], a[b, 0] - a[b, 1]):>9s}" for b in range(6)))
    for ln, sl in [("commute", slice(0, 1)), ("everyday", slice(1, 7))]:
        print(f"  {ln} only, rail of non-walk:")
        for key, lab in SC:
            a = lev[key]
            if a is None:
                continue
            a = arr(a)[sl].sum(axis=0)
            print(f"  {lab:32s}" + "".join(f"{pct(a[b, 2], a[b, 0] - a[b, 1]):>16s}" for b in range(6)))


def by_purpose(r, name):
    print(f"\n### {name}: by purpose, rail of all trips / of non-walk")
    lev = r["levers"]["base"]
    print(f"{'scenario':34s}" + "".join(f"{l:>14s}" for l in r["layers"]))
    for key, lab in SC:
        a = lev[key]
        if a is None:
            continue
        a = arr(a).sum(axis=1)
        print(f"{lab:34s}" + "".join(f"{pct(a[l, 2], a[l, 0])}/{pct(a[l, 2], a[l, 0] - a[l, 1]):>8s}" for l in range(7)))


def levers(r, name):
    print(f"\n### {name}: levers, rail share of all trips (and of non-walk), ceiling b and the real network")
    print(f"{'lever':10s} {'b all':>7s} {'b nw':>7s} {'c all':>7s} {'c nw':>7s} {'c commute nw':>13s} {'c everyday nw':>14s} {'c nw 10-50 km':>14s}")
    for k, lev in r["levers"].items():
        b = arr(lev["ceiling_b"])
        bt, bw, br = b[..., 0].sum(), b[..., 1].sum(), b[..., 2].sum()
        line = f"{k:10s} {pct(br, bt):>7s} {pct(br, bt - bw):>7s}"
        if lev["real"] is not None:
            c = arr(lev["real"])
            ct, cw, cr = c[..., 0].sum(), c[..., 1].sum(), c[..., 2].sum()
            cc = c[0]
            ce = c[1:].sum(axis=0)
            mid = c[:, 3:5].sum(axis=(0, 1))
            line += (f" {pct(cr, ct):>7s} {pct(cr, ct - cw):>7s} {pct(cc[:, 2].sum(), cc[:, 0].sum() - cc[:, 1].sum()):>13s}"
                     f" {pct(ce[:, 2].sum(), ce[:, 0].sum() - ce[:, 1].sum()):>14s} {pct(mid[2], mid[0] - mid[1]):>14s}")
        print(line)


def car(r, name):
    print(f"\n### {name}: the car as modelled, trips that do not walk, by straight-line band")
    print(f"{'layer':9s} {'band':6s} {'drive min':>9s} {'road km':>8s} {'str km':>7s} {'road km/h':>9s} {'str km/h':>9s} {'park pmin':>9s} {'paid pmin':>9s} {'cordon':>7s} {'park $':>7s}")
    c = arr(r["levers"]["base"]["car"])
    for ln, sel in [("commute", c[:1]), ("everyday", c[1:])]:
        s = sel.sum(axis=0)
        for b in range(6):
            t = s[b, 0]
            if t <= 0:
                continue
            m, rk, sk, pk, paid, cord = (s[b, 1:] / t)
            print(f"{ln:9s} {BANDS[b]:6s} {m:9.1f} {rk:8.1f} {sk:7.1f} {rk / m * 60:9.1f} {sk / m * 60:9.1f} {pk:9.1f} {paid:9.1f} {cord:7.1f} {pk / 3:7.2f}")


def examples(r, name):
    ex = r["levers"]["base"]["examples"]
    if not ex:
        return
    asc = r["levers"]["base"]["asc"][0]
    print(f"\n### {name}: example commutes in the morning peak (perceived minutes unless marked; asc {asc})")
    for e in ex:
        v = e["trip"]
        if v[0] != 1:
            print(f"{e['name']}: no answer ({v[0]})")
            continue
        head = v[:32]
        line = (f"{e['name']}: {head[6] / 1000:.1f} km straight; walk share {head[8]:.2f}; car {head[9]:.1f} real min "
                f"= {head[10]:.1f} perceived + parking {head[11]:.1f} = {head[12]:.1f}")
        if head[13] == 1:
            line += (f"; rail offered: {head[14]:.1f} (home end {head[20]:.1f}, work end {head[21]:.1f}, base fare {head[22]:.1f});"
                     f" util {head[16]:.2f}; rail of non-walk {head[15] * 100:.1f}% (this script's own: {e['check_r'] * 100:.1f}%)")
            npairs, nroutes = int(head[24]), int(head[25])
            body = v[32:]
            pairs = [body[k * 14:(k + 1) * 14] for k in range(npairs)]
            if pairs:
                p0 = pairs[0]
                line += (f"\n    top station pair ({p0[9] * 100:.0f}% of riders): access {'ride' if p0[12] else 'walk'} {p0[4]:.0f} m,"
                         f" {p0[5]:.1f} perceived ({p0[10]:.1f} real); egress {p0[6]:.0f} m, {p0[7]:.1f} perceived; station to station {p0[8]:.1f}")
            rest = body[npairs * 14:]
            k = 0
            for ri in range(min(nroutes, 2)):
                pi, share, nl = rest[k], rest[k + 1], int(rest[k + 2])
                k += 3
                legs = []
                for li in range(nl):
                    L = rest[k:k + 12]
                    k += 12
                    if L[0] == 1:
                        legs.append(f"ride {L[5]:.1f} min ({L[6]:.1f} p, fare {L[7]:.1f} p, wait {L[8]:.1f} min = {L[9]:.1f} p, {int(L[10])} stops)")
                    else:
                        legs.append(f"walk {L[5]:.1f} min ({L[6]:.1f} p)")
                line += f"\n    route {ri + 1} ({share * 100:.0f}%): " + "; ".join(legs)
        else:
            line += f"; rail not offered (reason {int(head[26])})"
        if e.get("park"):
            fx, paid, free, dens = e["park"]
            line += f"\n    destination zone: {dens:,.0f} jobs/km2 of land; every driver {fx:.1f}, a paid space {paid:.1f} more, {free * 100:.0f}% park free"
        print(line)


def checks(r, name):
    print(f"\n### {name}: checks (rail by pair summed against the model's own totals)")
    for c in r["checks"]:
        print(f"{c['lever']:10s} commute {c['commute_rail'][0]:,.0f} vs {c['commute_rail'][1]:,.0f}; everyday {c['everyday_rail'][0]:,.0f} vs {c['everyday_rail'][1]:,.0f}")


for path in sys.argv[1:]:
    r = json.load(open(path))
    name = r["city"]
    summary(r, name)
    by_band(r, name)
    by_purpose(r, name)
    levers(r, name)
    car(r, name)
    examples(r, name)
    checks(r, name)
