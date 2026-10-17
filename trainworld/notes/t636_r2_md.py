import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t636_tables import agg, load, modes  # noqa

S = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
AL = list(range(7))
CITIES = [("nyc", "New York"), ("lon", "London")]
BANDS = ["0-2", "2-5", "5-10", "10-20", "20-50", "50+"]
R1 = [("today", "Today"), ("simple", "Simple"), ("b075", "Simple, beta 0.075"), ("b100", "Simple, beta 0.10"),
      ("d3a", "Simple, beta 0.075, damp 26.7"), ("d3b", "Simple, beta 0.10, damp 15"),
      ("w15", "Simple, walk constant 1.5"), ("b100w15", "Simple, beta 0.10, walk 1.5"), ("d3aw15", "Simple, beta 0.075, damp 26.7, walk 1.5")]
R2 = [("today", "Today"), ("simple", "Simple"), ("b100", "Simple, beta 0.10"), ("newp3", "New model, parking base 3"), ("new", "New model")]


def f1(x):
    return f"{100 * x:.1f}"


def rc(m):
    return m[0] / (m[0] + m[2])


def R(c, n):
    return load(f"{S}/out/{c}_{n}.json")


def score(c, n):
    return json.load(open(f"{S}/calib_out/{c}_score_t636_{n}.json"))


def walkband(r, k):
    a = r[k]
    return [agg(a, AL, [b])[2] / agg(a, AL, [b])[0] for b in range(a.shape[1])]


def table_rc(names):
    out = ["| Setting | NY A | NY B | NY C | London A | London B | London C |", "|---|---|---|---|---|---|---|"]
    for n, label in names:
        row = [label]
        for c, _ in CITIES:
            r = R(c, n)
            row += [f1(rc(modes(r, k, AL))) for k in "ABC"]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def table_round2():
    out = []
    w = out.append
    w("| Setting | NY A | NY B | NY C | NY stations (x counts) | London A | London B | London C | London stations (x counts) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n, label in R2:
        row = [label]
        for c, _ in CITIES:
            r = R(c, n)
            sc = None
            try:
                sc = score(c, n)
            except Exception:
                pass
            row += [f"{f1(rc(modes(r, k, AL)))} ({f1(modes(r, k, AL)[0])})" for k in "ABC"]
            row.append(f"x{np.exp(sc['station_level']):.2f}" if sc else "")
        w("| " + " | ".join(row) + " |")
    return "\n".join(out)


def table_modes(c):
    out = ["| Setting | A | B | C |", "|---|---|---|---|"]
    for n, label in R2:
        r = R(c, n)
        out.append(f"| {label} | " + " | ".join(" / ".join(f1(x) for x in modes(r, k, AL)) for k in "ABC") + " |")
    return "\n".join(out)


NHTS = [57.9, 4.5, 1.2, 0.7, 0.5, 0.0]  # all legs, New York CBSA, weekday (artifacts/trainworld/T-551-nhts.txt)
NHTS_RC = [3.9, 10.4, 23.4, 28.1, 22.8, 14.1]  # rail / (rail + car), same file: 1.4/(1.4+34.3) and so on


def table_walk_bands():
    out = ["| km | " + " | ".join(BANDS) + " |", "|---|" + "---|" * 6]
    r0 = R("nyc", "today")
    sh = [agg(r0["A"], AL, [b])[0] / agg(r0["A"], AL)[0] for b in range(6)]
    out.append("| Share of the model's trips | " + " | ".join(f"{100 * x:.0f}" for x in sh) + " |")
    out.append("| NHTS 2017, walk | " + " | ".join(f"{x:.1f}" for x in NHTS) + " |")
    for n, label in R2:
        r = R("nyc", n)
        for k in "CAB":
            out.append(f"| {label}, {k} | " + " | ".join(f1(x) for x in walkband(r, k)) + " |")
    out.append("| NHTS 2017, rail of rail + car | " + " | ".join(f"{x:.1f}" for x in NHTS_RC) + " |")
    for n, label in R2[-1:]:
        r = R("nyc", n)
        a = r["C"]
        out.append(f"| {label}, C, rail of rail + car | " + " | ".join(
            f1(agg(a, AL, [b])[1] / (agg(a, AL, [b])[1] + (agg(a, AL, [b])[0] - agg(a, AL, [b])[1] - agg(a, AL, [b])[2]))) for b in range(6)) + " |")
    return "\n".join(out)


def table_purpose():
    out = ["| Layer | Today, C | New model, C | Survey |", "|---|---|---|---|"]
    names = ["Commutes", "Errands", "Social", "School", "University", "Airport", "Medical"]
    surv = {"nyc": {0: "6.2 (ACS commutes)", 1: "26.0 (NHTS)", 2: "35.1 (NHTS)", 6: "12.0 (NHTS)"},
            "lon": {0: "9.4 (census commutes)", 1: "42.3 (LTDS shopping and personal business)", 2: "41.0 (LTDS leisure)", 3: "43.8 (LTDS education)"}}
    res = []
    for c, cl in CITIES:
        out2 = [f"**{cl}**, walk share of the layer's trips (%), the real network:", "", "| Layer | Today | New model | Survey |", "|---|---|---|---|"]
        for l in range(7):
            vals = []
            for n in ("today", "new"):
                r = R(c, n)
                t = agg(r["C"], [l])
                vals.append(f1(t[2] / t[0]))
            out2.append(f"| {names[l]} | {vals[0]} | {vals[1]} | {surv[c].get(l, '')} |")
        for n, lab in (("today", "all trips, today"), ("new", "all trips, new model")):
            m = modes(R(c, n), "C", AL)
        res.append("\n".join(out2))
    return "\n\n".join(res)


if __name__ == "__main__":
    which = sys.argv[1]
    print({"rc1": lambda: table_rc(R1), "r2": table_round2, "nyc_modes": lambda: table_modes("nyc"), "lon_modes": lambda: table_modes("lon"),
           "walk": table_walk_bands, "purpose": table_purpose}[which]())
