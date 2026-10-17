"""Markdown tables for notes/T-636.md from the out/*.json and the calib scores."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t636_tables import band_rail, load, modes, share_of_trips  # noqa

S = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
NAMES = [("today", "Today"), ("simple", "Simple"), ("b075", "Simple, beta 0.075"), ("b100", "Simple, beta 0.10"),
         ("d3a", "Simple, beta 0.075, damp 26.7"), ("d3b", "Simple, beta 0.10, damp 15"),
         ("w15", "Simple, walk constant 1.5"), ("b100w15", "Simple, beta 0.10, walk 1.5"), ("d3aw15", "Simple, beta 0.075, damp 26.7, walk 1.5")]
CITIES = [("nyc", "New York"), ("lon", "London")]
BANDS = ["0-2", "2-5", "5-10", "10-20", "20-50", "50+"]
AL = list(range(7))


def score(city, name):
    return json.load(open(f"{TW}/data/work/calib/{city}/score_t636_{name}.json"))


def f1(x):
    return f"{100 * x:.1f}"


R = {(c, n): load(f"{S}/out/{c}_{n}.json") for c, _ in CITIES for n, _ in NAMES}
SC = {(c, n): score(c, n) for c, _ in CITIES for n, _ in NAMES}
out = []
w = out.append

w("### Headline: rail's share of all trips (%), and the real networks' station level\n")
w("| Setting | NY A | NY B | NY C | NY stations (x counts) | London A | London B | London C | London stations (x counts) |")
w("|---|---|---|---|---|---|---|---|---|")
for n, label in NAMES:
    row = [label]
    for c, _ in CITIES:
        r = R[(c, n)]
        row += [f1(modes(r, "A", AL)[0]), f1(modes(r, "B", AL)[0]), f1(modes(r, "C", AL)[0]), f"x{np.exp(SC[(c, n)]['station_level']):.2f}"]
    w("| " + " | ".join(row) + " |")
w("")
for c, cl in CITIES:
    w(f"### {cl}: shares of all trips (%), rail / walk / car / other\n")
    w("| Setting | A | B | C |")
    w("|---|---|---|---|")
    for n, label in NAMES:
        r = R[(c, n)]
        w(f"| {label} | " + " | ".join(" / ".join(f1(x) for x in modes(r, k, AL)) for k in "ABC") + " |")
    w("")
    w(f"### {cl}: rail's share of all trips (%) by straight-line distance (km)\n")
    sh = share_of_trips(R[(c, "today")], AL)
    w("| Setting | Test | " + " | ".join(BANDS) + " | Commutes | Everyday |")
    w("|---|---|" + "---|" * 8)
    w("| Share of trips | | " + " | ".join(f"{100 * x:.0f}" for x in sh) + " | | |")
    for n, label in NAMES:
        r = R[(c, n)]
        for k in "ABC":
            b = band_rail(r, k, AL)
            w(f"| {label} | {k} | " + " | ".join(f1(x) for x in b) + f" | {f1(modes(r, k, [0])[0])} | {f1(modes(r, k, AL[1:])[0])} |")
    w("")
    w(f"### {cl}: the real network against the counts\n")
    w("| Setting | Score | Stations | Stations x counts | Spread | Operators | Areas | Commute rail by area, model (census) | Commutes walking (census) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n, label in NAMES:
        s = SC[(c, n)]
        p = s["parts"]
        wk = s.get("walk_areas")
        w(f"| {label} | {s['total']:.2f} | {p['stations']:.2f} | x{np.exp(s['station_level']):.2f} | {s['station_spread']:.2f} | {p['operators']:.2f} | {p['areas']:.2f} | "
          f"{f1(s['rail_share_model'])} ({f1(s['rail_share_measured'])}) | " + (f"{f1(wk['model'])} ({f1(wk['measured'])})" if wk else "") + " |")
    w("")
print("\n".join(out))
