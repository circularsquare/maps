"""T-636: tables from the t636 test's JSON files. python t636_tables.py <out dir> [names...]"""
import json
import sys

import numpy as np


def load(path):
    r = json.load(open(path))
    for k in ("A", "B", "none", "C"):
        r[k] = np.array(r[k], float)  # layer, band, [trips, rail, walk, car, other]
    return r


def agg(a, layers=None, bands=None):
    layers = range(a.shape[0]) if layers is None else layers
    bands = range(a.shape[1]) if bands is None else bands
    return a[np.ix_(list(layers), list(bands))].sum(axis=(0, 1))


def modes(r, k, layers):
    """[rail, walk, car, other] shares of trips, for the given layers (list)."""
    if k == "C":
        # rail from the pair sums; walk and other from the model's day totals
        t = agg(r["C"], layers)
        trips = t[0]
        rail = t[1]
        if layers == [0]:
            d = r["day_c"]
        elif 0 not in layers:
            d = r["day_e"]
        else:
            d = [a + b for a, b in zip(r["day_c"], r["day_e"])]
        walk, other = d[2], d[3]
        return np.array([rail, walk, trips - rail - walk - other, other]) / trips
    t = agg(r[k], layers)
    return t[1:] / t[0]


def band_rail(r, k, layers):
    a = r[k]
    out = []
    for b in range(a.shape[1]):
        t = agg(a, layers, [b])
        out.append(t[1] / t[0])
    return out


def share_of_trips(r, layers):
    a = r["A"]
    return [agg(a, layers, [b])[0] / agg(a, layers)[0] for b in range(a.shape[1])]


if __name__ == "__main__":
    d = sys.argv[1]
    names = sys.argv[2:]
    for n in names:
        r = load(f"{d}/{n}.json")
        al = list(range(7))
        print(n, r["params"])
        for k in ("A", "B", "C"):
            print("  ", k, "all", np.round(100 * modes(r, k, al), 1), "commute", np.round(100 * modes(r, k, [0]), 1), "everyday", np.round(100 * modes(r, k, al[1:]), 1))
            print("     bands", np.round(100 * np.array(band_rail(r, k, al)), 1))
        nm = modes(r, "none", [0])
        print("   norail commute walk/other", np.round(100 * nm[1], 2), np.round(100 * nm[3], 2), "model walk", round(100 * r["no_rail_walk_model"], 2))
        print("   trips by band", np.round(100 * np.array(share_of_trips(r, al)), 1))
