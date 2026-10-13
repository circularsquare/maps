"""T-198: per-city before/after of the work gravity's distance curve, from T-185's candidate cache
(t185_candidates.py SET base,shape; both curves solved in numpy on the shipped zones with each
pack's border factor). Read-only.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t198_table.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t185_common as tc  # noqa: E402

NAMES = {"nyc": "New York", "phl": "Philadelphia", "bos": "Boston", "wdc": "Washington", "bal": "Baltimore",
         "pvd": "Providence", "nhv": "New Haven", "hfd": "Hartford", "spf": "Springfield"}
# "K" = the pack's border factor when T-185 ran (0.23, Washington 0.49); Washington's new curve is
# scored at its refitted 0.45 (t185_candidates.py t198 wdc)
OLD, NEW = "grav 1.0 40.0 K", "grav 0.8 30.0 K"


def new_key(c):
    return "grav 0.8 30.0 0.45" if c == "wdc" else NEW


def main():
    R = json.load(open(os.path.join(tc.CACHE, "cand.json")))
    X = json.load(open(os.path.join(tc.CACHE, "extra.json"))) if os.path.exists(os.path.join(tc.CACHE, "extra.json")) else {}
    if "--keys" in sys.argv:
        print(sorted(R[f"nyc|{OLD}"].keys()))
        return
    print("| pack | mean trip km, LODES / old / new | length misplaced % old / new | county pairs % old / new | "
          "opportunity rank % old / new | class table % old / new | sum old / new | centre->suburb x old / new | "
          "suburb->centre x old / new | job-centre trip km LODES / old / new | bedroom trip km LODES / old / new |")
    print("|---|" + "---|" * 10)
    for c in tc.CITIES:
        o, n = R.get(f"{c}|{OLD}"), R.get(f"{c}|{new_key(c)}")
        if o is None or n is None:
            print(f"| {NAMES[c]} | missing |")
            continue
        s = lambda r: (r["rel_off"] + r["tld_off"] + r["county_off"] + r["class_off"]) * 100
        lm = X.get(c, {}).get("rel_lodes", {}).get("km", [np.nan] * 5)
        rl = X.get(c, {}).get("rel_lodes")
        obs_mean = float(np.dot(rl["n"], rl["km"]) / np.sum(rl["n"])) if rl else np.nan
        print(f"| {NAMES[c]} | {obs_mean:.1f} / {o['mean']:.1f} / {n['mean']:.1f} | {o['tld_off'] * 100:.1f} / {n['tld_off'] * 100:.1f} | "
              f"{o['county_off'] * 100:.1f} / {n['county_off'] * 100:.1f} | {o['rel_off'] * 100:.1f} / {n['rel_off'] * 100:.1f} | "
              f"{o['class_off'] * 100:.1f} / {n['class_off'] * 100:.1f} | {s(o):.1f} / {s(n):.1f} | "
              f"{o['ratio'][0][2]:.2f} / {n['ratio'][0][2]:.2f} | {o['ratio'][2][0]:.2f} / {n['ratio'][2][0]:.2f} | "
              f"{lm[4]:.1f} / {o['rel']['km'][4]:.1f} / {n['rel']['km'][4]:.1f} | {lm[0]:.1f} / {o['rel']['km'][0]:.1f} / {n['rel']['km'][0]:.1f} |")
    print("\ncross (between-state share) old / new:")
    for c in tc.CITIES:
        o, n = R.get(f"{c}|{OLD}"), R.get(f"{c}|{new_key(c)}")
        if o and n:
            print(f"  {NAMES[c]}: {o['cross'] * 100:.1f} / {n['cross'] * 100:.1f}")


def plot():
    """../artifacts/trainworld/T-198-by-city.png: three measures per pack, old and new curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.load(open(os.path.join(tc.CACHE, "cand.json")))
    X = json.load(open(os.path.join(tc.CACHE, "extra.json")))
    cs = tc.CITIES
    x = np.arange(len(cs))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    s = lambda r: (r["rel_off"] + r["tld_off"] + r["county_off"] + r["class_off"]) * 100
    panels = [
        ("Trips misplaced, sum of four scores (%)", lambda r, c: s(r), None),
        ("Centre -> suburb commuting, model / LODES", lambda r, c: r["ratio"][0][2], 1.0),
        ("Job-centre residents' mean trip, model / LODES", lambda r, c: r["rel"]["km"][4] / X[c]["rel_lodes"]["km"][4], 1.0),
    ]
    for ax, (title, f, ref) in zip(axes, panels):
        ax.bar(x - 0.2, [f(R[f"{c}|{OLD}"], c) for c in cs], 0.4, color="#9aa5b1", label="d^-1 exp(-d/40 km)")
        ax.bar(x + 0.2, [f(R[f"{c}|{new_key(c)}"], c) for c in cs], 0.4, color="#2b6cb0", label="d^-0.8 exp(-d/30 km)")
        if ref:
            ax.axhline(ref, color="#333", lw=1)
            ax.set_ylim(0.6, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels([NAMES[c] for c in cs], rotation=50, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("T-198: work gravity curve in each of the nine packs (LODES 2023 as the yardstick)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(tc.HERE), "..", "artifacts", "trainworld", "T-198-by-city.png")
    fig.savefig(p, dpi=110)
    print(f"plot: {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
    plot()
