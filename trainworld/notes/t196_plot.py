"""T-196: what the park draw looks like, from enrich_parks.py's audits (data/work/<city>.parks.audit.json).

    ..\\venv\\Scripts\\python.exe trainworld/notes/t196_plot.py

../artifacts/trainworld/T-196-draw.png: for New York, Philadelphia and Hartford, each kept place by
its area and residents within 2 km of its edge, coloured by its draw; and every pack's draw shares
(top places and kinds) printed.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(os.path.dirname(HERE), "data", "work")
ART = os.path.join(os.path.dirname(HERE), "..", "artifacts", "trainworld")
CITIES = ["nyc", "phl", "bos", "wdc", "bal", "pvd", "nhv", "hfd", "spf"]
NAMES = {"nyc": "New York", "phl": "Philadelphia", "bos": "Boston", "wdc": "Washington", "bal": "Baltimore",
         "pvd": "Providence", "nhv": "New Haven", "hfd": "Hartford", "spf": "Springfield"}


def load(c):
    p = os.path.join(WORK, f"{c}.parks.audit.json")
    return json.load(open(p))["places"] if os.path.exists(p) else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("| pack | places | parks | reserves (quarter) | other | draw from the 10 largest-draw places | largest area share of draw |")
    print("|---|---|---|---|---|---|---|")
    for c in CITIES:
        P = load(c)
        if not P:
            continue
        d = np.array([r["draw"] for r in P])
        kinds = [r["kind"] for r in P]
        tot = d.sum()
        k = lambda name: sum(r["draw"] for r in P if r["kind"] == name) / tot
        top = np.sort(d)[::-1][:10].sum() / tot
        big = max(P, key=lambda r: r["ha"])
        print(f"| {NAMES[c]} | {len(P):,} | {k('park'):.0%} | {k('nature_reserve'):.0%} | {1 - k('park') - k('nature_reserve'):.0%} | "
              f"{top:.0%} | {big['name'][:30]} {big['ha']:,.0f} ha: {big['draw'] / tot:.1%} |")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, c in zip(axes, ["nyc", "phl", "hfd"]):
        P = load(c)
        if not P:
            continue
        ha = np.array([max(r["ha"], 0.1) for r in P])
        pop = np.array([max(r["density_near"], 1.0) for r in P])
        d = np.array([r["draw"] for r in P])
        sc = ax.scatter(ha, pop, c=np.log10(np.maximum(d, 1e-3)), s=6, cmap="viridis", vmin=-2, vmax=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("area, ha")
        ax.set_title(NAMES[c], fontsize=10)
        ax.grid(alpha=0.3)
        for r in sorted(P, key=lambda r: -r["draw"])[:5]:
            ax.annotate(r["name"][:22], (max(r["ha"], 0.1), max(r["density_near"], 1)), fontsize=7)
    axes[0].set_ylabel("residents per km2 within 2 km of the edge")
    fig.colorbar(sc, ax=axes, label="log10 draw")
    fig.suptitle("T-196: park draw = sqrt(area) x D / (D + 2,000), reserves at a quarter")
    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, "T-196-draw.png")
    fig.savefig(p, dpi=110)
    print(f"plot: {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
