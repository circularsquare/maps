"""T-185: summarise CACHE/cand.json (t185_candidates.py) and extra.json (t185_extra.py) as markdown,
and draw the opportunity-rank plot. Read-only.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t185_cand_report.py
"""
import json
import os

import numpy as np

import t185_common as tc

ART = os.path.join(os.path.dirname(tc.HERE), "..", "artifacts", "trainworld")
NAMES = {"nyc": "New York", "phl": "Philadelphia", "bos": "Boston", "wdc": "Washington", "bal": "Baltimore",
         "pvd": "Providence", "nhv": "New Haven", "hfd": "Hartford", "spf": "Springfield"}
RB = ["under 0.5", "0.5-0.8", "0.8-1.25", "1.25-2", "over 2"]


def main():
    R = json.load(open(os.path.join(tc.CACHE, "cand.json")))
    X = json.load(open(os.path.join(tc.CACHE, "extra.json"))) if os.path.exists(os.path.join(tc.CACHE, "extra.json")) else {}
    specs = sorted({k.split("|", 1)[1] for k in R})
    rows = []
    for s in specs:
        have = [c for c in tc.CITIES if f"{c}|{s}" in R]
        if len(have) < 9:
            continue
        g = lambda f: np.mean([R[f"{c}|{s}"][f] for c in have]) * 100
        cso = np.mean([R[f"{c}|{s}"]["ratio"][0][2] for c in have])
        sco = np.mean([R[f"{c}|{s}"]["ratio"][2][0] for c in have])
        jm = g("job_margin_off")
        rows.append((g("rel_off") + g("tld_off") + g("county_off") + g("class_off"), s, g("rel_off"), g("tld_off"),
                     g("county_off"), g("class_off"), jm, cso, sco))
    rows.sort()
    print("| candidate | opportunity-rank | length | county pairs | class | job margin | C->S x | S->C x | sum |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tot, s, a, b, c, d, jm, cso, sco in rows:
        print(f"| {s} | {a:.1f} | {b:.1f} | {c:.1f} | {d:.1f} | {jm:.1f} | {cso:.2f} | {sco:.2f} | {tot:.1f} |")
    print("\n(mean over the nine packs, % of trips misplaced; C->S x and S->C x = model/LODES share, mean of packs)\n")
    # per city, the main candidate's strength curve
    print("| pack | " + " | ".join(f"a={a}" for a in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.4)) + " |")
    print("|---|" + "---|" * 6)
    for c in tc.CITIES:
        cells = []
        for a in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.4):
            r = R.get(f"{c}|hybL {a} K")
            cells.append("" if r is None else f"{(r['rel_off'] + r['tld_off'] + r['county_off'] + r['class_off']) * 100:.1f} ({r['km']:.0f} km)")
        print(f"| {NAMES[c]} | " + " | ".join(cells) + " |")
    # relative yardstick, shipped model
    if X:
        print("\n## By local job ratio (jobs per resident commuter within 5 km): LODES / now\n")
        print("| pack | " + " | ".join(RB) + " |")
        print("|---|" + "---|" * 5)
        for c in tc.CITIES:
            if c not in X:
                continue
            o, m = X[c]["rel_lodes"], R[f"{c}|grav 1.0 40.0 K"]["rel"]
            sh = np.array(o["n"]) / np.sum(o["n"])
            cells = [f"{sh[b]:.0%} of homes: {o['km'][b]:.1f} / {m['km'][b]:.1f} km, down {o['down'][b]:.0%} / {m['down'][b]:.0%}" for b in range(5)]
            print(f"| {NAMES[c]} | " + " | ".join(cells) + " |")
        print("\n## Central county: out / in (LODES | now | k1 | pre | io-0.2 | io+0.1 | io+0.2)\n")
        print("| pack | residents working outside | jobs held by outsiders |")
        print("|---|---|---|")
        for c in tc.CITIES:
            if "central" not in X.get(c, {}):
                continue
            ce = X[c]["central"]
            labs = ["lodes", "now", "k1", "pre", "io-0.2", "io+0.1", "io+0.2"]
            print(f"| {NAMES[c]} | " + " / ".join(f"{ce[l][0] * 100:.0f}" for l in labs if l in ce) + " | "
                  + " / ".join(f"{ce[l][1] * 100:.0f}" for l in labs if l in ce) + " |")
        plot(R, X)


def plot(R, X):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = ["own zone", "<1k", "1-3k", "3-10k", "10-30k", "30-100k", "100-300k", "0.3-1M", "1-3M", ">3M"]
    cities = [c for c in tc.CITIES if c in X]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharey=True)
    cols = ["#2b6cb0", "#38a169", "#d69e2e", "#dd6b20", "#c53030"]
    for ax, c in zip(axes.ravel(), cities):
        Jo = np.array(X[c]["rel_lodes"]["J"])
        Jm = np.array(R[f"{c}|grav 1.0 40.0 K"]["rel"]["J"])
        Ji = np.array(R[f"{c}|hybL 0.2 K"]["rel"]["J"]) if f"{c}|hybL 0.2 K" in R else None
        for b in range(5):
            if Jo[b].sum() <= 0:
                continue
            x = np.arange(Jo.shape[1])
            ax.plot(x, np.cumsum(Jo[b]) / Jo[b].sum(), color=cols[b], lw=2, label=f"LODES, ratio {RB[b]}")
            ax.plot(x, np.cumsum(Jm[b]) / max(Jm[b].sum(), 1e-9), color=cols[b], lw=1, ls="--", label="model now" if b == 0 else None)
        ax.set_title(NAMES[c])
        ax.set_xticks(range(len(edges)))
        ax.set_xticklabels(edges, rotation=60, fontsize=7)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Share of commutes that end within N intervening jobs of home, by home's local job ratio\n"
                 "(solid LODES 2023, dashed shipped model)")
    fig.tight_layout()
    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, "T-185-opportunity-rank.png")
    fig.savefig(p, dpi=110)
    print(f"\nplot: {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
