"""T-011: the score files as tables, per tile and as means by kind of land.

    python isochrone/notes/T-011/summary.py <dir> [variant,variant,...]
"""
import glob
import json
import os
import sys

import numpy as np

KIND = {"n46e007": "mount", "n45e006": "mount", "n27e085": "mount", "n28e083": "mount", "s14w072": "mount",
        "n61e007": "mount", "n44e007": "mount", "n11e038": "mount", "n46e023": "hills", "n37w083": "hills",
        "n41w094": "flat", "n42w093": "flat", "n38w098": "flat", "n50w105": "flat", "s34w061": "flat",
        "n13w002": "flat", "n50e002": "flat", "n61e025": "water", "n10e105": "water"}
LABEL = {"n46e007": "Valais", "n45e006": "Mont Blanc", "n27e085": "Nepal hills", "n28e083": "Annapurna",
         "s14w072": "Cusco", "n61e007": "Sogndal", "n44e007": "Piedmont", "n11e038": "Ethiopia",
         "n46e023": "Cluj", "n37w083": "Kentucky", "n41w094": "Iowa", "n42w093": "Iowa 2", "n38w098": "Kansas",
         "n50w105": "Saskatchewan", "s34w061": "Pampas", "n13w002": "Sahel", "n50e002": "N France",
         "n61e025": "Finland", "n10e105": "Mekong"}


def main(d, want):
    S = {}
    for p in sorted(glob.glob(os.path.join(d, "*_score.json"))):
        r = json.load(open(p))
        S[r["name"]] = r
    for grid in ("plain", "roads"):
        vs = want or list(next(iter(S.values()))[grid])
        vs = [v for v in vs if all(v in S[t][grid] for t in S)]
        print("\n%s: median |t100 - t1| / t1 and ratio (sum t100 / sum t1); shift = sum t1 / sum t1 today" % grid)
        print("%-13s" % "tile" + "".join("%17s" % v for v in vs))
        order = sorted(S, key=lambda t: (["mount", "hills", "flat", "water"].index(KIND[t]), t))
        for t in order:
            print("%-13s" % LABEL[t] + "".join("  %5.1f%% %4.2f %4.2f" % (100 * S[t][grid][v]["rel"], S[t][grid][v]["ratio"],
                                                                     S[t][grid][v]["shift"]) for v in vs))
        for kinds, lab in ((("mount",), "mean mount"), (("hills",), "mean hills"), (("flat",), "mean flat"),
                           (("mount", "hills", "flat"), "mean 17"), (("mount", "hills", "flat"), "worst 17")):
            ts = [t for t in S if KIND[t] in kinds]
            if lab.startswith("worst"):
                print("%-13s" % lab + "".join("  %5.1f%%          " % (100 * max(S[t][grid][v]["rel"] for t in ts)) for v in vs))
            else:
                print("%-13s" % lab + "".join("  %5.1f%% %4.2f %4.2f" % (100 * np.mean([S[t][grid][v]["rel"] for t in ts]),
                                                                       np.mean([S[t][grid][v]["ratio"] for t in ts]),
                                                                       np.mean([S[t][grid][v]["shift"] for t in ts])) for v in vs))
        # up and down trips, mountains
        print("up / down ratio, mountain tiles (dz over +-200 m)")
        for v in vs:
            ts = [t for t in S if KIND[t] == "mount" and "up" in S[t][grid][v] and "down" in S[t][grid][v]]
            if ts:
                print("  %-10s up %.3f down %.3f" % (v, np.mean([S[t][grid][v]["up"]["ratio"] for t in ts]),
                                                    np.mean([S[t][grid][v]["down"]["ratio"] for t in ts])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else None)
