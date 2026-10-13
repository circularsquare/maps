"""T-185 question 1: centre / middle / suburb flows, model against LODES, in all nine packs.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t185_classes.py [city ...]

Variants: "now" = shipped gravity (d^-1 exp(-d/40), border factor 0.23, Washington 0.49);
"k1" = the same decay with no border factor; "pre" = T-162's before (d^-1 exp(-d/29), no factor).
Writes CACHE/classes.json and prints the tables.
"""
import json
import os
import sys

import numpy as np

import t185_common as tc

VARIANTS = {"now": None, "k1": (1.0, 40.0, 1.0), "pre": (1.0, 29.0, 1.0)}


def main(names):
    out = os.path.join(tc.CACHE, "classes.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    for name in names:
        c = tc.load(name)
        r = {"lodes": c.lodes_summary(), "zones": np.bincount(c.cls, minlength=3).tolist(),
             "home": [float(c.pop[c.cls == i].sum()) for i in range(3)],
             "jobs": [float(c.jobs[c.cls == i].sum()) for i in range(3)], "density_cut": c.density_cut}
        for v, p in VARIANTS.items():
            if p is None:
                p = (1.0, 40.0, tc.PACK_K.get(name, tc.BORDER_US))
            T = c.solve(*p)
            r[v] = c.evaluate(T)
            del T
            e = r[v]
            print(f"  {name} {v}: class off {e['class_off']:.1%}, county off {e['county_off']:.1%}, "
                  f"length off {e['tld_off']:.1%}, cross {e['cross']:.1%} (LODES {c.obs_cross:.1%})", flush=True)
        res[name] = r
        json.dump(res, open(out, "w"), indent=1)
        del c


if __name__ == "__main__":
    main(sys.argv[1:] or tc.CITIES)
