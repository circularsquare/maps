"""T-420: Greater Tokyo's pack figures against the census, by prefecture and near Tokyo Station.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t420_figures.py
"""
import gzip
import json
import os
import sys

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import canada  # noqa: E402
import japan  # noqa: E402

CENSUS_POP = {"11": 7344765, "12": 6284480, "13": 14047594, "14": 9237337}  # 2020 census (Statistics Bureau)
NAMES = {"11": "Saitama", "12": "Chiba", "13": "Tokyo", "14": "Kanagawa"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    city = "tyo"
    stem, header, buf, cells = canada.pack_cells(city)
    arr = {a["name"]: a for a in header["arrays"]}
    get = lambda n: np.frombuffer(buf, "<f4", arr[n]["count"], arr[n]["offset"]).astype(np.float64)
    pop, jobs, cw = get("pop"), get("jobs"), get("commute_work")
    x, y = get("x_m"), get("y_m")
    raw = os.path.getsize(os.path.join(b.PACKS, f"{city}.bin"))
    with open(os.path.join(b.PACKS, f"{city}.bin"), "rb") as f:
        gz = len(gzip.compress(f.read(), 6))
    g = header.get("gravity", {})
    print(f"cells {len(cells):,}; pop {pop.sum():,.0f}; jobs {jobs.sum():,.0f}; commuters {cw.sum():,.0f}; "
          f"zones {g.get('zones', g.get('n_zones', '?'))}; .bin {raw / 1e6:.2f} MB raw, {gz / 1e6:.2f} MB gzipped")
    import h3
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in cells])
    ms = japan.municipalities(list(CENSUS_POP))
    idx, _ = __import__("england").cell_units(ll[:, 0], ll[:, 1], [(m[0], m[3]) for m in ms])
    pref = np.array([ms[i][2] if i >= 0 else "" for i in idx])
    want, _ = japan.workers_at_work(list(CENSUS_POP))
    for p in CENSUS_POP:
        sel = pref == p
        wj = sum(v for k, v in want.items() if k.startswith(p) and not (p == "13" and int(k) >= japan.ISLANDS))
        print(f"  {NAMES[p]}: pop {pop[sel].sum():,.0f} (census {CENSUS_POP[p]:,}, islands included), "
              f"jobs {jobs[sel].sum():,.0f} (census workers by place of work {wj:,.0f})")
    d = np.hypot(x, y) / 1000
    for r in (1, 2, 5, 10):
        print(f"  within {r} km of Tokyo Station: pop {pop[d < r].sum():,.0f}, jobs {jobs[d < r].sum():,.0f}")
    wards = np.array([i >= 0 and ms[i][0].startswith("131") for i in idx])
    wj = sum(v for k, v in want.items() if k.startswith("131"))
    isl = sum(v for k, v in want.items() if k.startswith("13") and int(k) >= japan.ISLANDS)
    print(f"  the 23 wards: pop {pop[wards].sum():,.0f}, jobs {jobs[wards].sum():,.0f} (census {wj:,.0f}); "
          f"Tokyo's islands' workers left out {isl:,.0f}")
    for k in ("school", "university", "medical", "parks", "everyday"):
        if k in header:
            print(f"  {k}: " + json.dumps({a: v for a, v in header[k].items() if isinstance(v, (int, float))})[:300])


if __name__ == "__main__":
    main()
