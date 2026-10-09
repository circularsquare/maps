"""T-014: list job cells far above their neighbourhood, with the LODES blocks behind them and
their sector mix, so each can be judged (real centre, or an employer's jobs reported at one
office). Reads the built pack and data/work/<city>.blocks.csv.gz; writes nothing.

    python pipeline/lumps.py nyc [top]

Score: a cell's jobs against the mean of the 18 cells in rings 1-2 around it (missing cells count
as 0), for cells with at least 3,000 jobs.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys

import h3
import numpy as np
import pandas as pd

from read_pack import ROOT, load_pack

RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "work")
SECTORS = {  # LODES CNS codes (NAICS sectors)
    "CNS01": "farm", "CNS02": "mining", "CNS03": "utilities", "CNS04": "construction",
    "CNS05": "manufacturing", "CNS06": "wholesale", "CNS07": "retail", "CNS08": "transport",
    "CNS09": "information", "CNS10": "finance", "CNS11": "real estate", "CNS12": "professional",
    "CNS13": "management", "CNS14": "admin/support", "CNS15": "education", "CNS16": "health",
    "CNS17": "arts", "CNS18": "food/hotels", "CNS19": "other services", "CNS20": "public admin",
}


def wac_detail(geoids):
    out = []
    for st in ("ny", "nj", "ct"):
        w = pd.read_csv(os.path.join(RAW, f"{st}_wac_S000_JT00_2023.csv.gz"), dtype={"w_geocode": str})
        out.append(w[w["w_geocode"].isin(geoids)])
    return pd.concat(out).set_index("w_geocode")


def candidates(city, top=40, min_jobs=3000):
    header, a = load_pack(city)
    ids = a["h3"]
    jobs = dict(zip(ids.tolist(), a["jobs"].astype(np.float64).tolist()))
    rows = []
    for c, j in jobs.items():
        if j < min_jobs:
            continue
        s = h3.int_to_str(c)
        ring = [h3.str_to_int(n) for n in h3.grid_disk(s, 2) if n != s]
        mean = sum(jobs.get(n, 0.0) for n in ring) / len(ring)
        lat, lon = h3.cell_to_latlng(s)
        rows.append((s, lat, lon, j, mean, j / max(mean, 50.0)))
    df = pd.DataFrame(rows, columns=["cell", "lat", "lon", "jobs", "ring_mean", "score"])
    return df.sort_values("score", ascending=False).head(top)


def top_blocks(city, top):
    """The biggest LODES blocks with their sector mix: lumps live in single blocks, and since
    T-013 spreads a block over its polygon a cell score can hide one."""
    blocks = pd.read_csv(os.path.join(WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str})
    b = blocks.sort_values("jobs", ascending=False).head(top)
    det = wac_detail(set(b["geoid"]))
    for _, r in b.iterrows():
        d = det.loc[r["geoid"]]
        mix = sorted(((d[k], v) for k, v in SECTORS.items()), reverse=True)[:3]
        mixs = ", ".join(f"{v} {n / d['C000']:.0%}" for n, v in mix if n > 0)
        print(f"block {r['geoid']}  {r['lat']:.4f},{r['lon']:.4f}  jobs {r['jobs']:>7,}  {mixs}")


def main(city, top):
    if os.environ.get("LUMPS_BLOCKS"):
        return top_blocks(city, top)
    cand = candidates(city, top)
    blocks = pd.read_csv(os.path.join(WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str})
    blocks = blocks[blocks["jobs"] >= 1000]
    blocks["cell"] = [h3.latlng_to_cell(la, lo, 9) for la, lo in zip(blocks["lat"], blocks["lon"])]
    near = {}
    for s in cand["cell"]:
        disk = set(h3.grid_disk(s, 1))
        near[s] = blocks[blocks["cell"].isin(disk)].sort_values("jobs", ascending=False).head(3)
    det = wac_detail(set(g for b in near.values() for g in b["geoid"]))
    for _, r in cand.iterrows():
        print(f"{r['cell']}  {r['lat']:.4f},{r['lon']:.4f}  jobs {r['jobs']:>8,.0f}  ring mean {r['ring_mean']:>7,.0f}  x{r['score']:.0f}")
        for _, b in near[r["cell"]].iterrows():
            d = det.loc[b["geoid"]]
            mix = sorted(((d[k], v) for k, v in SECTORS.items()), reverse=True)[:3]
            mixs = ", ".join(f"{v} {n / d['C000']:.0%}" for n, v in mix if n > 0)
            size = ""
            print(f"      block {b['geoid']}  {b['lat']:.4f},{b['lon']:.4f}  jobs {b['jobs']:>7,}  {mixs}; {size}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nyc", int(sys.argv[2]) if len(sys.argv) > 2 else 40)
