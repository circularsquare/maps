"""Append customer-sector weights to an existing city pack without changing commutes.

Uses cached LODES blocks and the same land-area spreading as build_city.py. Any registered
US city (T-066 New York, T-142 the rest); re-running replaces the two arrays in place.
Run from maps: venv/Scripts/python.exe trainworld/pipeline/enrich_everyday.py phl
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import build_city as b
import education_sources as education
import spread


def main(city="nyc"):
    stem = os.path.join(b.PACKS, city)
    with open(stem + ".json") as f:
        header = json.load(f)
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    cfg = b.CITIES[city]
    boundary = b.build_boundary(city, cfg)
    mask, _ = b.build_water(cfg, boundary)
    df = pd.read_csv(os.path.join(b.WORK, city + ".blocks.csv.gz"), dtype={"geoid": str})
    df["errand_jobs"] = df.CNS07 + df.CNS19
    df["leisure_jobs"] = df.CNS17 + df.CNS18
    ids, vals, stats = spread.spread(df, set(cfg["counties"]), b.fetch, mask,
                                   *cfg["origin"], cols=("errand_jobs", "leisure_jobs"))
    a = next(a for a in header["arrays"] if a["name"] == "h3")
    cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"]).copy()
    index = np.searchsorted(cells, ids)
    hit = (index < len(cells)) & (cells[np.minimum(index, len(cells)-1)] == ids)
    out = {}
    for name, v in vals.items():
        missing = float(v[~hit].sum())
        if missing > 0.01:
            raise ValueError(f"{name}: {missing} jobs outside existing cells; rebuild city pack")
        weights = np.zeros(len(cells), dtype="<f4")
        weights[index[hit]] = v[hit]
        out[name] = weights
        print(name, float(weights.sum(dtype=np.float64)), flush=True)
    # Compacts the directory: a re-run leaves no orphaned bytes, other arrays copied bit for bit.
    buf = education.replace_cell_arrays(header, buf, out)
    header["everyday"] = {"enabled": True, "source": "LODES CNS07+CNS19 errands; CNS17+CNS18 leisure, original block locations, spread by land area",
                          "errand_jobs": float(out["errand_jobs"].sum(dtype=np.float64)),
                          "leisure_jobs": float(out["leisure_jobs"].sum(dtype=np.float64))}
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    print(stats, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
