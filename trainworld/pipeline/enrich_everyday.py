"""Append customer-sector weights to an existing NYC pack without changing commutes.

Uses cached LODES blocks and the same land-area spreading as build_city.py.
Run from maps: venv/Scripts/python.exe trainworld/pipeline/enrich_everyday.py
"""
import json
import os
import numpy as np
import pandas as pd
import build_city as b
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
    for name, v in vals.items():
        missing = float(v[~hit].sum())
        if missing > 0.01:
            raise ValueError(f"{name}: {missing} jobs outside existing cells; rebuild city pack")
        weights = np.zeros(len(cells), dtype="<f4")
        weights[index[hit]] = v[hit]
        # Re-running replaces the directory entry; older bytes remain harmlessly unreferenced.
        header["arrays"] = [a for a in header["arrays"] if a["name"] != name]
        buf += b"\0" * (-len(buf) % 8)
        header["arrays"].append(dict(name=name, dtype="f32", per="cell", offset=len(buf), count=len(cells)))
        buf += weights.tobytes()
        print(name, float(weights.sum()), flush=True)
    header["everyday"] = {"enabled": city == "nyc", "source": "LODES CNS07+CNS19 errands; CNS17+CNS18 leisure, original block locations, spread by land area"}
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    print(stats, flush=True)


if __name__ == "__main__":
    main()
