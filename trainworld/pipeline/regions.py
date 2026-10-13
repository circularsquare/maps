"""T-162: the first-level region of every pack cell, for the region-aware work gravity.

    python pipeline/regions.py <city> [--border-factor K] [--decay-km D]

adds a `region` array to a finished pack in data/packs/ (if it has none) and re-solves its zone
gravity in place with the sim crate's `pack_gravity`: zones split where a region border cuts a
2 km square, and the decay between zones of different regions multiplied by the border factor.
Nothing else in the pack changes. build_city.py does the same for a new pack as it builds it.

A cell's region is the one holding most of what its census blocks put into it (people, jobs and
resident job holders, plus a little for land so every cell has one), spread over the block
polygons exactly as the pack's own figures were (spread.py), so the border follows the blocks.
In the US the region is the state: the first two digits of the block GEOID. Outside the US any
admin-1 layer (geoBoundaries ADM1, OSM admin_level 4) gives the same per-cell id.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOURCE = "US state: the first two digits of each 2020 census block GEOID; a cell takes the state holding most of its blocks' people, jobs and land"


def add_columns(df):
    """Per region, a column weighting the blocks in it. Returns (df, codes, column names)."""
    codes = sorted(df["geoid"].str[:2].unique())
    assert len(codes) < 256, "region ids are one byte"
    w = df["pop"].to_numpy(np.float64) + df["jobs"].to_numpy(np.float64) + 1e-3
    if "rac" in df:
        w = w + df["rac"].to_numpy(np.float64)
    st = df["geoid"].str[:2].to_numpy()
    cols = [f"region_{c}" for c in codes]
    df = df.assign(**{col: np.where(st == c, w, 0.0) for c, col in zip(codes, cols)})
    return df, codes, cols


def of_cells(vals, cols):
    """Region index per spread cell: the column with the most weight."""
    return np.argmax(np.stack([vals[c] for c in cols], axis=1), axis=1).astype(np.uint8)


def header_block(codes):
    import build_city as bc
    return {"level": "admin-1", "ids": codes, "names": [bc.STATES.get(c, c).upper() for c in codes], "source": SOURCE}


def add_to_pack(city):
    """Add the `region` array and `regions` header block to data/packs/<city> if missing."""
    import build_city as bc
    import spread
    import water
    from read_pack import load_pack

    header, a = load_pack(city)
    if "region" in a:
        print(f"{city}: pack already has regions {header.get('regions', {}).get('names')}")
        return
    cfg = bc.CITIES[city]
    wb = header["water"]
    (x0, y0), (W, H) = wb["origin_m"], wb["size"]
    mask = water.WaterMask(x0, y0, wb["cell_m"], W, H, np.array(a["water_row"]), np.array(a["water_x"]))
    df = pd.read_csv(os.path.join(bc.WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str})
    df, codes, cols = add_columns(df)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    ids, vals, stats = spread.spread(df, set(cfg["counties"]), bc.fetch, mask, lon0, lat0, cols=tuple(cols))
    reg = of_cells(vals, cols)
    cells = a["h3"]
    i = np.searchsorted(ids, cells)
    ok = (i < len(ids)) & (ids[np.minimum(i, len(ids) - 1)] == cells)
    assert ok.all(), f"{int((~ok).sum())} pack cells got no block weight; rebuild the pack"
    region = reg[i]
    counts = np.bincount(region, minlength=len(codes))
    print(f"{city}: regions " + ", ".join(f"{bc.STATES.get(c, c).upper()} {n:,} cells" for c, n in zip(codes, counts)) + f" ({stats['seconds']} s)")
    stem = os.path.join(bc.PACKS, city)
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    header["arrays"] = [e for e in header["arrays"] if e["name"] != "region"]
    buf += b"\0" * (-len(buf) % 8)
    header["arrays"].append({"name": "region", "dtype": "u8", "per": "cell", "offset": len(buf), "count": int(len(cells))})
    buf += region.astype(np.uint8).tobytes()
    header["regions"] = header_block(codes)
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")


def main(args):
    import build_city as bc
    city = args[0]
    extra = args[1:]
    add_to_pack(city)
    pack = os.path.join(bc.PACKS, f"{city}.json")
    bc.add_gravity(pack, city, extra)


if __name__ == "__main__":
    main(sys.argv[1:])
