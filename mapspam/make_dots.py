"""
Convert MapSPAM harvested area grid into dot map data using Hilbert curve accumulation.
Each dot represents HA_PER_DOT hectares of harvested crop area.
"""
import pandas as pd
import numpy as np
import struct
import sys
import os

HA_PER_DOT = 10_000

# ── Hilbert curve helpers ──
def xy_to_hilbert(x, y, order):
    """Convert (x, y) grid coords to Hilbert curve distance, for a 2^order grid."""
    d = 0
    s = order - 1
    while s >= 0:
        n = 1 << s
        rx = 1 if (x & n) > 0 else 0
        ry = 1 if (y & n) > 0 else 0
        d += n * n * ((3 * rx) ^ ry)
        # rotate
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s -= 1
    return d


def main():
    crop = sys.argv[1].upper() if len(sys.argv) > 1 else "BARL"
    col = f"{crop}_A"

    print(f"Loading data for {crop}...")
    df = pd.read_csv("data/Global_CSV/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv",
                      usecols=["x", "y", col])
    mask = df[col] > 0
    df = df[mask].reset_index(drop=True)
    print(f"  {len(df)} pixels, total {df[col].sum():,.0f} ha")

    cell_size = 5.0 / 60.0  # 1/12 degree
    ix = np.round((df["x"].values + 180) / cell_size).astype(int)
    iy = np.round((df["y"].values + 90) / cell_size).astype(int)
    vals = df[col].values

    order = 13
    print(f"  Computing Hilbert distances (order {order})...")
    distances = np.array([xy_to_hilbert(int(xi), int(yi), order) for xi, yi in zip(ix, iy)])

    sort_idx = np.argsort(distances)
    ix = ix[sort_idx]
    iy = iy[sort_idx]
    vals = vals[sort_idx]
    lons = df["x"].values[sort_idx]
    lats = df["y"].values[sort_idx]

    # Walk along curve, accumulate, drop dots
    print(f"  Accumulating dots (1 dot per {HA_PER_DOT:,} ha)...")
    rng = np.random.default_rng(42)
    dots = []
    accum = 0.0
    for i in range(len(vals)):
        accum += vals[i]
        while accum >= HA_PER_DOT:
            dx = rng.uniform(-cell_size / 2, cell_size / 2)
            dy = rng.uniform(-cell_size / 2, cell_size / 2)
            dots.append((lons[i] + dx, lats[i] + dy))
            accum -= HA_PER_DOT

    print(f"  {len(dots)} dots generated")

    out_bin = f"data/{crop.lower()}_dots.bin"
    with open(out_bin, "wb") as f:
        for lon, lat in dots:
            f.write(struct.pack("<ff", lon, lat))
    size_kb = os.path.getsize(out_bin) / 1024
    print(f"  Saved {out_bin} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        crops = ["BANA","BARL","BEAN","CASS","CHIC","CITR","CNUT","COCO","COFF",
                 "COTT","COWP","GROU","LENT","MAIZ","MILL","OCER","OFIB","OILP",
                 "ONIO","OOIL","OPUL","ORTS","PIGE","PLNT","PMIL","POTA","RAPE",
                 "RCOF","REST","RICE","RUBB","SESA","SORG","SOYB","SUGB","SUGC",
                 "SUNF","SWPO","TEAS","TEMF","TOBA","TOMA","TROF","VEGE","WHEA","YAMS"]
        for c in crops:
            sys.argv[1] = c
            main()
    else:
        main()
