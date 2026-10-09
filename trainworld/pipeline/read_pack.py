"""Load a city pack back and check it against the format in notes/T-004.md, including the zone
gravity (factors re-checked against both constraints in numpy, independently of the Rust solve).

    python pipeline/read_pack.py nyc

Written separately from build_city.py (no shared code) so it tests the format, not the writer.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import math
import sys

import h3
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS = os.path.join(ROOT, "data", "packs")
DTYPES = {"u64": "<u8", "f32": "<f4", "u32": "<u4", "i32": "<i4", "f64": "<f8", "u16": "<u2", "u8": "u1"}
R_EARTH = 6371008.8


def load_pack(city, packs=PACKS):
    with open(os.path.join(packs, f"{city}.json")) as f:
        header = json.load(f)
    with open(os.path.join(packs, f"{city}.bin"), "rb") as f:
        buf = f.read()
    arrays = {}
    for a in header["arrays"]:
        if a["dtype"] not in DTYPES:
            continue  # readers ignore what they do not know
        dt = np.dtype(DTYPES[a["dtype"]])
        assert a["offset"] % 8 == 0, f"{a['name']} not 8-byte aligned"
        assert a["offset"] + a["count"] * dt.itemsize <= len(buf), f"{a['name']} runs past the end"
        arrays[a["name"]] = np.frombuffer(buf, dtype=dt, count=a["count"], offset=a["offset"])
    return header, arrays


def check_gravity(header, a):
    """The shipped zone gravity, checked from scratch: zone ids against cell positions, and the
    balancing factors against the two constraints (row sums = workers, column sums = jobs)."""
    g = header["gravity"]
    n, nz = header["cells"], header["zones"]
    for name, count in (("zone", n), ("zone_gx", nz), ("zone_gy", nz), ("zone_row", nz), ("zone_col", nz)):
        assert name in a and len(a[name]) == count, f"array {name} missing or wrong length"
    assert g["decay"] in ("exp", "pow_exp"), g["decay"]
    a_pow = g.get("decay_pow", 0.0) if g["decay"] == "pow_exp" else 0.0
    zm, (x0, y0), (w, h) = g["zone_m"], g["grid_origin_m"], g["grid"]
    zone = a["zone"].astype(np.int64)
    gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
    assert zone.max() < nz and gx.max() < w and gy.max() < h
    assert len(np.unique(gx * h + gy)) == nz, "two zones share a grid square"
    cx = ((a["x_m"] - np.float32(x0)) / np.float32(zm)).astype(np.int64)
    cy = ((a["y_m"] - np.float32(y0)) / np.float32(zm)).astype(np.int64)
    bad = int(((cx != gx[zone]) | (cy != gy[zone])).sum())
    assert bad == 0, f"{bad} cells sit outside their zone's square"
    # the gravity balances on commuters when the pack has them (T-042), else on pop and jobs
    hw = a["commute_home"] if "commute_home" in a else a["pop"]
    ww = a["commute_work"] if "commute_work" in a else a["jobs"]
    zpop = np.bincount(zone, weights=hw.astype(np.float64), minlength=nz)
    zjobs = np.bincount(zone, weights=ww.astype(np.float64), minlength=nz)
    workers = zpop * zjobs.sum() / zpop.sum()
    row, col = a["zone_row"].astype(np.float64), a["zone_col"].astype(np.float64)
    zk = zm / 1000.0
    rsum = np.zeros(nz)
    csum = np.zeros(nz)
    tot = dsum = 0.0
    for s in range(0, nz, 400):
        e = min(nz, s + 400)
        dx = gx[s:e, None] - gx[None, :]
        dy = gy[s:e, None] - gy[None, :]
        d = zk * np.sqrt(dx * dx + dy * dy)
        d[d == 0] = g["intrazonal_km"]
        t = row[s:e, None] * col[None, :] * np.exp(-d / g["decay_km"]) * d ** -a_pow
        rsum[s:e] = t.sum(axis=1)
        csum += t.sum(axis=0)
        tot += t.sum()
        dsum += (t * d).sum()
    rerr = np.max(np.abs(rsum[workers > 0] / workers[workers > 0] - 1))
    cerr = np.max(np.abs(csum[zjobs > 0] / zjobs[zjobs > 0] - 1))
    assert rerr < 2e-3 and cerr < 2e-3, f"factors do not balance: rows {rerr:.1e}, columns {cerr:.1e}"
    assert abs(tot / g["trips"] - 1) < 1e-4 and abs(dsum / tot - g["mean_trip_km"]) < 0.01
    print(f"  gravity: {nz:,} zones of {zm:.0f} m on a {w} x {h} grid; trips {tot:,.0f}, "
          f"mean {dsum / tot:.2f} km; worst row sum {rerr:.1e} off workers, column {cerr:.1e} off jobs; OK")


def water_mask(header, a):
    """The water mask (T-030, notes/T-004.md), checked: row offsets monotone and even, each row's
    run bounds strictly increasing (sorted, disjoint, non-touching) and within the grid."""
    wb = header.get("water")
    if wb is None:
        return None
    W, H = wb["size"]
    off, xs = a["water_row"].astype(np.int64), a["water_x"].astype(np.int64)
    assert len(off) == H + 1 and off[0] == 0 and off[-1] == len(xs), "water_row does not index water_x"
    d = np.diff(off)
    assert np.all(d >= 0) and np.all(d % 2 == 0), "a water row has an odd number of run bounds"
    assert xs.size == 0 or (xs.max() <= W), "a water run ends past the grid"
    same_row = np.repeat(np.arange(H), d)
    inc = xs[1:] > xs[:-1]
    assert np.all(inc | (same_row[1:] != same_row[:-1])), "water runs not sorted, disjoint and apart"
    pix = int((xs[1::2] - xs[0::2]).sum())
    share = pix / (W * H)
    assert abs(share - wb["water_share"]) < 1e-3, (share, wb["water_share"])
    return {"row": off, "x": xs, "size": (W, H), "origin_m": wb["origin_m"], "cell_m": wb["cell_m"],
            "runs": len(xs) // 2, "bytes": a["water_row"].nbytes + a["water_x"].nbytes, "water_share": share}


def is_water(m, x, y):
    """Point lookup exactly as notes/T-004.md describes it (outside the grid is land)."""
    c = math.floor((x - m["origin_m"][0]) / m["cell_m"])
    r = math.floor((y - m["origin_m"][1]) / m["cell_m"])
    W, H = m["size"]
    if not (0 <= c < W and 0 <= r < H):
        return False
    seg = m["x"][m["row"][r]:m["row"][r + 1]]
    return int(np.searchsorted(seg, c, side="right")) % 2 == 1


# Places the T-030 mask must get right (lon, lat), per city.
WATER_TESTS = {
    "nyc": {
        True: {"Hudson at 34th St": (-74.0150, 40.7500), "East River at the Williamsburg Br": (-73.9700, 40.7135),
               "Upper Bay": (-74.0450, 40.6700), "Jamaica Bay": (-73.8692, 40.6110), "Grassy Bay": (-73.7795, 40.6206),
               "Harlem River at 155th St": (-73.9339, 40.8288),
               "Newark Bay": (-74.1300, 40.6700), "Long Island Sound": (-73.4500, 41.0000),
               "Kensico Reservoir": (-73.7550, 41.0750), "Raritan Bay": (-74.1800, 40.4900),
               "Atlantic off Rockaway": (-73.8500, 40.5300)},
        False: {"Times Square": (-73.9855, 40.7580), "Roosevelt Island": (-73.9500, 40.7620),
                "Governors Island": (-74.0165, 40.6895), "Central Park": (-73.9660, 40.7820),
                "Hoboken": (-74.0300, 40.7440), "JFK": (-73.7781, 40.6413), "Staten Island": (-74.1500, 40.5800)},
    },
}


def check_water(city, header, a):
    m = water_mask(header, a)
    if m is None:
        print("  water: none in this pack")
        return
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    k = math.cos(math.radians(lat0))
    wrong = []
    for want, pts in WATER_TESTS.get(city, {}).items():
        for name, (lo, la) in pts.items():
            x, y = R_EARTH * math.radians(lo - lon0) * k, R_EARTH * math.radians(la - lat0)
            if is_water(m, x, y) != want:
                wrong.append(name)
    assert not wrong, f"water mask wrong at {wrong}"
    W, H = m["size"]
    print(f"  water: {W} x {H} pixels of {m['cell_m']:.0f} m, {m['runs']:,} runs, {m['bytes'] / 1e6:.2f} MB, "
          f"water {m['water_share']:.1%}; {sum(len(p) for p in WATER_TESTS.get(city, {}).values())} test places right; OK")


def check(city):
    header, a = load_pack(city)
    n = header["cells"]
    assert header["format"] == 1, header["format"]
    for name in ("h3", "x_m", "y_m", "pop", "jobs"):
        assert name in a and len(a[name]) == n, f"array {name} missing or wrong length"
    ids = a["h3"]
    assert np.all(ids[1:] > ids[:-1]), "h3 ids not sorted and unique"
    res = header["h3_res"]
    sample = ids[:: max(1, n // 2000)]
    cells = [h3.int_to_str(int(c)) for c in sample]
    assert all(h3.is_valid_cell(c) and h3.get_resolution(c) == res for c in cells), "bad h3 id"
    assert np.all((a["pop"] > 0) | (a["jobs"] > 0)), "empty cell in pack"
    assert np.all(a["pop"] >= 0) and np.all(a["jobs"] >= 0)
    for k in ("pop", "jobs"):
        s = float(a[k].astype(np.float64).sum())
        assert abs(s - header["totals"][k]) < 0.5, f"{k}: arrays sum {s} vs header {header['totals'][k]}"
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    worst = 0.0
    for i in range(0, n, max(1, n // 500)):
        lat, lon = h3.cell_to_latlng(h3.int_to_str(int(ids[i])))
        x = R_EARTH * math.radians(lon - lon0) * math.cos(math.radians(lat0))
        y = R_EARTH * math.radians(lat - lat0)
        worst = max(worst, abs(x - a["x_m"][i]), abs(y - a["y_m"][i]))
    assert worst < 1.0, f"x_m/y_m off by {worst:.2f} m from the h3 centre"
    print(f"{city}: format {header['format']}, {n:,} cells, pop {header['totals']['pop']:,}, "
          f"jobs {header['totals']['jobs']:,}; x {a['x_m'].min() / 1e3:.0f}..{a['x_m'].max() / 1e3:.0f} km, "
          f"y {a['y_m'].min() / 1e3:.0f}..{a['y_m'].max() / 1e3:.0f} km; max x/y error {worst:.3f} m; OK")
    if "commute_home" in a or "commute_work" in a:
        h, w = a["commute_home"].astype(np.float64), a["commute_work"].astype(np.float64)
        assert len(h) == n and len(w) == n and h.min() >= 0 and w.min() >= 0
        assert np.all(h <= a["pop"] * 1.0001 + 1e-3), "more home commuters than people in a cell"
        assert np.all(w <= a["jobs"] * 1.0001 + 1e-3), "more commuting jobs than jobs in a cell"
        for k, v in (("commute_home", h), ("commute_work", w)):
            assert abs(v.sum() - header["totals"][k]) < 1.0, k
        print(f"  commute: {h.sum():,.0f} home-end ({h.sum() / a['pop'].sum():.1%} of people), {w.sum():,.0f} "
              f"work-end ({w.sum() / a['jobs'].sum():.1%} of jobs); OK")
    check_gravity(header, a)
    check_water(city, header, a)
    return header, a


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "nyc")
