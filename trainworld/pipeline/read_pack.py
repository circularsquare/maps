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
    # T-162: zones are (square, region); a pack without zone_region is one region, factor 1
    zr = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(nz, np.int64)
    assert len(zr) == nz, "zone_region is not one per zone"
    bf = float(g.get("border_factor", 1.0))
    assert bf > 0, bf
    assert len(np.unique((gx * h + gy) * 256 + zr)) == nz, "two zones share a grid square and a region"
    if "region" in a:
        bad = int((a["region"].astype(np.int64) != zr[zone]).sum())
        assert bad == 0, f"{bad} cells are in a zone of another region"
        assert zr.max() < len(header["regions"]["ids"]), "zone region outside the header's list"
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
    tot = dsum = cross = 0.0
    for s in range(0, nz, 400):
        e = min(nz, s + 400)
        dx = gx[s:e, None] - gx[None, :]
        dy = gy[s:e, None] - gy[None, :]
        d = zk * np.sqrt(dx * dx + dy * dy)
        d[d == 0] = g["intrazonal_km"]
        other = zr[s:e, None] != zr[None, :]
        t = row[s:e, None] * col[None, :] * np.exp(-d / g["decay_km"]) * d ** -a_pow * np.where(other, bf, 1.0)
        rsum[s:e] = t.sum(axis=1)
        csum += t.sum(axis=0)
        tot += t.sum()
        dsum += (t * d).sum()
        cross += t[other].sum()
    joint = "zone_home" in a
    if joint:
        # T-276: a joint gravity over every zone pack: each zone's trips kept in the pack are its
        # rows' and columns' sums here, and never more than its workers and jobs
        kept_h, kept_w = a["zone_home"].astype(np.float64), a["zone_work"].astype(np.float64)
        assert (kept_h <= workers * 1.002 + 1e-3).all() and (kept_w <= zjobs * 1.002 + 1e-3).all(), "a zone keeps more than it has"
        workers, zjobs = kept_h, kept_w
    rerr = np.max(np.abs(rsum[workers > 0] / workers[workers > 0] - 1))
    cerr = np.max(np.abs(csum[zjobs > 0] / zjobs[zjobs > 0] - 1))
    assert rerr < 2e-3 and cerr < 2e-3, f"factors do not balance: rows {rerr:.1e}, columns {cerr:.1e}"
    assert abs(tot / g["trips"] - 1) < 1e-4 and (joint or abs(dsum / tot - g["mean_trip_km"]) < 0.01)
    if joint:
        j = g["joint"]
        print(f"  joint gravity (T-276) over {', '.join(j['packs'])}: {j['commuters_out']:,.0f} commuters to other packs, {j['commuters_in']:,.0f} from them")
    squares = len(np.unique(gx * h + gy))
    names = header.get("regions", {}).get("names", [])
    print(f"  gravity: {nz:,} zones of {zm:.0f} m on a {w} x {h} grid ({squares:,} squares, {nz - squares} split by a "
          f"region border); trips {tot:,.0f}, mean {dsum / tot:.2f} km; worst row sum {rerr:.1e} off workers, column "
          f"{cerr:.1e} off jobs; OK")
    print(f"  regions: {', '.join(names) if names else 'one'}; border factor {bf:g}; trips between regions "
          f"{cross:,.0f} ({cross / tot:.2%})")


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
    "phl": {
        True: {"Delaware at Penn's Landing": (-75.1350, 39.9460),
               "Schuylkill by 30th Street": (-75.1795, 39.9560),
               "Delaware River by Wilmington": (-75.4980, 39.7150),
               "Chesapeake Bay by Cecil": (-76.0150, 39.4350)},
        False: {"Philadelphia City Hall": (-75.1636, 39.9526),
                "30th Street Station": (-75.1820, 39.9553),
                "Camden City Hall": (-75.1196, 39.9450),
                "Wilmington station": (-75.5518, 39.7373)},
    },
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
    "tor": {  # T-277: StatCan lakes and the land outside the cartographic shorelines
        True: {"Lake Ontario off Toronto": (-79.3800, 43.6000), "Toronto Harbour": (-79.3750, 43.6380),
               "Hamilton Harbour": (-79.8500, 43.2900), "Lake Simcoe": (-79.4000, 44.3300),
               "Lake Ontario off Oshawa": (-78.8500, 43.8100)},
        False: {"Toronto City Hall": (-79.3832, 43.6534), "Billy Bishop airport": (-79.3962, 43.6285),
                "Pearson": (-79.6248, 43.6777), "Hamilton City Hall": (-79.8711, 43.2557),
                "Union Station": (-79.3806, 43.6453)},
    },
    "lon": {  # T-041: OSM water (sea polygons + Geofabrik England); first checked on the water-only pack
        True: {"Thames by Waterloo Bridge": (-0.1170, 51.5085), "Thames off Greenwich": (-0.0094, 51.4857),
               "Thames at Gravesend": (0.3700, 51.4480), "Royal Victoria Dock": (0.0284, 51.5066),
               "Royal Albert Dock": (0.0580, 51.5061), "Queen Mary Reservoir": (-0.4602, 51.4160),
               "King George V Reservoir": (-0.0197, 51.6441), "William Girling Reservoir": (-0.0256, 51.6294)},
        False: {"Trafalgar Square": (-0.1281, 51.5080), "Buckingham Palace": (-0.1419, 51.5014),
                "Mudchute, Isle of Dogs": (-0.0110, 51.4920), "Heathrow": (-0.4543, 51.4700),
                "Stratford station": (-0.0035, 51.5416), "Greenwich Observatory": (-0.0015, 51.4769)},
    },
    "par": {  # T-289: OSM water (Geofabrik Ile-de-France)
        True: {"Seine by the Pont des Arts": (2.3375, 48.8582), "Seine by the Pont d'Iena": (2.2925, 48.8605),
               "Bassin de la Villette": (2.3761, 48.8867), "Etang de Saint-Quentin": (2.0180, 48.7870),
               "Lac Inferieur, Bois de Boulogne": (2.2611, 48.8635)},
        False: {"Notre-Dame": (2.3499, 48.8530), "Eiffel Tower": (2.2945, 48.8584), "Place de la Concorde": (2.3212, 48.8656),
                "Gare du Nord": (2.3553, 48.8809), "Chateau de Versailles": (2.1204, 48.8049), "Roissy CDG terminal 2": (2.5700, 49.0040)},
    },
    "sha": {  # T-291: OSM water (Geofabrik Shanghai), points checked on the mask
        True: {"Huangpu off the Bund": (121.4889, 31.2384), "Dianshan Lake": (120.9684, 31.1328), "Dishui Lake": (121.9434, 30.8999),
               "Yangtze, Baoshan to Changxing": (121.55, 31.43), "Yangtze, Changxing to Chongming": (121.60, 31.52)},
        False: {"People's Square": (121.4737, 31.2304), "Lujiazui": (121.5050, 31.2400), "Pudong airport": (121.8080, 31.1500),
                "Hongqiao station": (121.3200, 31.1940), "Chongming, Chengqiao": (121.3990, 31.6250), "Hengsha island": (121.84, 31.33)},
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
        # RAC counts jobs held by residents, not unique people, and is newer than the
        # census population. New housing / multiple jobs can exceed POP20 locally.
        assert np.isfinite(h).all() and np.isfinite(w).all(), "non-finite commute weight"
        assert h.sum() <= a["pop"].astype(np.float64).sum(), "regional home commuters exceed population"
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
