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
    with open(os.path.join(packs, f"{city}.json"), encoding="utf-8") as f:  # the sim writes raw UTF-8 (T-768)
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


def zone_side(a, nz):
    """Each zone's side in squares (T-852: `zone_size`, 2 for a 2x2 block; 1 without it)."""
    return a["zone_size"].astype(np.int64) if "zone_size" in a else np.ones(nz, np.int64)


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
    side = zone_side(a, nz)
    assert zone.max() < nz and (gx + side).max() <= w and (gy + side).max() <= h
    assert set(np.unique(side)) <= {1, 2} and (gx[side == 2] % 2 == 0).all() and (gy[side == 2] % 2 == 0).all(), "a block off the even squares"
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
    bad = int(((cx - gx[zone] < 0) | (cx - gx[zone] >= side[zone]) | (cy - gy[zone] < 0) | (cy - gy[zone] >= side[zone])).sum())
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
    # T-852: a pair with a block decays as the mean over their squares; distances are the centres'
    subs = [(0, 0)] if side.max() == 1 else [(u, v) for u in (0, 1) for v in (0, 1)]
    wsub = {q: np.where(side == 2, 0.25, 1.0 if q == (0, 0) else 0.0) for q in subs}
    for s in range(0, nz, 400):
        e = min(nz, s + 400)
        dx = gx[s:e, None] - gx[None, :]
        dy = gy[s:e, None] - gy[None, :]
        f = 0.0
        for qi in subs:
            for qj in subs:
                wq = wsub[qi][s:e, None] * wsub[qj][None, :]
                if not wq.any():
                    continue
                ex, ey = dx + qi[0] - qj[0], dy + qi[1] - qj[1]
                dq = zk * np.sqrt(ex * ex + ey * ey)
                dq[dq == 0] = g["intrazonal_km"]
                f = f + wq * np.exp(-dq / g["decay_km"]) * dq ** -a_pow
        if len(subs) == 1:
            d = dq
        else:
            hx, hy = 2 * dx + side[s:e, None] - side[None, :], 2 * dy + side[s:e, None] - side[None, :]
            d = zk * 0.5 * np.sqrt(hx * hx + hy * hy)
            d = np.where(d == 0, g["intrazonal_km"] * side[s:e, None], d)
        other = zr[s:e, None] != zr[None, :]
        t = row[s:e, None] * col[None, :] * f * np.where(other, bf, 1.0)
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
    "mum": {  # T-292: OSM water (Geofabrik India western zone), points taken from the water cache
        True: {"Mumbai Harbour": (72.8800, 18.9300), "Thane Creek off Vashi": (72.9810, 19.0400),
               "Thane Creek at Airoli": (72.9750, 19.1290), "Vasai Creek": (72.8300, 19.3260),
               "Ulhas river at Kalyan": (73.1130, 19.2360), "Vihar Lake": (72.9131, 19.1547), "Powai Lake": (72.9062, 19.1277),
               "Dharamtar creek": (72.9950, 18.7310), "Arabian Sea off Marine Drive": (72.8100, 18.9400)},
        False: {"Dadar station": (72.8428, 19.0192), "CSMT": (72.8355, 18.9400), "Thane station": (72.9762, 19.1860),
                "Vashi station": (72.9989, 19.0632), "Kalyan station": (73.1305, 19.2355), "Panvel": (73.1210, 18.9894),
                "Mumbai airport": (72.8656, 19.0896)},
    },
    "sao": {  # T-458: OSM water (Geofabrik Sudeste), points taken from the water cache
        True: {"Represa Billings": (-46.6329, -23.7702), "Represa de Guarapiranga": (-46.7285, -23.7186),
               "Represa do Rio Taiacupeba": (-46.2780, -23.5826), "Rio Tiete at Penha": (-46.5400, -23.5000),
               "Rio Pinheiros at Morumbi": (-46.6957, -23.5995), "Atlantic off Bertioga": (-46.10, -23.90)},
        False: {"Se station": (-46.6334, -23.5504), "Trianon-Masp station": (-46.6542, -23.5633),
                "Tatuape station": (-46.5766, -23.5403), "Tucuruvi station": (-46.6033, -23.4801),
                "Jabaquara station": (-46.6410, -23.6463), "Palmeiras-Barra Funda station": (-46.6675, -23.5255)},
    },
    "lag": {  # T-459: OSM water (Geofabrik Nigeria), lagoon points taken from the water cache
        True: {"Lagos Lagoon": (3.4492, 6.5129), "Lekki Lagoon": (4.1425, 6.5375), "Badagry Creek": (2.7562, 6.4473),
               "Epe Lagoon": (4.0384, 6.5529), "Ologe Lagoon": (3.1048, 6.4790), "Commodore Channel": (3.4002, 6.4148),
               "Atlantic off Victoria Island": (3.4200, 6.3950)},
        False: {"Yaba": (3.3786, 6.5158), "Ikeja": (3.3373, 6.5919), "Marina station": (3.3950, 6.4462),
                "Victoria Island": (3.4219, 6.4281), "Ikorodu": (3.5076, 6.6194), "Ota": (3.1977, 6.6894),
                "Murtala Muhammed airport": (3.3211, 6.5774), "Epe": (3.9841, 6.5841)},
    },
    "tyo": {  # T-420: OSM water (Geofabrik Kanto), river and lake points taken from the water cache
        True: {"Tokyo Bay off Kisarazu and Haneda": (139.90, 35.50), "Sumida at Oku": (139.73781, 35.77691),
               "Sumida at Kanegafuchi": (139.80253, 35.73861), "Tama at Futako-Tamagawa": (139.54408, 35.63761),
               "Edo at Matsudo": (139.88978, 35.83662), "Lake Inba (west)": (140.19632, 35.75514),
               "Lake Inba (north)": (140.24888, 35.79018), "Teganuma": (140.03787, 35.85774),
               "Kasumigaura (Nishiura)": (140.40592, 36.02831), "Kitaura": (140.56703, 36.01317),
               "Tone at Toride": (140.09440, 35.88376), "Tone at Sawara": (140.51394, 35.90064)},
        False: {"Tokyo Station": (139.7671, 35.6812), "Shinjuku Station": (139.7006, 35.6896),
                "Yokohama Station": (139.6221, 35.4660), "Omiya Station": (139.6239, 35.9063),
                "Chiba Station": (140.1233, 35.6131), "Haneda terminal 1": (139.7853, 35.5491),
                "Tsukuba Station": (140.1117, 36.0826), "Mito Station": (140.4767, 36.3708)},
    },
    "ber": {  # T-290: OSM water (Geofabrik Brandenburg with Berlin), points taken from the water cache
        True: {"Spree by the Reichstag": (13.37632, 52.52105), "Spree at the Oberbaumbruecke": (13.44487, 52.50182),
               "Havel at Spandau": (13.20524, 52.53462),
               "Grosser Wannsee": (13.1731, 52.4278), "Grosser Mueggelsee": (13.6501, 52.4368), "Tegeler See": (13.2550, 52.5746),
               "Templiner See": (13.0168, 52.3670), "Schwielowsee": (12.9542, 52.3360), "Scharmuetzelsee": (14.0569, 52.2473)},
        False: {"Brandenburger Tor": (13.3777, 52.5163), "Alexanderplatz": (13.4132, 52.5219), "Tempelhofer Feld": (13.4030, 52.4730),
                "Potsdam Hauptbahnhof": (13.0667, 52.3917), "BER terminal 1": (13.5104, 52.3644),
                "Frankfurt (Oder) station": (14.5465, 52.3365)},
    },
    "seo": {  # T-457: OSM water (Geofabrik South Korea), points by polylabel on the water cache
        True: {"Han by Yeouido": (126.94011, 37.53012), "Han at Jamsil": (127.09354, 37.52554),
               "Han by Hangang Bridge": (126.96542, 37.51516), "Paldang Lake": (127.29064, 37.50886),
               "Han at Gimpo": (126.78416, 37.61760), "Seokchon Lake": (127.10423, 37.51069)},
        False: {"Seoul City Hall": (126.9780, 37.5665), "Gangnam station": (127.0276, 37.4979),
                "Yeouido": (126.9240, 37.5250), "Ichon, north bank": (126.9960, 37.5250), "Banpo, south bank": (126.9960, 37.5050)},
    },
    "osa": {  # T-431: OSM water (Geofabrik Kansai and Chubu), river and lake points taken from the water cache
        True: {"Osaka Bay off Sakai": (135.30, 34.55), "Lake Biwa (south basin)": (135.88944, 35.02554),
               "Lake Biwa (north basin)": (136.14479, 35.34479), "Yodo below Juso": (135.46302, 34.70039),
               "Yodo at Hirakata": (135.62163, 34.80986), "Yamato at Sakai": (135.44846, 34.60778),
               "Muko river": (135.37409, 34.76692)},
        False: {"Osaka Station": (135.4959, 34.7025), "Kyoto Station": (135.7588, 34.9858),
                "Sannomiya Station": (135.1955, 34.6946), "Nara Station": (135.8197, 34.6809),
                "Kansai airport terminal 1": (135.2440, 34.4320), "Himeji Station": (134.6913, 34.8270)},
    },
    "ngo": {  # T-431: OSM water (Geofabrik Chubu and Kansai), river points taken from the water cache
        True: {"Ise Bay": (136.75, 34.85), "Kiso at Ichinomiya": (136.72987, 35.30114),
               "Kiso mouth": (136.74242, 35.03825), "Ibi river": (136.71409, 35.10212),
               "Nagara at Gifu": (136.80578, 35.44391), "Shonai river": (136.83735, 35.15626)},
        False: {"Nagoya Station": (136.8815, 35.1709), "Gifu Station": (136.7581, 35.4093),
                "Tsu Station": (136.5084, 34.7340), "Toyohashi Station": (137.3816, 34.7629),
                "Centrair terminal": (136.8164, 34.8584), "Sakae": (136.9088, 35.1700)},
    },
    "fuk": {  # T-432: OSM water (Geofabrik Kyushu and Chugoku); no named river polygon in the cache, so the sea
        True: {"Hakata Bay": (130.34, 33.64), "Dokai Bay": (130.79101, 33.88795), "Genkai Sea off Itoshima": (130.15, 33.65),
               "Ariake Sea off Omuta": (130.35, 33.02)},
        False: {"Hakata Station": (130.4207, 33.5897), "Tenjin": (130.3990, 33.5911), "Kokura Station": (130.8823, 33.8869),
                "Fukuoka Airport terminal": (130.4511, 33.5859), "Kurume Station": (130.5086, 33.3165)},
    },
    "spk": {  # T-432: OSM water (Geofabrik Hokkaido), river points taken from the water cache
        True: {"Ishikari at Tobetsu": (141.38137, 43.21861), "Toyohira at its mouth": (141.45534, 43.12916),
               "Ishikari Bay": (141.10, 43.30)},
        False: {"Sapporo Station": (141.3508, 43.0687), "Odori": (141.3545, 43.0606), "Otaru Station": (140.9935, 43.1979),
                "New Chitose terminal": (141.6923, 42.7882), "Ebetsu Station": (141.5366, 43.1036)},
    },
    "sdj": {  # T-432: OSM water (Geofabrik Tohoku), river points taken from the water cache
        True: {"Matsushima Bay": (141.06, 38.33), "Pacific off Sendai port": (141.10, 38.20),
               "Hirose at Nagamachi": (140.88679, 38.23838)},
        False: {"Sendai Station": (140.8824, 38.2601), "Kotodai Park": (140.8737, 38.2685), "Ishinomaki Station": (141.3022, 38.4295),
                "Sendai Airport terminal": (140.9230, 38.1375)},
    },
    "hij": {  # T-432: OSM water (Geofabrik Chugoku and Shikoku), Ota delta channels taken from the water cache
        True: {"Hiroshima Bay": (132.40, 34.31), "Ota floodway": (132.43757, 34.35587), "Kyobashi": (132.46134, 34.37563),
               "Motoyasu": (132.44934, 34.36168), "Temma": (132.42385, 34.36446)},
        False: {"Hiroshima Station": (132.4757, 34.3975), "Hondori": (132.4562, 34.3925), "Kure Station": (132.5654, 34.2463),
                "Fukuyama Station": (133.3624, 34.4893), "Hiroshima Airport terminal": (132.9194, 34.4361)},
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


def check(city, packs=PACKS):
    header, a = load_pack(city, packs)
    n = header["cells"]
    assert header["format"] == 1, header["format"]
    for name in ("h3", "x_m", "y_m", "pop", "jobs"):
        assert name in a and len(a[name]) == n, f"array {name} missing or wrong length"
    ids = a["h3"]
    res = header["h3_res"]
    mb = header.get("merged")  # T-044: res-8 parents merged within a zone, one cell per zone
    is8 = ((ids >> np.uint64(52)) & np.uint64(0xF)) == res - 1
    if mb is None:
        assert np.all(ids[1:] > ids[:-1]), "h3 ids not sorted and unique"
    else:
        key, z = ids & ~(np.uint64(0xF) << np.uint64(52)), a["zone"]
        assert np.all((key[1:] > key[:-1]) | ((key[1:] == key[:-1]) & (z[1:] > z[:-1]) & is8[1:])), \
            "h3 ids (resolution cleared) not sorted, or a non-merged id twice, or one zone's twice"
        assert mb["cells"] == n and int(is8.sum()) == mb["parents"], "merged block does not match the cells"
    sample = np.unique(np.concatenate([np.arange(0, n, max(1, n // 2000)), np.nonzero(is8)[0][:: max(1, int(is8.sum()) // 500)]]))
    cells = [h3.int_to_str(int(ids[i])) for i in sample]
    assert all(h3.is_valid_cell(c) and (h3.get_resolution(c) == res or (mb and h3.get_resolution(c) == res - 1))
               for c in cells), "bad h3 id"
    assert np.all((a["pop"] > 0) | (a["jobs"] > 0)), "empty cell in pack"
    assert np.all(a["pop"] >= 0) and np.all(a["jobs"] >= 0)
    for k in ("pop", "jobs"):
        s = float(a[k].astype(np.float64).sum())
        # float32 cells: a merge's re-summed cells drift from the rounded header by a fraction of a person a
        # million (fuz, 4.47M, was 0.50008 off, T-897)
        tol = max(1.0, 1e-6 * abs(header["totals"][k]))
        assert abs(s - header["totals"][k]) < tol, f"{k}: arrays sum {s} vs header {header['totals'][k]}"
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    worst = worst8 = worst9 = 0.0
    for i in sample:
        lat, lon = h3.cell_to_latlng(h3.int_to_str(int(ids[i])))
        x = R_EARTH * math.radians(lon - lon0) * math.cos(math.radians(lat0))
        y = R_EARTH * math.radians(lat - lat0)
        if is8[i]:  # a merged cell sits at its children's weighted centre, inside the parent
            worst8 = max(worst8, math.hypot(x - a["x_m"][i], y - a["y_m"][i]))
        elif mb and mb.get("sparse_groups") and max(abs(x - a["x_m"][i]), abs(y - a["y_m"][i])) >= 1.0:
            # T-466: touching sparse siblings merged under their busiest child's index
            worst9 = max(worst9, math.hypot(x - a["x_m"][i], y - a["y_m"][i]))
        else:
            worst = max(worst, abs(x - a["x_m"][i]), abs(y - a["y_m"][i]))
    assert worst < 1.0, f"x_m/y_m off by {worst:.2f} m from the h3 centre"
    # T-510: lone cells from across the parent's edge can pull it out to about two res-9 cells
    assert worst8 < 900.0, f"a merged cell {worst8:.0f} m from its parent's centre"
    assert worst9 < 700.0, f"merged siblings {worst9:.0f} m from their index's centre"
    if mb is not None and "zone_cells9" in a:
        assert int(a["zone_cells9"].astype(np.int64).sum()) == mb["cells_before"], "zone_cells9 does not sum to the res-9 cells"
    print(f"{city}: format {header['format']}, {n:,} cells, pop {header['totals']['pop']:,}, "
          f"jobs {header['totals']['jobs']:,}; x {a['x_m'].min() / 1e3:.0f}..{a['x_m'].max() / 1e3:.0f} km, "
          f"y {a['y_m'].min() / 1e3:.0f}..{a['y_m'].max() / 1e3:.0f} km; max x/y error {worst:.3f} m; OK")
    if mb is not None:
        print(f"  merged (T-044): {mb['cells_before']:,} res-{res} cells -> {n:,}, {mb['parents']:,} res-{res - 1} cells, "
              f"{mb.get('sparse_groups', 0):,} merged siblings (T-466); furthest from its parent's centre {worst8:.0f} m, "
              f"from its index's {worst9:.0f} m; OK")
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
    uni = header.get("university")
    if uni and uni.get("sites"):  # T-441: Fukuoka's first build had 73 campuses and no students
        s = float(a["university_enrollment"].astype(np.float64).sum()) if "university_enrollment" in a else 0.0
        assert s > 0, f"{uni['sites']} university sites but no students"
        print(f"  university: {uni['sites']} sites, {s:,.0f} students; OK")
    check_gravity(header, a)
    check_water(city, header, a)
    return header, a


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "nyc")
