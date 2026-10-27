"""Per-country histograms of how many people live at each population density, from GHS-POP.

    python -I build.py 2020 raw/ghsl/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif

Reads one GHS-POP 100 m Mollweide epoch, assigns every populated cell to a Natural Earth
country (cells outside every polygon go to the nearest country), bins each cell's people by
its density at 100 m and at its aligned 1 km block, and merges that epoch into data/hist.json.
NOTES.md has the sources, decisions and checks.
"""
import os

THREADS = os.environ.get("LD_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = THREADS
os.environ.setdefault("GDAL_CACHEMAX", "512")

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
import shapely
from pyproj import Transformer
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
NE = os.path.join(os.path.dirname(HERE), "religiondots", "data", "geo",
                  "ne_10m_admin_0_countries.geojson")
OUT = os.path.join(HERE, "data", "hist.json")
STATE = os.path.join(HERE, "logs")

LO, STEP, NBINS = -1.0, 0.02, 350
WIN = 5000          # window side in 100 m cells (multiple of 10, so 1 km blocks stay whole)
MARGIN = 1000       # extra cells around a window searched for the nearest country (100 km)
SUB_YEARS = {2025}  # epochs that also get state and metro histograms (data/sub)
SUB_REF = 2025      # the index lists units with at least 50,000 people in this epoch
SOURCE = "GHS-POP R2023A"
CITATION = ("Schiavina, M., Freire, S., MacManus, K. (2023): GHS-POP R2023A - GHS population grid "
            "multitemporal (1975-2030). European Commission, Joint Research Centre (JRC). "
            "doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE")

# Countries whose WP_YEARS come from WorldPop R2025A instead of GHS-POP: GHS-POP squeezes their
# people onto too little land (old or coarse census inputs; NOTES.md has the evidence).
# worldpop.py writes them over this build's output, so run it after build.py and after
# cities.py hist (both write GHSL numbers for these countries too). Add a country with one line
# here, then run worldpop.py.
WORLDPOP = [
    "COG",   # Republic of the Congo: GHSL 12 departments, 2007; WorldPop 57 units, 2023 census
    "SOM",   # Somalia: 74 districts, 2005 estimate; WorldPop 2021 estimate
    "GNQ",   # Equatorial Guinea: 7 units, 2015; WorldPop 18 units
    "SSD",   # South Sudan: 78 units, 2008 (pre-independence); WorldPop 28 units, 2022
    "COD",   # DR Congo: 181 units, 2008; WorldPop 164 units, 2020
    "ERI",   # Eritrea: 6 units, 2012 (both)
    "ZMB",   # Zambia: 150 units, 2010; WorldPop 116 units, 2022
    "SLE",   # Sierra Leone: 160 units, 2004; WorldPop 208 units, 2021
    "NER",   # Niger: 66 units, 2012; WorldPop 266 units, 2023
    "YEM",   # Yemen: 333 units, 2004 census; WorldPop 2016 projection
    "AFG",   # Afghanistan: 400 districts, 2011 estimate; WorldPop 34 provinces, 2021 estimate
]
WP_YEARS = (2020, 2025)
WP_SOURCE = "WorldPop R2025A"
WP_CITATION = ("Bondarenko M., Priyatikanto R., Tejedor-Garavito N., Zhang W., McKeen T., "
               "Cunningham A., Woods T., Hilton J., Cihan D., Nosatiuk B., Brinkhoff T., Tatem A., "
               "Sorichetta A. (2025): Constrained estimates of 2015-2030 total number of people per "
               "grid square at a resolution of 3 arc (approximately 100m at the equator) R2025A "
               "version v1. WorldPop, University of Southampton. doi:10.5258/SOTON/WP00839")
SOURCES = {SOURCE: CITATION, WP_SOURCE: WP_CITATION}

# Natural Earth features whose own code is not the one we want. Value None drops the polygon,
# so its few cells (if any) go to the nearest country instead.
KEY_FIX = {
    "SOL": "SOM",   # Somaliland -> Somalia
    "CYN": "CYP",   # Northern Cyprus -> Cyprus
    "CNM": "CYP",   # Cyprus UN buffer zone
    "ESB": "CYP",   # Dhekelia base area
    "WSB": "CYP",   # Akrotiri base area
    "KOS": "XKX",   # Kosovo (user-assigned code used by the EU, World Bank)
    "KAS": "IND",   # Siachen Glacier, Indian-administered
    "KAB": "KAZ",   # Baikonur lease
    "USG": "CUB",   # Guantanamo Bay lease
    "ALD": "FIN",   # Aland, counted inside Finland by the UN
    "IOA": "AUS",   # Christmas and Cocos Islands
    "BRI": "BRA",   # Brazilian Island
    "SPI": None, "BRT": None, "BJN": None, "SER": None, "SCR": None,
    "PGA": None, "CLP": None, "ATC": None, "CSI": None,
}

# Polygon parts moved between countries: (from, to, a lon/lat point inside the part, what).
# Natural Earth draws de facto control; these follow the UN, so totals line up with UN WPP.
MOVE_PARTS = [
    ("RUS", "UKR", (34.1, 45.0), "Crimea and Sevastopol"),
]
# Pieces cut out of a polygon by a lon/lat box: (from, to, (lon0, lat0, lon1, lat1), what).
# This Natural Earth copy (5.1+) draws Morocco down to Mauritania and keeps only the strip east
# of the berm as W. Sahara; the box is everything south of 27 deg 40' N, Morocco's own border.
CUTS = [
    ("MAR", "ESH", (-18.0, 20.0, -8.6666, 27.6667), "Moroccan-held Western Sahara"),
]

NAMES = {
    "UKR": "Ukraine",
    "USA": "United States", "RUS": "Russia", "COD": "DR Congo", "COG": "Republic of the Congo",
    "CIV": "Côte d'Ivoire", "TUR": "Türkiye", "KOR": "South Korea", "PRK": "North Korea",
    "LAO": "Laos", "SWZ": "Eswatini", "CPV": "Cabo Verde", "CZE": "Czechia", "BRN": "Brunei",
    "GMB": "Gambia", "FSM": "Micronesia", "FRO": "Faroe Islands", "FLK": "Falkland Islands",
    "HMD": "Heard and McDonald Islands", "ATF": "French Southern Lands",
    "SGS": "South Georgia", "UMI": "US Minor Outlying Islands", "VIR": "US Virgin Islands",
    "VGB": "British Virgin Islands", "MAC": "Macao", "HKG": "Hong Kong", "PSE": "Palestine",
    "XKX": "Kosovo", "ESH": "Western Sahara", "STP": "São Tomé and Príncipe",
    "IOT": "British Indian Ocean Territory", "SOM": "Somalia", "CYP": "Cyprus",
    "TLS": "Timor-Leste", "MKD": "North Macedonia", "SRB": "Serbia", "TZA": "Tanzania",
    "BHS": "Bahamas", "ATA": "Antarctica", "VAT": "Vatican City",
}


def country_key(p):
    a = p["ADM0_A3"]
    if a in KEY_FIX:
        return KEY_FIX[a]
    if p["ISO_A3"] != "-99":
        return p["ISO_A3"]
    if p["ISO_A3_EH"] != "-99":
        return p["ISO_A3_EH"]
    return a


def load_countries(crs_wkt):
    """Natural Earth polygons in the raster's CRS, as (keys, names, part WKBs, part key index)."""
    with open(NE, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    tr = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)

    def proj(xy):
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    keys, names, parts, part_cid, crossers, moved = [], {}, [], [], [], []
    for feat in feats:
        p = feat["properties"]
        k = country_key(p)
        if k is None:
            continue
        g = shapely.geometry.shape(feat["geometry"])
        # a ring with a lon jump over 180 would come out as a band across the whole map
        for poly in getattr(g, "geoms", [g]):
            xs = np.asarray(poly.exterior.coords)[:, 0]
            if len(xs) > 1 and np.abs(np.diff(xs)).max() > 180:
                crossers.append(k)
        if k not in names:
            keys.append(k)
        if k not in names or p["ADM0_A3"] not in KEY_FIX:   # a merged-in piece never names it
            names[k] = NAMES.get(k, p["NAME_LONG"])
        pieces = []
        for piece in shapely.get_parts(g):
            pk = k
            for src, dst, (lon, lat), _ in MOVE_PARTS:
                if k == src and piece.contains(shapely.Point(lon, lat)):
                    pk = dst
                    moved.append((src, dst, round(piece.area, 3)))
            for src, dst, box, _ in CUTS:
                if k == src and piece.intersects(shapely.box(*box)):
                    cut = shapely.intersection(piece, shapely.box(*box))
                    pieces.append((cut, dst))
                    moved.append((src, dst, round(cut.area, 3)))
                    piece = shapely.difference(piece, shapely.box(*box))
            pieces.append((piece, pk))
        for piece, pk in pieces:
            gp = shapely.make_valid(shapely.transform(shapely.segmentize(piece, 0.05), proj))
            for part in shapely.get_parts(gp):
                if part.geom_type in ("Polygon", "MultiPolygon") and not part.is_empty:
                    parts.append(shapely.to_wkb(part))
                    part_cid.append(pk)
    for k in {m[1] for m in MOVE_PARTS + CUTS} - set(keys):
        keys.append(k)
        names[k] = NAMES[k]
    cid = np.array([keys.index(k) + 1 for k in part_cid], dtype=np.int16)   # 0 = no country
    print("moved parts:", moved, flush=True)
    return keys, names, parts, cid, crossers


# ---------------------------------------------------------------- worker

G = {}


def init_worker(tif, parts, part_cid, nc, sub=None):
    """sub: None, or dict with a1_parts, a1_uid, a1_parent, a1_whole, fua_parts, fua_uid,
    fua_parent (unit ids are 1-based; parents are country ids, 1-based)."""
    G["ds"] = rasterio.open(tif)
    G["nc"] = nc
    G["layers"] = {}

    def layer(name, wkbs, ids):
        geoms = shapely.from_wkb(wkbs)
        G["layers"][name] = (geoms, shapely.STRtree(geoms), np.asarray(ids, dtype=np.int32))
    layer("c", parts, part_cid)
    G["geoms"], G["tree"], G["cid"] = G["layers"]["c"]
    G["sub"] = sub
    if sub:
        layer("a", sub["a1_parts"], sub["a1_uid"])
        layer("f", sub["fua_parts"], sub["fua_uid"])
        G["a_parent"] = np.asarray(sub["a1_parent"], dtype=np.int64)
        G["f_parent"] = np.asarray(sub["fua_parent"], dtype=np.int64)
        G["a_whole"] = np.asarray(sub["a1_whole"], dtype=np.int64)   # country id -> unit or 0
        G["a_trees"] = {}


def rasterize(row0, col0, h, w, name="c"):
    ds = G["ds"]
    geoms, tree, ids = G["layers"][name]
    tf = rasterio.windows.transform(rasterio.windows.Window(col0, row0, w, h), ds.transform)
    x0, y1 = tf * (0, 0)
    x1, y0 = tf * (w, h)
    idx = tree.query(shapely.box(x0, y0, x1, y1))
    out = np.zeros((h, w), dtype=np.int16 if name == "c" else np.int32)
    if len(idx):
        clipped = shapely.clip_by_rect(geoms[idx], x0 - 1000, y0 - 1000, x1 + 1000, y1 + 1000)
        shapes = [(g, int(c)) for g, c in zip(clipped, ids[idx])
                  if not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon")]
        if shapes:
            rasterio.features.rasterize(shapes, out=out, transform=tf)
    return out


def nearest_admin1(row0, col0, h, w, rr, cc, cty):
    """Admin-1 unit for cells (rr, cc in the window) whose own admin-1 is missing or belongs to
    another country than their country `cty`: the nearest admin-1 cell of that same country."""
    ds = G["ds"]
    out = np.zeros(len(rr), dtype=np.int64)
    r0, c0 = max(row0 - MARGIN, 0), max(col0 - MARGIN, 0)
    r1, c1 = min(row0 + h + MARGIN, ds.height), min(col0 + w + MARGIN, ds.width)
    big = None
    for ci in np.unique(cty):
        sel = cty == ci
        if G["a_whole"][ci]:
            out[sel] = G["a_whole"][ci]
            continue
        if big is None:
            big = rasterize(r0, c0, r1 - r0, c1 - c0, "a")
            bigp = G["a_parent"][big]
        mask = bigp == ci
        mr, mc = rr[sel] + (row0 - r0), cc[sel] + (col0 - c0)
        got = np.zeros(len(mr), dtype=np.int64)
        far = np.ones(len(mr), dtype=bool)
        if mask.any():
            ind = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            nr, nn = ind[0][mr, mc], ind[1][mr, mc]
            del ind
            got = big[nr, nn].astype(np.int64)
            far = np.hypot(nr - mr, nn - mc) > MARGIN
        if far.any():
            if ci not in G["a_trees"]:
                geoms, _, ids = G["layers"]["a"]
                k = np.nonzero(G["a_parent"][ids] == ci)[0]
                G["a_trees"][ci] = (shapely.STRtree(geoms[k]), ids[k])
            t, kid = G["a_trees"][ci]
            x, y = ds.transform * (cc[sel][far] + 0.5, rr[sel][far] + 0.5)
            got[far] = kid[t.query_nearest(shapely.points(np.asarray(x), np.asarray(y)),
                                           all_matches=False)[1]]
        out[sel] = got
    return out


def sparse_bincount(idx, w, n):
    h = np.bincount(idx, weights=w, minlength=n)
    nz = np.flatnonzero(h)
    return nz, h[nz]


def process(win):
    row0, col0, h, w = win
    ds = G["ds"]
    nc = G["nc"]
    pop = ds.read(1, window=rasterio.windows.Window(col0, row0, w, h)).astype(np.float64)
    pop[~(pop > 0)] = 0.0              # nodata (-200) and NaN
    res = {"win": win, "total": 0.0, "near": 0.0, "far": 0.0, "near_by": {}}
    rr, cc = np.nonzero(pop)
    if len(rr) == 0:
        return res
    p = pop[rr, cc]
    res["total"] = float(p.sum())

    cid = rasterize(row0, col0, h, w)
    c = cid[rr, cc].astype(np.int64)
    miss = c == 0
    if miss.any():
        # nearest assigned cell within the window plus a margin
        r0, c0 = max(row0 - MARGIN, 0), max(col0 - MARGIN, 0)
        r1, c1 = min(row0 + h + MARGIN, ds.height), min(col0 + w + MARGIN, ds.width)
        big = rasterize(r0, c0, r1 - r0, c1 - c0)
        mr, mc = rr[miss] + (row0 - r0), cc[miss] + (col0 - c0)
        got = np.zeros(len(mr), dtype=np.int64)
        far = np.ones(len(mr), dtype=bool)
        if (big > 0).any():
            ind = ndimage.distance_transform_edt(big == 0, return_distances=False,
                                                 return_indices=True)
            nr, nn = ind[0][mr, mc], ind[1][mr, mc]
            del ind
            dist = np.hypot(nr - mr, nn - mc)
            got = big[nr, nn].astype(np.int64)
            far = dist > MARGIN
        if far.any():
            # nothing near: the nearest polygon anywhere (lone islands)
            x, y = ds.transform * (cc[miss][far] + 0.5, rr[miss][far] + 0.5)
            pts = shapely.points(np.asarray(x), np.asarray(y))
            nearest = G["tree"].query_nearest(pts, all_matches=False)[1]
            got[far] = G["cid"][nearest]
        c[miss] = got
        pm = p[miss]
        res["near"] = float(pm[~far].sum())
        res["far"] = float(pm[far].sum())
        by = np.bincount(got, weights=pm, minlength=nc + 1)
        res["near_by"] = {int(i): float(v) for i, v in enumerate(by) if v > 0}

    def binof(d):
        b = np.floor((np.log10(d) - LO) / STEP).astype(np.int64)
        return np.clip(b, 0, NBINS - 1)

    b100 = binof(p * 100.0)
    blocks = pop.reshape(h // 10, 10, w // 10, 10).sum(axis=(1, 3))
    b1k = binof(blocks[rr // 10, cc // 10])
    del pop, blocks
    res["h100"] = np.bincount(c * NBINS + b100, weights=p, minlength=(nc + 1) * NBINS)
    res["h1k"] = np.bincount(c * NBINS + b1k, weights=p, minlength=(nc + 1) * NBINS)

    sub = G["sub"]
    if sub:
        # admin-1: always inside the cell's own country
        a = rasterize(row0, col0, h, w, "a")[rr, cc].astype(np.int64)
        bad = G["a_parent"][a] != c
        res["a_moved"] = float(p[bad].sum())
        if bad.any():
            a[bad] = nearest_admin1(row0, col0, h, w, rr[bad], cc[bad], c[bad])
        assert (G["a_parent"][a] == c).all()
        na = (len(G["a_parent"])) * NBINS
        res["a100"] = sparse_bincount(a * NBINS + b100, p, na)
        res["a1k"] = sparse_bincount(a * NBINS + b1k, p, na)
        # metros: only the cells of the metro's own country
        f = rasterize(row0, col0, h, w, "f")[rr, cc].astype(np.int64)
        keep = (f > 0) & (G["f_parent"][f] == c)
        res["f_clipped"] = float(p[(f > 0) & ~keep].sum())
        nf = len(G["f_parent"]) * NBINS
        res["f100"] = sparse_bincount(f[keep] * NBINS + b100[keep], p[keep], nf)
        res["f1k"] = sparse_bincount(f[keep] * NBINS + b1k[keep], p[keep], nf)
    return res


# ---------------------------------------------------------------- main

def pack(row):
    v = np.rint(row).astype(np.int64)
    nz = np.nonzero(v)[0]
    if len(nz) == 0:
        return {"o": 0, "v": []}
    return {"o": int(nz[0]), "v": v[nz[0]:nz[-1] + 1].tolist()}


def write_json(year, keys, names, h100, h1k):
    data = None
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("bins") != {"lo": LO, "step": STEP, "n": NBINS}:
            data = None
    if data is None:
        data = {"hist": {"100m": {}, "1km": {}}, "countries": {}}
    y = str(year)
    pops = h100.sum(axis=1)
    for i, k in enumerate(keys):
        c = data["countries"].setdefault(k, {"name": names[k], "pop": {}})
        c["name"] = names[k]
        c["pop"][y] = int(round(pops[i]))
        if int(year) in WP_YEARS:
            c.pop("source", None)      # GHSL again until worldpop.py is run
    data["hist"]["100m"][y] = {k: pack(h100[i]) for i, k in enumerate(keys) if pops[i] > 0}
    data["hist"]["1km"][y] = {k: pack(h1k[i]) for i, k in enumerate(keys) if pops[i] > 0}
    # drop countries with nobody in any epoch
    data["countries"] = {k: v for k, v in sorted(data["countries"].items())
                         if any(n > 0 for n in v["pop"].values())}
    for c in data["countries"].values():
        c["pop"] = dict(sorted(c["pop"].items()))
    years = sorted(int(t) for t in data["hist"]["100m"])
    out = {"source": SOURCE, "citation": CITATION, "sources": SOURCES,
           "years": years, "grids": ["100m", "1km"],
           "bins": {"lo": LO, "step": STEP, "n": NBINS}, "countries": data["countries"],
           "hist": {g: dict(sorted(data["hist"][g].items())) for g in ("100m", "1km")}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUT)


def write_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def largest_remainder(vals, total):
    """Round vals to ints that add up to `total` exactly."""
    fl = np.floor(vals).astype(np.int64)
    short = int(total - fl.sum())
    if short > 0:
        fl[np.argsort(-(vals - np.floor(vals)))[:short]] += 1
    elif short < 0:
        fl[np.argsort(vals - np.floor(vals))[:-short]] -= 1
    return fl


def write_sub(year, keys, names, cpop, adm1, fua, A100, A1K, F100, F1K):
    """data/sub/index.json (units over 50,000 people in SUB_REF) and data/sub/<ISO3>.json."""
    y = str(year)
    allp = os.path.join(STATE, "sub_pops.json")      # every unit's people, every epoch
    pops = {}
    if os.path.exists(allp):
        with open(allp, encoding="utf-8") as f:
            pops = json.load(f)
    units = adm1 + fua
    upop = np.concatenate([A100[1:].sum(axis=1), F100[1:].sum(axis=1)])
    ints = np.rint(upop).astype(np.int64)
    # states add up to their country exactly
    for ci, k in enumerate(keys):
        ix = [i for i, u in enumerate(adm1) if u["parent"] == k]
        if ix:
            ints[ix] = largest_remainder(upop[ix], int(round(cpop[ci])))
    for u, n in zip(units, ints):
        pops.setdefault(u["id"], {})[y] = int(n)
    write_atomic(allp, pops)

    index = []
    for u in units:
        if u["parent"] is None:
            continue
        pp = pops.get(u["id"], {})
        ref = pp.get(str(SUB_REF), pp[max(pp)] if pp else 0)
        if ref >= 50000:
            index.append({"id": u["id"], "name": u["name"], "kind": u["kind"],
                          "parent": u["parent"], "pop": dict(sorted(pp.items()))})
    write_atomic(os.path.join(HERE, "data", "sub", "index.json"), {"units": index})

    by = {}
    for i, u in enumerate(adm1):
        by.setdefault(u["parent"], []).append((u["id"], A100[i + 1], A1K[i + 1]))
    for i, u in enumerate(fua):
        if u["parent"]:
            by.setdefault(u["parent"], []).append((u["id"], F100[i + 1], F1K[i + 1]))
    for k, rows in by.items():
        path = os.path.join(HERE, "data", "sub", f"{k}.json")
        d = {"hist": {"1km": {}, "100m": {}}}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        d["hist"]["100m"][y] = {uid: pack(h) for uid, h, _ in rows if h.sum() >= 0.5}
        d["hist"]["1km"][y] = {uid: pack(h) for uid, _, h in rows if h.sum() >= 0.5}
        d["hist"] = {g: dict(sorted(d["hist"][g].items())) for g in ("1km", "100m")}
        write_atomic(path, d)
    return len(index)


def sub_layers(keys, names):
    """Sub-national units (subunits.py) as rasterizable parts with 1-based unit ids."""
    sys.path.insert(0, HERE)     # -I leaves the script's folder off the path
    import subunits
    adm1, fua = subunits.load_units(keys, names)
    kix = {k: i + 1 for i, k in enumerate(keys)}
    sub = {"a1_parts": [], "a1_uid": [], "fua_parts": [], "fua_uid": [],
           "a1_parent": [0] + [kix[u["parent"]] for u in adm1],
           "fua_parent": [0] + [kix.get(u["parent"], 0) for u in fua],
           "a1_whole": [0] * (len(keys) + 1)}
    for i, u in enumerate(adm1, 1):
        sub["a1_parts"] += u["parts"]
        sub["a1_uid"] += [i] * len(u["parts"])
        if u.get("whole"):
            sub["a1_whole"][kix[u["parent"]]] = i
    for i, u in enumerate(fua, 1):
        sub["fua_parts"] += u["parts"]
        sub["fua_uid"] += [i] * len(u["parts"])
    return adm1, fua, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("tif")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bbox", type=float, nargs=4, help="lon0 lat0 lon1 lat1: test run, no output")
    a = ap.parse_args()

    t0 = time.time()
    ds = rasterio.open(a.tif)
    print(f"{a.tif}: {ds.width}x{ds.height} {ds.dtypes[0]} nodata={ds.nodata} "
          f"blocks={ds.block_shapes[0]}", flush=True)
    print(f"transform {tuple(ds.transform)[:6]}", flush=True)
    tf = ds.transform
    assert abs(tf.a - 100) < 1e-6 and abs(tf.e + 100) < 1e-6, "not a 100 m grid"
    assert ds.width % 10 == 0 and ds.height % 10 == 0
    keys, names, parts, part_cid, crossers = load_countries(ds.crs.to_wkt())
    nc = len(keys)
    print(f"{nc} countries, {len(parts)} polygon parts; rings over 180 deg: {crossers} "
          f"({time.time() - t0:.0f}s)", flush=True)

    adm1, fua, sub = sub_layers(keys, names)
    na, nf = (len(adm1) + 1) * NBINS, (len(fua) + 1) * NBINS
    print(f"{len(adm1)} admin-1 units, {len(fua)} metros", flush=True)

    wins = [(r, c, min(WIN, ds.height - r), min(WIN, ds.width - c))
            for r in range(0, ds.height, WIN) for c in range(0, ds.width, WIN)]
    if a.bbox:
        tr = Transformer.from_crs("EPSG:4326", ds.crs.to_wkt(), always_xy=True)
        xs, ys = tr.transform([a.bbox[0], a.bbox[2], a.bbox[0], a.bbox[2]],
                              [a.bbox[1], a.bbox[1], a.bbox[3], a.bbox[3]])
        (r0, r1), (c0, c1) = rasterio.transform.rowcol(tf, [min(xs), max(xs)], [max(ys), min(ys)])
        wins = [w for w in wins if w[0] < r1 and w[0] + w[2] > r0 and w[1] < c1 and w[1] + w[3] > c0]
    ds.close()

    state_path = os.path.join(STATE, f"state_{a.year}.npz")
    # a resumed run must use the same countries and windows
    sig = json.dumps([keys, sorted(KEY_FIX.items(), key=str), MOVE_PARTS, CUTS, WIN, MARGIN, a.tif,
                      [u["id"] for u in adm1], [u["parent"] for u in adm1],
                      [u["id"] for u in fua], [u["parent"] for u in fua]])
    acc = {"h100": np.zeros((nc + 1) * NBINS), "h1k": np.zeros((nc + 1) * NBINS),
           "a100": np.zeros(na), "a1k": np.zeros(na), "f100": np.zeros(nf), "f1k": np.zeros(nf),
           "near_by": np.zeros(nc + 1)}
    done = set()
    tot = {"total": 0.0, "near": 0.0, "far": 0.0, "a_moved": 0.0, "f_clipped": 0.0}
    if not a.bbox and os.path.exists(state_path):
        with np.load(state_path, allow_pickle=True) as z:   # closed again, so save() can replace it
            s = {k: z[k] for k in z.files}
        if str(s.get("sig", "")) == sig:
            for k in acc:
                acc[k] = s[k]
            done = set(map(tuple, s["done"].tolist()))
            tot = dict(s["tot"].item())
            print(f"resuming: {len(done)} windows done", flush=True)
    todo = [w for w in wins if w not in done]
    print(f"{len(wins)} windows, {len(todo)} to do, {a.workers} workers", flush=True)

    def save():
        tmp = state_path + ".tmp.npz"
        np.savez(tmp, sig=np.array(sig), done=np.array(sorted(done)),
                 tot=np.array(tot, dtype=object), **acc)
        os.replace(tmp, state_path)

    t1 = time.time()
    last = t1
    do_sub = a.year in SUB_YEARS or bool(a.bbox)
    with mp.Pool(a.workers, init_worker, (a.tif, parts, part_cid, nc, sub if do_sub else None)) as pool:
        for i, res in enumerate(pool.imap_unordered(process, todo, chunksize=1), 1):
            done.add(tuple(res["win"]))
            for k in tot:
                tot[k] += res.get(k, 0.0)
            for ci, v in res["near_by"].items():
                acc["near_by"][ci] += v
            if "h100" in res:
                acc["h100"] += res["h100"]
                acc["h1k"] += res["h1k"]
                for k in ("a100", "a1k", "f100", "f1k"):
                    ix, v = res[k]
                    acc[k][ix] += v
            now = time.time()
            if now - last > 60 or i == len(todo):
                el = now - t1
                print(f"{i}/{len(todo)} windows, {el / 60:.1f} min, eta {el / i * (len(todo) - i) / 60:.1f} min, "
                      f"pop so far {tot['total']:,.0f}", flush=True)
                if not a.bbox:
                    save()
                last = now

    near_by = acc["near_by"]
    H100 = acc["h100"].reshape(nc + 1, NBINS)
    H1K = acc["h1k"].reshape(nc + 1, NBINS)
    A100, A1K = acc["a100"].reshape(-1, NBINS), acc["a1k"].reshape(-1, NBINS)
    F100, F1K = acc["f100"].reshape(-1, NBINS), acc["f1k"].reshape(-1, NBINS)
    if do_sub:
        assert A100[0].sum() == 0, "people left without a state"
        # states add up to their country
        apar = np.array(sub["a1_parent"])
        asum = np.bincount(apar, weights=A100.sum(axis=1), minlength=nc + 1)
        worst = np.max(np.abs(asum[1:] - H100[1:].sum(axis=1)))
        print(f"states vs countries: largest difference {worst:.6f} people; "
              f"{tot['a_moved']:,.0f} people moved to the nearest state of their own country; "
              f"{tot['f_clipped']:,.0f} people inside a metro but over its country's border",
              flush=True)
        assert worst < 1.0
    assert H100[0].sum() == 0, "people left without a country"
    H100, H1K = H100[1:], H1K[1:]
    print(f"raster total {tot['total']:,.0f}; histogram total {H100.sum():,.0f} / {H1K.sum():,.0f}")
    print(f"reassigned to nearest country: {tot['near'] + tot['far']:,.0f} "
          f"({tot['far']:,.0f} of them more than {MARGIN * 100 // 1000} km from any polygon)")
    top = np.argsort(-near_by[1:])[:25]
    print("reassigned by country: " + ", ".join(f"{keys[i]} {near_by[i + 1]:,.0f}" for i in top
                                                 if near_by[i + 1] > 0))
    pops = H100.sum(axis=1)
    order = np.argsort(-pops)
    print("top: " + ", ".join(f"{keys[i]} {pops[i]:,.0f}" for i in order[:30]))
    summary = {"year": a.year, "total": tot, "keys": keys,
               "pop": {k: float(pops[i]) for i, k in enumerate(keys)},
               "near_by": {keys[i]: float(near_by[i + 1]) for i in range(nc) if near_by[i + 1] > 0},
               "minutes": (time.time() - t0) / 60}
    if a.bbox:
        print(json.dumps({k: round(v) for k, v in summary["pop"].items() if v > 0}))
        ap_ = A100[1:].sum(axis=1)
        print("states:", [(adm1[i]["name"], adm1[i]["parent"], round(ap_[i]))
                          for i in np.argsort(-ap_)[:15] if ap_[i] > 0])
        fp_ = F100[1:].sum(axis=1)
        print("metros:", [(fua[i]["name"], fua[i]["parent"], round(fp_[i]), round(fua[i]["pop2015"]))
                          for i in np.argsort(-fp_)[:15] if fp_[i] > 0])
        return
    with open(os.path.join(STATE, f"summary_{a.year}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    write_json(a.year, keys, names, H100, H1K)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    if do_sub:
        n = write_sub(a.year, keys, names, pops, adm1, fua, A100, A1K, F100, F1K)
        print(f"wrote data/sub ({n} units in the index) in {(time.time() - t0) / 60:.1f} min",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
