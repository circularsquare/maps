"""WorldPop R2025A in place of GHS-POP for the countries in build.WORLDPOP, years build.WP_YEARS.

    python -I worldpop.py run 2025 [ISO3 ...]   # download, resample, bin -> logs/worldpop/
    python -I worldpop.py write                  # merge every result into data/
    python -I worldpop.py clean                  # delete the downloads in raw/worldpop/

Run it after build.py and after cities.py hist: both write GHSL numbers for these countries,
and `write` puts the WorldPop ones back over them. NOTES.md has why and the checks.

Each country's WorldPop 3 arc-second file is resampled with a sum onto GHS-POP's own 100 m
Mollweide grid (aligned 1 km blocks, same origin), so people are kept and everything downstream
is binned exactly as build.py does. Every person in a country's WorldPop file counts for that
country (WorldPop's LSIB borders); neighbours keep their GHSL people, and a 1 km block on a
border only sums this country's people. In SUB_YEARS the states, metros and cities of the
country are binned from the same raster.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import json
import math
import pickle
import sys
import time
import urllib.request

import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from affine import Affine
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script dir out
import build  # noqa: E402

HERE = build.HERE
RAW = os.path.join(HERE, "raw", "worldpop")
RES = os.path.join(HERE, "logs", "worldpop")
URL = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{y}/{ISO}/v1/100m/"
       "constrained/{iso}_pop_{y}_CN_100m_R2025A_v1.tif")
X0, Y0 = -18041000.0, 9000000.0          # GHS-POP grid origin; 1 km blocks start here
MOLL = "ESRI:54009"
WIN = 5000
NB = build.NBINS


def fetch(iso, year):
    os.makedirs(RAW, exist_ok=True)
    name = f"{iso.lower()}_pop_{year}_CN_100m_R2025A_v1.tif"
    path = os.path.join(RAW, name)
    if os.path.exists(path):
        return path
    url = URL.format(y=year, ISO=iso, iso=iso.lower())
    req = urllib.request.Request(url, headers={"User-Agent": "liveddensity-build/1.0"})
    tmp = path + ".part"
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    print(f"  downloaded {name} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
    return path


def src_total(path):
    tot = 0.0
    with rasterio.open(path) as ds:
        for _, w in ds.block_windows(1):
            a = ds.read(1, window=w).astype(np.float64)
            tot += a[a > 0].sum()
    return tot


def binof(d):
    return np.clip(np.floor((np.log10(d) - build.LO) / build.STEP).astype(np.int64), 0, NB - 1)


def run(year, isos):
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    keys, names, parts, part_cid, _ = build.load_countries(MOLL)
    kix = {k: i + 1 for i, k in enumerate(keys)}
    do_sub = year in build.SUB_YEARS
    adm1, fua, sub = build.sub_layers(keys, names)
    cities = pickle.load(open(os.path.join(HERE, "logs", "city_units.pkl"), "rb")) if do_sub else []
    for iso in isos:
        path = fetch(iso, year)
        raw_tot = src_total(path)
        ci = kix[iso]
        build.init_worker(path, parts, part_cid, len(keys), sub if do_sub else None)
        if do_sub:
            cu = [u for u in cities if u["parent"] == iso]
            cparts, cids = [], []
            for j, u in enumerate(cu, 1):
                cparts += u["parts"]
                cids += [j] * len(u["parts"])
            if cparts:
                g = shapely.from_wkb(cparts)
                build.G["layers"]["city"] = (g, shapely.STRtree(g), np.asarray(cids, dtype=np.int32))
        with rasterio.open(path) as src:
            l, b, r, t = transform_bounds(src.crs, MOLL, *src.bounds, densify_pts=101)
            c0 = int(math.floor((l - X0) / 1000)) * 10 - 10
            c1 = int(math.ceil((r - X0) / 1000)) * 10 + 10
            r0 = int(math.floor((Y0 - t) / 1000)) * 10 - 10
            r1 = int(math.ceil((Y0 - b) / 1000)) * 10 + 10
            tf = Affine(100, 0, X0 + c0 * 100, 0, -100, Y0 - r0 * 100)
            H, W = r1 - r0, c1 - c0
            vrt = WarpedVRT(src, crs=MOLL, transform=tf, width=W, height=H,
                            resampling=Resampling.sum, src_nodata=src.nodata, nodata=-1.0,
                            dtype="float64")
            build.G["ds"] = vrt
            h100 = np.zeros(NB)
            h1k = np.zeros(NB)
            na, nf = len(sub["a1_parent"]), len(sub["fua_parent"])
            A100, A1K = np.zeros((na, NB)), np.zeros((na, NB))
            F100, F1K = np.zeros((nf, NB)), np.zeros((nf, NB))
            C100, C1K = np.zeros((len(cu) + 1 if do_sub else 1, NB)), np.zeros((len(cu) + 1 if do_sub else 1, NB))
            tot, moved = 0.0, 0.0
            a_parent = np.asarray(sub["a1_parent"])
            f_parent = np.asarray(sub["fua_parent"])
            for row0 in range(0, H, WIN):
                for col0 in range(0, W, WIN):
                    h, w = min(WIN, H - row0), min(WIN, W - col0)
                    pop = vrt.read(1, window=rasterio.windows.Window(col0, row0, w, h))
                    pop[~(pop > 0)] = 0.0
                    rr, cc = np.nonzero(pop)
                    if len(rr) == 0:
                        continue
                    p = pop[rr, cc]
                    tot += p.sum()
                    b100 = binof(p * 100.0)
                    blocks = pop.reshape(h // 10, 10, w // 10, 10).sum(axis=(1, 3))
                    b1k = binof(blocks[rr // 10, cc // 10])
                    del pop, blocks
                    h100 += np.bincount(b100, weights=p, minlength=NB)
                    h1k += np.bincount(b1k, weights=p, minlength=NB)
                    if not do_sub:
                        continue
                    a = build.rasterize(row0, col0, h, w, "a")[rr, cc].astype(np.int64)
                    bad = a_parent[a] != ci
                    if bad.any():
                        moved += p[bad].sum()
                        a[bad] = build.nearest_admin1(row0, col0, h, w, rr[bad], cc[bad],
                                                      np.full(int(bad.sum()), ci))
                    assert (a_parent[a] == ci).all()
                    np.add.at(A100, (a, b100), p)
                    np.add.at(A1K, (a, b1k), p)
                    f = build.rasterize(row0, col0, h, w, "f")[rr, cc].astype(np.int64)
                    keep = (f > 0) & (f_parent[f] == ci)
                    np.add.at(F100, (f[keep], b100[keep]), p[keep])
                    np.add.at(F1K, (f[keep], b1k[keep]), p[keep])
                    if "city" in build.G["layers"]:
                        c = build.rasterize(row0, col0, h, w, "city")[rr, cc].astype(np.int64)
                        k = c > 0
                        np.add.at(C100, (c[k], b100[k]), p[k])
                        np.add.at(C1K, (c[k], b1k[k]), p[k])
            vrt.close()
        build.G.pop("layers", None)
        out = {"iso": iso, "year": year, "raw_total": raw_tot, "total": tot,
               "h100": build.pack(h100), "h1k": build.pack(h1k), "h100_f": h100.tolist(),
               "h1k_f": h1k.tolist()}
        msg = (f"{iso} {year}: WorldPop file {raw_tot:,.0f}, resampled {tot:,.0f} "
               f"({100 * (tot / raw_tot - 1):+.3f}%)")
        if do_sub:
            units = {}
            for i in np.nonzero(a_parent == ci)[0]:
                units[adm1[i - 1]["id"]] = (A100[i], A1K[i])
            for i in np.nonzero(f_parent == ci)[0]:
                if F100[i].sum() > 0:
                    units[fua[i - 1]["id"]] = (F100[i], F1K[i])
            for j, u in enumerate(cu, 1):
                units[u["id"]] = (C100[j], C1K[j])
            out["units"] = {uid: {"h100": v[0].tolist(), "h1k": v[1].tolist()} for uid, v in units.items()}
            states = sum(A100[i].sum() for i in np.nonzero(a_parent == ci)[0])
            assert A100[0].sum() == 0
            msg += (f"; states {states:,.0f} (diff {states - tot:.4f}); {moved:,.0f} moved to the "
                    f"nearest state; cities: " + ", ".join(f"{u['name']} {C100[j].sum():,.0f}"
                                                             for j, u in enumerate(cu, 1)))
        with open(os.path.join(RES, f"{iso}_{year}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f)
        print(msg + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def write():
    t0 = time.time()
    with open(build.OUT, encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for iso in build.WORLDPOP:
        for y in build.WP_YEARS:
            p = os.path.join(RES, f"{iso}_{y}.json")
            if not os.path.exists(p):
                sys.exit(f"missing {p}: run `worldpop.py run {y} {iso}` first")
            results[(iso, y)] = json.load(open(p, encoding="utf-8"))
    for (iso, y), r in results.items():
        c = data["countries"][iso]
        c["pop"][str(y)] = int(round(r["total"]))
        c["source"] = build.WP_SOURCE
        data["hist"]["100m"][str(y)][iso] = r["h100"]
        data["hist"]["1km"][str(y)][iso] = r["h1k"]
    # countries that left the list go back to GHSL only after build.py is rerun; say so
    stale = [k for k, c in data["countries"].items() if c.get("source") and k not in build.WORLDPOP]
    if stale:
        print("WARNING: still marked WorldPop but no longer in build.WORLDPOP:", stale)
    out = {"source": build.SOURCE, "citation": build.CITATION, "sources": build.SOURCES}
    out.update({k: v for k, v in data.items() if k not in out})
    build.write_atomic(build.OUT, out)
    print(f"wrote {build.OUT} ({os.path.getsize(build.OUT) / 1e6:.2f} MB)")

    # sub-national units, SUB_YEARS only
    sp_path = os.path.join(HERE, "logs", "sub_pops.json")
    sub_pops = json.load(open(sp_path, encoding="utf-8"))
    ip = os.path.join(HERE, "data", "sub", "index.json")
    index = json.load(open(ip, encoding="utf-8"))
    keys, names, _, _, _ = build.load_countries(MOLL)
    adm1, fua = __import__("subunits").load_units(keys, names)
    meta = {u["id"]: u for u in adm1 + fua}
    for u in pickle.load(open(os.path.join(HERE, "logs", "city_units.pkl"), "rb")):
        meta[u["id"]] = u
    for (iso, y), r in results.items():
        if y not in build.SUB_YEARS:
            continue
        ys = str(y)
        units = {uid: (np.array(v["h100"]), np.array(v["h1k"])) for uid, v in r["units"].items()}
        st = [uid for uid in units if uid.startswith("adm1:")]
        vals = np.array([units[uid][0].sum() for uid in st])
        ints = dict(zip(st, build.largest_remainder(vals, int(round(r["total"])))))
        city_pop = {}
        for uid, (h100, _) in units.items():
            if uid.startswith("city:"):      # cities are not in sub_pops (cities.py keeps them apart)
                city_pop[uid] = int(round(h100.sum()))
                continue
            sub_pops.setdefault(uid, {})[ys] = int(ints[uid]) if uid in ints else int(round(h100.sum()))
        assert sum(sub_pops[uid][ys] for uid in st) == data["countries"][iso]["pop"][ys]
        fp = os.path.join(HERE, "data", "sub", f"{iso}.json")
        d = json.load(open(fp, encoding="utf-8"))
        for g, ix in (("100m", 0), ("1km", 1)):
            d["hist"][g][ys] = {uid: build.pack(v[ix]) for uid, v in units.items()
                                if v[0].sum() >= 0.5 or uid.startswith("city:")}
        build.write_atomic(fp, d)
        # index: this country's entries rebuilt in place
        new = []
        for uid in units:
            u = meta[uid]
            kind = u["kind"]
            if kind == "city":
                new.append({"id": uid, "name": u["name"], "kind": "city", "parent": iso,
                            "pop": {ys: city_pop[uid]}})
            else:
                pp = sub_pops.get(uid, {})
                ref = pp.get(str(build.SUB_REF), pp[max(pp)] if pp else 0)
                if ref >= 50000:
                    new.append({"id": uid, "name": u["name"], "kind": kind, "parent": iso,
                                "pop": dict(sorted(pp.items()))})
        order = {"state": 0, "metro": 1, "city": 2}
        new.sort(key=lambda e: order[e["kind"]])
        pos = [i for i, e in enumerate(index["units"]) if e["parent"] == iso]
        at = pos[0] if pos else len(index["units"])
        rest = [e for e in index["units"] if e["parent"] != iso]
        at -= sum(1 for i in pos if i < at)
        index["units"] = rest[:at] + new + rest[at:]
    build.write_atomic(sp_path, sub_pops)
    build.write_atomic(ip, index)
    print(f"wrote data/sub ({len(index['units'])} units in the index) in {time.time() - t0:.0f} s")


def clean():
    if os.path.isdir(RAW):
        for f in os.listdir(RAW):
            os.remove(os.path.join(RAW, f))
        os.rmdir(RAW)
        print("deleted", RAW)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "run":
        year = int(sys.argv[2])
        assert year in build.WP_YEARS, year
        isos = sys.argv[3:] or build.WORLDPOP
        run(year, isos)
    elif cmd == "write":
        write()
    elif cmd == "clean":
        clean()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
