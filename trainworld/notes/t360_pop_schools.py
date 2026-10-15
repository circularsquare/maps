"""T-360: tier 0's London population and schools against Census 2021 by LSOA and GIAS.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t360_pop_schools.py [stage a pack folder]

Population: the stage a signals (GHSL scaled to district totals; and GHSL alone) against Census
2021 usual residents by LSOA spread inside each LSOA by GHSL. Schools: the pack's school pupils
(OSM sites, population x the national 0-14 share x 0.9) against GIAS open establishments'
pupils on roll at their Easting/Northing. "Misplaced" as t287_compare.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")

import json
import sys

import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import make_valid, union_all
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import england  # noqa: E402
from read_pack import load_pack  # noqa: E402

OUT = os.path.join(ROOT, "..", "artifacts", "trainworld", "T-360-pop-schools.json")
GIAS = england.raw("gias_edubasealldata20261014.csv")


def miss(a, b_):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    return 0.5 * float(np.abs(a / a.sum() - b_ / b_.sum()).sum())


def main(args):
    origin = b.CITIES["lon"]["origin"]
    sig = dict(np.load(os.path.join(b.WORK, "lon.tier0_signals.npz")))
    with open(england.raw("lon_itl3.geojson"), encoding="utf-8") as f:
        boundary = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
    with open(england.raw("lad_dec2023_uk_bgc.geojson"), encoding="utf-8") as f:
        lads = [(ft["properties"]["LAD23NM"], make_valid(shape(ft["geometry"]))) for ft in json.load(f)["features"]]
    lads = [x for x in lads if boundary.contains(x[1].representative_point())]
    nl = len(lads)

    def zones(lat, lon):
        lon0, lat0 = origin
        x = b.R_EARTH * np.radians(lon - lon0) * np.cos(np.radians(lat0))
        y = b.R_EARTH * np.radians(lat - lat0)
        return np.floor(x / 2000).astype(np.int64) * 100000 + np.floor(y / 2000).astype(np.int64)

    res = {}
    # ---------------- population
    units = england.lsoas("lon", boundary)
    census, _ = england.lsoa_population("lon", boundary, sig)
    ui, _ = england.cell_units(sig["lat"], sig["lon"], units)
    ui = np.where(ui < 0, len(units), ui)
    zk = zones(sig["lat"], sig["lon"])
    _, zi = np.unique(zk, return_inverse=True)
    li, _ = england.cell_units(sig["lat"], sig["lon"], lads)
    li = np.where(li < 0, 0, li)
    pop = {}
    for name, v in (("tier 0 (GHSL scaled to districts)", sig["pop"]), ("GHSL alone", sig["pop_ghsl"])):
        pop[name] = dict(total=float(v.sum()), lsoa=miss(np.bincount(ui, weights=v), np.bincount(ui, weights=census)),
                         zone=miss(np.bincount(zi, weights=v), np.bincount(zi, weights=census)),
                         district=miss(np.bincount(li, weights=v, minlength=nl), np.bincount(li, weights=census, minlength=nl)))
    pop["census"] = dict(total=float(census.sum()))
    # per-LSOA error spread, by population density decile of the census
    lp_t, lp_c = np.bincount(ui, weights=sig["pop"]), np.bincount(ui, weights=census)
    area = np.bincount(ui, minlength=len(units) + 1).astype(float)
    dens = lp_c / np.maximum(area, 1)
    ok = (lp_c > 0) & (area > 0)
    dec = pd.qcut(dens[ok], 5, labels=False)
    pop["by_density_quintile"] = [dict(q=int(q), census=float(lp_c[ok][dec == q].sum()), tier0=float(lp_t[ok][dec == q].sum()),
                                       err=float(lp_t[ok][dec == q].sum() / lp_c[ok][dec == q].sum() - 1),
                                       median_abs_lsoa_err=float(np.median(np.abs(lp_t[ok][dec == q] / lp_c[ok][dec == q] - 1))))
                                  for q in range(5)]
    res["pop"] = pop
    print(json.dumps(pop, indent=1), flush=True)

    # ---------------- schools
    packs = args[0] if args else os.path.join(ROOT, "data", "packs")
    h, a = load_pack("lon", packs)
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
    g = pd.read_csv(GIAS, encoding="cp1252", low_memory=False,
                    usecols=["EstablishmentStatus (name)", "TypeOfEstablishment (name)", "PhaseOfEducation (name)", "NumberOfPupils",
                             "StatutoryLowAge", "StatutoryHighAge", "Easting", "Northing"])
    g = g[(g["EstablishmentStatus (name)"].str.startswith("Open")) & g.Easting.notna() & (g.Easting > 0)]
    lon_, lat_ = Transformer.from_crs(27700, 4326, always_xy=True).transform(g.Easting.to_numpy(float), g.Northing.to_numpy(float))
    import shapely
    shapely.prepare(boundary)
    inn = shapely.contains_xy(boundary, lon_, lat_)
    g, lat_, lon_ = g[inn], lat_[inn], lon_[inn]
    pupils = g.NumberOfPupils.fillna(0).to_numpy(float)
    sch = dict(gias_sites=int(len(g)), gias_with_pupils=int((pupils > 0).sum()), gias_pupils=float(pupils.sum()),
               gias_by_type={k: float(v) for k, v in g.assign(p=pupils).groupby("TypeOfEstablishment (name)").p.sum().sort_values(ascending=False).head(12).items()},
               gias_by_phase={k: float(v) for k, v in g.assign(p=pupils).groupby("PhaseOfEducation (name)").p.sum().sort_values(ascending=False).items()},
               pack_pupils=float(a["school_enrollment"].sum()), pack_sites=int((a["school_enrollment"] > 0).sum()))
    pz, gz = zones(ll[:, 0], ll[:, 1]), zones(lat_, lon_)
    keys = np.union1d(pz, gz)
    sch["zone"] = miss(np.bincount(np.searchsorted(keys, pz), weights=a["school_enrollment"].astype(float), minlength=len(keys)),
                       np.bincount(np.searchsorted(keys, gz), weights=pupils, minlength=len(keys)))
    pl, _ = england.cell_units(ll[:, 0], ll[:, 1], lads)
    gl_, _ = england.cell_units(lat_, lon_, lads)
    sch["district"] = miss(np.bincount(np.where(pl < 0, 0, pl), weights=a["school_enrollment"].astype(float), minlength=nl),
                           np.bincount(np.where(gl_ < 0, 0, gl_), weights=pupils, minlength=nl))
    # pupils a resident of each age group: what the rule's 0.9 x share 0-14 implies
    sch["pupils_per_resident_gias"] = sch["gias_pupils"] / float(a["pop"].sum())
    sch["pupils_per_resident_pack"] = sch["pack_pupils"] / float(a["pop"].sum())
    res["schools"] = sch
    print(json.dumps(sch, indent=1), flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])
