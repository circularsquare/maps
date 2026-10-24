"""T-290: Germany's inputs for a tier 0 city (Berlin first), through tier0's hooks.

- **Outline**: whole Kreise (kreisfreie Städte and Landkreise), from BKG's VG250-EW (31 December,
  dl-de/by-2-0), land parts only.
- **Municipalities**: VG250-EW's Gemeinden with their population (EWZ, the Länder's
  Fortschreibung on the Zensus 2022 base). Berlin is one Gemeinde.
- **Population**: the Zensus 2022 100 m grid (residents per square), each square's people to its
  H3 cell, then scaled to each Gemeinde's EWZ.
- **Jobs**: the Länder's Pendlerrechnung (Pendleratlas, statistical offices of the Länder with the
  BA), employed people by Gemeinde of work: in-commuters plus those working in their home
  Gemeinde (EIP + IOP). All employed people, not only those insured.
- **Curve**: the Pendlerrechnung's flows between Gemeinden, each Gemeinde's ten largest in and out
  (curves/t290_curve.py).

All open. Cached in data/raw/tier0/germany/.
"""
import json
import os

import numpy as np
import pandas as pd
import shapely
from shapely import make_valid
from shapely.geometry import mapping, shape

import build_city as b
import england

VG250_URL = "https://daten.gdz.bkg.bund.de/produkte/vg/vg250-ew_ebenen_1231/aktuell/vg250-ew_12-31.utm32s.gpkg.ebenen.zip"
VG250_IN_ZIP = "vg250-ew_12-31.utm32s.gpkg.ebenen/vg250-ew_ebenen_1231/DE_VG250.gpkg"
ZENSUS_URL = "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip"
ZENSUS_CSV = "Zensus2022_Bevoelkerungszahl_100m-Gitter.csv"
PA_URL = "https://pendleratlas.statistikportal.de/data/csv/{y}/{y}_{kind}_L{land}.csv"
PA_YEAR = 2024


def raw(name):
    return os.path.join(b.RAW, "tier0", "germany", name)


def _vg250(layer, kreise):
    import pyogrio
    path = england._get(VG250_URL, raw(os.path.basename(VG250_URL)))
    df = pyogrio.read_dataframe(f"/vsizip/{path}/{VG250_IN_ZIP}", layer=layer,
                                columns=["ARS", "GEN", "GF", "EWZ"], where="GF = 4")
    df = df[df.ARS.str[:5].isin(kreise)].to_crs(4326)
    return df


def _cached_geojson(path, make):
    if not os.path.exists(path):
        feats = make()
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


def kreise_boundary(city, kreise):
    """Write the Kreise as one GeoJSON for tier0's `boundary_file`; its path."""
    out = raw(f"{city}_kreise.geojson")

    def make():
        df = _vg250("vg250_krs", kreise)
        return [{"type": "Feature", "properties": {"ars": r.ARS, "name": r.GEN, "ewz": int(r.EWZ)},
                 "geometry": mapping(make_valid(r.geometry))} for r in df.itertuples()]
    _cached_geojson(out, make)
    return out


def gemeinden(city, kreise):
    """[(ARS, name, polygon, population)] for the Gemeinden of the Kreise."""
    def make():
        df = _vg250("vg250_gem", kreise)
        return [{"type": "Feature", "properties": {"ars": r.ARS, "name": r.GEN, "ewz": int(r.EWZ)},
                 "geometry": mapping(make_valid(r.geometry))} for r in df.itertuples()]
    feats = _cached_geojson(raw(f"{city}_gemeinden.geojson"), make)
    return [(f["properties"]["ars"], f["properties"]["name"], make_valid(shape(f["geometry"])), float(f["properties"]["ewz"]))
            for f in feats]


def zensus_squares(city, boundary):
    """Zensus 2022 100 m squares inside the boundary: lat, lon, residents. Cached small."""
    import zipfile
    from pyproj import Transformer
    cache = raw(f"zensus2022_100m_{city}.npz")
    if not os.path.exists(cache):
        path = england._get(ZENSUS_URL, raw(os.path.basename(ZENSUS_URL)))
        t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        w, s, e, n = boundary.bounds
        xs, ys = t.transform([w, w, e, e], [s, n, s, n])
        with zipfile.ZipFile(path) as z:
            df = pd.read_csv(z.open(ZENSUS_CSV), sep=";", usecols=["x_mp_100m", "y_mp_100m", "Einwohner"])
        df["Einwohner"] = pd.to_numeric(df["Einwohner"], errors="coerce")
        print(f"  Zensus 2022 100 m: {len(df):,} squares in Germany, {df.Einwohner.sum():,.0f} residents, "
              f"{df.Einwohner.isna().sum():,} without a figure", flush=True)
        df = df[(df.x_mp_100m >= min(xs) - 100) & (df.x_mp_100m <= max(xs) + 100) &
                (df.y_mp_100m >= min(ys) - 100) & (df.y_mp_100m <= max(ys) + 100) & (df.Einwohner > 0)]
        lon, lat = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True).transform(
            df.x_mp_100m.to_numpy(float), df.y_mp_100m.to_numpy(float))
        shapely.prepare(boundary)
        ok = shapely.contains_xy(boundary, lon, lat)
        np.savez_compressed(cache, lat=lat[ok], lon=lon[ok], pop=df.Einwohner.to_numpy(float)[ok])
    z = np.load(cache)
    return z["lat"], z["lon"], z["pop"]


def zensus_population(city, boundary, sig, kreise):
    """Zensus 2022 100 m residents into the signal cells (a square whose H3 cell has no signals goes
    to the nearest signal cell), scaled to each Gemeinde's EWZ; tier0's `population` hook."""
    import h3
    from scipy.spatial import cKDTree
    lat, lon, pop = zensus_squares(city, boundary)
    index = {int(c): i for i, c in enumerate(sig["h3"])}
    cell = np.array([index.get(h3.str_to_int(h3.latlng_to_cell(la, lo, 9)), -1) for la, lo in zip(lat, lon)])
    miss = cell < 0
    if miss.any():
        k = np.cos(np.radians(np.mean(lat)))
        tree = cKDTree(np.c_[sig["lon"] * k, sig["lat"]])
        cell[miss] = tree.query(np.c_[lon[miss] * k, lat[miss]])[1]
    grid = np.bincount(cell, weights=pop, minlength=len(sig["h3"]))
    gs = gemeinden(city, kreise)
    out = england.spread_totals(sig["lat"], sig["lon"], [(g[0], g[2]) for g in gs], {g[0]: g[3] for g in gs}, grid)
    print(f"  Zensus 2022 100 m: {len(pop):,} squares, {pop.sum():,.0f} residents ({miss.sum():,} squares, "
          f"{pop[miss].sum():,.0f} people to a neighbouring cell); scaled to {len(gs)} Gemeinden: {out.sum():,.0f}", flush=True)
    return out, ("Zensus 2022 residents per 100 m grid square (Statistisches Bundesamt), scaled to each Gemeinde's "
                 "population on 31 December (BKG VG250-EW)")


def pendler(kind, lands, year=PA_YEAR):
    """A Pendleratlas table (Pendlerrechnung der Länder) for the Länder, as one DataFrame."""
    frames = []
    for land in lands:
        p = england._get(PA_URL.format(y=year, kind=kind, land=land), raw(f"pa_{year}_{kind}_L{land}.csv"))
        frames.append(pd.read_csv(p, sep=";", dtype=str))
    return pd.concat(frames, ignore_index=True)


def workplace_jobs(lands):
    """{ARS: employed people working in the Gemeinde} = in-commuters + those working where they live."""
    t = pendler("EZ", lands)
    num = lambda v: float(v) if str(v).replace(".", "", 1).isdigit() else 0.0  # the table writes a suppressed count as "-"
    return {r.ARS: num(r.EIP) + num(r.IOP) for r in t.itertuples()}


# Eurostat LFS lfsa_ehomp, Germany 2025, employed 15-64: 13.0% usually work from home, 11.7% sometimes;
# judged at about 0.6 and 0.2 of their days, as Paris's (T-290)
WFH = 0.10


def nuts_kreise(nuts_ids):
    """The Kreise (5-digit ARS) of NUTS-3 units, by VG250's own NUTS column (a Kreis is a NUTS-3 unit)."""
    import pyogrio
    path = england._get(VG250_URL, raw(os.path.basename(VG250_URL)))
    df = pyogrio.read_dataframe(f"/vsizip/{path}/{VG250_IN_ZIP}", layer="vg250_krs", columns=["ARS", "NUTS"],
                                read_geometry=False, where="GF = 4")
    by = dict(zip(df.NUTS, df.ARS.str[:5]))
    missing = [u for u in nuts_ids if u not in by]
    if missing:
        raise ValueError(f"no Kreis in VG250 for NUTS {missing}")
    return [by[u] for u in nuts_ids]


def plan_hooks(row):
    """T-555: Berlin's country keys for any planned row of German NUTS-3 units (Kreise). Not
    carried: the curve (Berlin's own fit on the Pendlerrechnung's flows)."""
    cid, units = row["id"], row["units"]["DEU"]
    kreise = lambda: nuts_kreise(units)  # noqa: E731
    return {
        "boundary_file": lambda: kreise_boundary(cid, kreise()),
        "population": lambda city, boundary, sig: zensus_population(city, boundary, sig, kreise()),
        "jobs": lambda city, boundary, sig, w: pendler_jobs(city, boundary, sig, w, kreise()),
        "jobs_url": "https://pendleratlas.statistikportal.de/",
        "wfh": WFH,
    }


def pendler_jobs(city, boundary, sig, weight, kreise):
    """Jobs per signal cell: the Pendlerrechnung's employed people by Gemeinde of work, spread inside
    each Gemeinde by weight; tier0's `jobs` hook."""
    gs = gemeinden(city, kreise)
    jobs = workplace_jobs(sorted({k[:2] for k in kreise}))
    missing = [g[1] for g in gs if g[0] not in jobs]
    out = england.spread_totals(sig["lat"], sig["lon"], [(g[0], g[2]) for g in gs], jobs, weight)
    print(f"  Pendlerrechnung {PA_YEAR} employed by Gemeinde of work: {len(gs)} Gemeinden ({len(missing)} without: "
          f"{', '.join(missing[:5])}), {out.sum():,.0f} jobs", flush=True)
    return out, f"Pendlerrechnung der Länder {PA_YEAR} (Pendleratlas), employed people by Gemeinde of work"
