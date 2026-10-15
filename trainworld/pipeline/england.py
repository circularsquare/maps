"""T-288: England's inputs for a tier 0 city (London first). Stage a uses only what tier 0 takes
from a country: the outline as whole ITL3 units and district population totals; stage b (OA
population, BRES jobs, the curve fitted on 2011 flows, GIAS, HESA) replaces single inputs here.

- **Outline**: ONS International Territorial Level 3 (January 2025) boundaries, generalised 20 m,
  clipped to the coast (BGC), 182 UK units; the city config lists the units it takes.
- **District totals**: Census 2021 usual residents (TS001, Nomis NM_2021_1) per local authority
  district (April 2023), on ONS's Local Authority Districts (December 2023) BGC polygons. ITL3 units
  are whole districts, so a district is in the pack when its representative point is.

Both from ONS Open Geography and Nomis, open (OGL; Contains OS data). Cached in data/raw/tier0/.
"""
import io
import json
import os

import numpy as np
import pandas as pd
import requests
import shapely
from shapely import STRtree, make_valid
from shapely.geometry import shape

import build_city as b

ARC = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/{}/FeatureServer/{}/query?where=1%3D1&outFields={}&outSR=4326&f=geojson"
ITL3_URL = ARC.format("ITL3_JAN_2025_UK_BGC_V2", 1, "ITL325CD,ITL325NM")
LAD_URL = ARC.format("Local_Authority_Districts_December_2023_Boundaries_UK_BGC", 0, "LAD23CD,LAD23NM")
TS001_URL = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_2021_1.data.csv?date=latest&geography=2092957699TYPE424"
             "&c2021_restype_3=0&measures=20100&select=geography_code,geography_name,obs_value")


ARC_PAGED = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/{}/FeatureServer/0/query?where=1%3D1"
             "&geometry={},{},{},{}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects"
             "&outFields={}&outSR=4326&f=geojson&resultOffset={}&resultRecordCount=2000")
LSOA_SERVICE = "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5"
# BRES open access: employment (employees plus working owners), all industries, LSOA 2021
BRES_URL = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv?date=latest&geography=2092957699TYPE151"
            "&industry=37748736&employment_status=4&measure=1&measures=20100&select=date_name,geography_code,obs_value"
            "&recordoffset={}")
NOMIS_PAGE = 25000


def raw(name):
    return os.path.join(b.RAW, "tier0", name)


def _paged(make_url, path, rows_of, page):
    """Fetch every page of a paged query into one cached file (JSON list of rows)."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    out, off = [], 0
    while True:
        r = requests.get(make_url(off), headers=b.UA, timeout=600)
        r.raise_for_status()
        got = rows_of(r)
        out += got
        if len(got) < page:
            break
        off += len(got)
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(out, f)
    os.replace(path + ".part", path)
    return out


def lsoas(city, boundary):
    """[(LSOA 2021 code, polygon)] for the LSOAs whose representative point is in the boundary."""
    w, s, e, n = boundary.bounds
    feats = _paged(lambda off: ARC_PAGED.format(LSOA_SERVICE, w, s, e, n, "LSOA21CD", off),
                   raw(f"lsoa2021_bgc_{city}.json"), lambda r: r.json()["features"], 2000)
    out = []
    for ft in feats:
        g = make_valid(shape(ft["geometry"]))
        if boundary.contains(g.representative_point()):
            out.append((ft["properties"]["LSOA21CD"], g))
    return out


def ts001_lsoa():
    """{LSOA 2021 code: Census 2021 usual residents}. All of England."""
    url = TS001_URL.replace("TYPE424", "TYPE151") + "&recordoffset={}"
    rows = _paged(lambda off: url.format(off), raw("census2021_ts001_lsoa_england.json"),
                  lambda r: [(c, float(v)) for c, _, v in pd.read_csv(io.StringIO(r.text)).itertuples(index=False, name=None)],
                  NOMIS_PAGE)
    return dict(rows)


def lsoa_population(city, boundary, sig):
    """Census 2021 usual residents by LSOA, spread over the signal cells inside each LSOA by GHSL
    (T-360); tier0's `population` hook."""
    out = spread_totals(sig["lat"], sig["lon"], lsoas(city, boundary), ts001_lsoa(), sig["pop_ghsl"])
    print(f"  Census 2021 usual residents by LSOA: {out.sum():,.0f}", flush=True)
    return out, "Census 2021 usual residents by LSOA 2021 (TS001, Nomis NM_2021_1), spread inside each LSOA by GHSL GHS-POP E2020"


def bres_lsoa():
    """{LSOA 2021 code: BRES employment}, and the BRES year. All of England."""
    rows = _paged(lambda off: BRES_URL.format(off), raw("bres_lsoa2021_england.json"),
                  lambda r: [(str(d), c, float(v)) for d, c, v in pd.read_csv(io.StringIO(r.text)).itertuples(index=False, name=None)],
                  NOMIS_PAGE)
    return {c: float(v) for _, c, v in rows}, str(rows[0][0])


def _get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, headers=b.UA, timeout=600)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    return path


WU01EW_URL = "https://www.nomisweb.co.uk/output/census/2011/wu01ew_msoa.zip"
MSOA11_PWC = "MSOA_Dec_2011_PWC_in_England_and_Wales_2022"


def msoa11_points(city, boundary):
    """{MSOA 2011 code: (lat, lon)} of the population-weighted centroids inside the boundary."""
    w, s, e, n = boundary.bounds
    feats = _paged(lambda off: ARC_PAGED.format(MSOA11_PWC, w, s, e, n, "msoa11cd", off),
                   raw(f"msoa2011_pwc_{city}.json"), lambda r: r.json()["features"], 2000)
    out = {}
    for ft in feats:
        lon, lat = ft["geometry"]["coordinates"][:2]
        if boundary.contains(shapely.Point(lon, lat)):
            out[ft["properties"].get("msoa11cd") or ft["properties"].get("MSOA11CD")] = (lat, lon)
    return out


def wu01ew():
    """2011 census commuting flows, MSOA of residence to MSOA of workplace, all people 16+ in
    work (WU01EW): DataFrame h, w, n. Workplace codes outside England and Wales (at home, offshore,
    abroad) are kept as given; callers keep the pairs they can place."""
    import zipfile
    z = zipfile.ZipFile(_get(WU01EW_URL, raw("wu01ew_msoa.zip")))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    t = pd.read_csv(z.open(name), usecols=[0, 1, 2])
    t.columns = ["h", "w", "n"]
    return t


GIAS_DATE = "20261014"
GIAS_URL = "https://ea-edubase-api-prod.azurewebsites.net/edubase/downloads/public/edubasealldata{}.csv"


def gias_schools(boundary):
    """Open establishments in Get Information about Schools (DfE) with pupils on roll, inside the
    boundary: (DataFrame lat, lon, pupils; coverage text; url); tier0's `schools` hook (T-360).
    State and independent schools, nursery to sixth form; no pupil count means none placed."""
    from pyproj import Transformer
    url = GIAS_URL.format(GIAS_DATE)
    g = pd.read_csv(_get(url, raw(f"gias_edubasealldata{GIAS_DATE}.csv")), encoding="cp1252", low_memory=False,
                    usecols=["EstablishmentStatus (name)", "NumberOfPupils", "Easting", "Northing"])
    g = g[g["EstablishmentStatus (name)"].str.startswith("Open") & (g.Easting > 0) & (g.NumberOfPupils > 0)]
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(g.Easting.to_numpy(float), g.Northing.to_numpy(float))
    shapely.prepare(boundary)
    inn = shapely.contains_xy(boundary, lon, lat)
    out = pd.DataFrame({"lat": lat[inn], "lon": lon[inn], "pupils": g.NumberOfPupils.to_numpy(float)[inn]})
    return out, (f"Get Information about Schools ({GIAS_DATE}): {len(out):,} open establishments with pupils on roll, state and "
                 "independent, at their Easting/Northing"), url


def cell_units(lat, lon, units):
    """Index into units ([(code, polygon)]) of the unit holding each point, -1 for none; and for
    each unit holding no point, the point nearest its representative point (so no total is lost)."""
    pts = shapely.points(lon, lat)
    tree = STRtree([g for _, g in units])
    idx = np.full(len(pts), -1)
    pi, ui = tree.query(pts, predicate="within")
    idx[pi] = ui
    empty = np.setdiff1d(np.arange(len(units)), idx)
    near = STRtree(pts).nearest([units[u][1].representative_point() for u in empty]) if len(empty) else []
    return idx, dict(zip(empty.tolist(), np.asarray(near).tolist()))


def spread_totals(lat, lon, units, totals, weight):
    """Per point: each unit's total spread over the points inside it by weight (evenly where the
    weights are all zero); a unit with no point inside gives its total to the nearest one."""
    idx, near = cell_units(lat, lon, units)
    n = len(units)
    t = np.array([totals.get(c, 0.0) for c, _ in units])
    ok = idx >= 0
    w = np.asarray(weight, float)
    ws = np.bincount(idx[ok], weights=w[ok], minlength=n)
    cnt = np.bincount(idx[ok], minlength=n)
    out = np.zeros(len(lat))
    even = ws[idx[ok]] <= 0
    share = np.where(even, 1.0 / np.maximum(cnt[idx[ok]], 1), w[ok] / np.maximum(ws[idx[ok]], 1e-30))
    out[ok] = t[idx[ok]] * share
    for u, p in near.items():
        out[p] += t[u]
    return out


def bres_jobs(city, boundary, sig, weight):
    """Jobs per signal cell from BRES employment by LSOA, spread inside each LSOA by weight (T-360);
    tier0's `jobs` hook."""
    units = lsoas(city, boundary)
    jobs, year = bres_lsoa()
    out = spread_totals(sig["lat"], sig["lon"], units, jobs, weight)
    print(f"  BRES {year} employment: {len(units):,} LSOAs, {out.sum():,.0f} jobs", flush=True)
    return out, f"ONS Business Register and Employment Survey {year}, employment by LSOA 2021 (Nomis NM_189_1)"


def itl3_boundary(city, units):
    """Write the city's ITL3 units (codes) as one GeoJSON for tier0's `boundary_file`; its path."""
    out = raw(f"{city}_itl3.geojson")
    if os.path.exists(out):
        return out
    with open(_get(ITL3_URL, raw("itl3_jan2025_uk_bgc.geojson")), encoding="utf-8") as f:
        fc = json.load(f)
    feats = [ft for ft in fc["features"] if ft["properties"]["ITL325CD"] in units]
    missing = set(units) - {ft["properties"]["ITL325CD"] for ft in feats}
    if missing:
        raise ValueError(f"ITL3 units not found: {sorted(missing)}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return out


def lad_totals(boundary):
    """[(district name, polygon, Census 2021 usual residents)] for the districts in the boundary."""
    with open(_get(LAD_URL, raw("lad_dec2023_uk_bgc.geojson")), encoding="utf-8") as f:
        fc = json.load(f)
    pop = pd.read_csv(_get(TS001_URL, raw("census2021_ts001_lad_england.csv")))
    pop = dict(zip(pop.GEOGRAPHY_CODE, pop.OBS_VALUE))
    out = []
    for ft in fc["features"]:
        g = make_valid(shape(ft["geometry"]))
        if not boundary.contains(g.representative_point()):
            continue
        code, name = ft["properties"]["LAD23CD"], ft["properties"]["LAD23NM"]
        if code not in pop:
            raise ValueError(f"no Census 2021 total for {code} {name}")
        out.append((name, g, float(pop[code])))
    print(f"  Census 2021 district totals: {len(out)} districts, {sum(r[2] for r in out):,.0f} people", flush=True)
    return out
