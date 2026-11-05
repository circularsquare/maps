"""T-2080, T-2081: Mexico's inputs for any planned Mexican row, through tier0's hooks.

- **Population**: INEGI Censo de Poblacion y Vivienda 2020, people (POBTOT) in three kinds of unit, each spread
  inside it by GHSL: urban AGEBs (the Marco Geoestadistico 2020's polygons, people from "Principales resultados por
  AGEB y manzana urbana"); rural localities (ITER) as points, each holding the cells nearer it than any other locality
  and within RURAL_KM; and the rest of each municipio (ITER's 9998 and 9999 roll-ups of localities of one or two
  dwellings, localities with no point), over the municipio's cells the first two left. A unit is the row's when its
  representative point (a locality's point) is inside the outline; a municipio's rest goes by its GHSL share inside.
- **Jobs**: DENUE, every economic unit with its point and an employee band (per_ocu), at the band's midpoint
  (DENUE_BAND) in the cell holding it. DENUE leaves out farm work, street trade with no fixed site and paid domestic
  work, so the gap between tier 0's World Bank jobs a resident and DENUE's national total (DENUE_GAP) is added over the
  row's population by the jobs proxy.

    python pipeline/mexico.py fetch                 download every source into data/raw/mexico
    python pipeline/mexico.py units                 build the national unit and DENUE caches, print their totals
    python pipeline/mexico.py check [ids|states]    the hooks on stand-in outlines; writes nothing outside data/raw
"""
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import shapely
from shapely import make_valid
from shapely.geometry import shape

import build_city as b

RAW_MX = os.path.join(b.RAW, "mexico")
INEGI = "https://www.inegi.org.mx/contenidos"
ITER_URL = f"{INEGI}/programas/ccpv/2020/datosabiertos/iter/iter_00_cpv2020_csv.zip"
AGEB_URL = f"{INEGI}/programas/ccpv/2020/datosabiertos/ageb_manzana/ageb_mza_urbana_{{ent}}_cpv2020_csv.zip"
MG_URL = (f"{INEGI}/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/"
          "mg_2020_integrado.zip")
DENUE_URL = f"{INEGI}/masiva/denue/denue_{{ent}}_csv.zip"  # no national file: denue_00_csv.zip is a soft 404
DENUE_PAGE = "https://www.inegi.org.mx/app/descarga/?ti=6"
ENTS = [f"{i:02d}" for i in range(1, 33)]
DENUE_PARTS = [p for e in ENTS for p in ((f"{e}_1", f"{e}_2") if e == "15" else (e,))]  # the State of Mexico in two
# INEGI answers a wrong path with HTTP 200 and a page of about 2 KB; every real file here is over 1 MB
MIN_BYTES = 1 << 20
NATIONAL = 126_014_024  # ITER 2020, Total nacional, POBTOT
RURAL_KM = 2.0
# DENUE's per_ocu bands, workers counted at each: Censos Economicos 2019's mean sizes for the bottom and top
# (units of 0-10 people 2.2, of 251 and more about 900), about the middle of the others (notes/T-2080.md)
DENUE_BAND = {0: 2.0, 6: 8.0, 11: 18.0, 31: 40.0, 51: 72.0, 101: 160.0, 251: 900.0}  # by the band's first figure
POP_SOURCE = ("INEGI Censo de Poblacion y Vivienda 2020: urban AGEBs (Principales resultados por AGEB y manzana urbana) "
              "and rural localities (ITER) on the Marco Geoestadistico 2020, the rest of each municipio from ITER, "
              "spread inside each by GHSL GHS-POP E2020")


def raw(*p):
    return os.path.join(RAW_MX, *p)


def fetch(url, name=None):
    """url into data/raw/mexico once; a download under MIN_BYTES or not a zip is INEGI's soft 404 and raises."""
    name = name or url.rsplit("/", 1)[1]
    path = raw(name)
    if os.path.exists(path):
        return path
    os.makedirs(RAW_MX, exist_ok=True)
    b.fetch(url, os.path.join("mexico", name))
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4)
    if size < MIN_BYTES or magic != b"PK\x03\x04":
        os.remove(path)
        raise SystemExit(f"{url}: {size:,} bytes, not a zip (INEGI's answer to a wrong path); find the file on its page")
    return path


def fetch_all(threads=6):
    """Every source; several at once, since INEGI serves each connection at about 200 KB/s."""
    from concurrent.futures import ThreadPoolExecutor

    def one(u):
        p = fetch(u)
        print(f"  {os.path.basename(p)}: {os.path.getsize(p):,} bytes", flush=True)
    urls = [MG_URL, ITER_URL] + [DENUE_URL.format(ent=e) for e in DENUE_PARTS] + [AGEB_URL.format(ent=e) for e in ENTS]
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, urls))


BOM = "ï»¿"  # UTF-8's byte order mark read as Latin-1


def _csv(zpath, part, cols):
    """Columns `cols` (as strings) of the data CSV(s) of an INEGI zip (members under conjunto_de_datos with `part`
    in the name), concatenated."""
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if "conjunto_de_datos" in n and n.lower().endswith(".csv") and part in n]
        if not names:
            raise SystemExit(f"{zpath}: no data CSV with {part!r}")
        # a state's file is UTF-8 with a BOM or Latin-1; only code and number columns are read, so Latin-1 reads both
        frames = [pd.read_csv(z.open(n), dtype=str, encoding="latin-1", usecols=lambda c: c.replace(BOM, "") in cols)
                  for n in names]
    for f in frames:
        f.columns = [c.replace(BOM, "") for c in f.columns]
    return pd.concat(frames, ignore_index=True)[cols]


def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _dms(s):
    """ITER's LONGITUD / LATITUD (102°17'45.739\" W) in decimal degrees, west and south negative."""
    p = s.str.extract(r"(\d+)\D+(\d+)\D+([\d.]+)\D*([NSEW])")
    v = p[0].astype(float) + p[1].astype(float) / 60 + p[2].astype(float) / 3600
    return np.where(p[3].isin(["W", "S"]), -v, v)


# ---------------------------------------------------------------- national caches

def iter_table():
    """ITER 2020's municipio and locality rows: DataFrame ent, mun, loc, name, pop, lat, lon (CVEGEO parts as
    zero-padded strings)."""
    t = _csv(fetch(ITER_URL), "iter", ["ENTIDAD", "MUN", "LOC", "LONGITUD", "LATITUD", "POBTOT"])
    t.columns = ["ent", "mun", "loc", "lon", "lat", "pop"]
    t = t[(t.ent != "00") & (t.mun != "000")].copy()
    t["pop"] = _num(t["pop"])
    t["lon"], t["lat"] = _dms(t.lon.fillna("")), _dms(t.lat.fillna(""))
    return t


def ageb_table():
    """Urban AGEB totals from the 32 states' AGEB files: DataFrame cvegeo (13 characters), pop (POBTOT)."""
    rows = []
    for e in ENTS:
        t = _csv(fetch(AGEB_URL.format(ent=e)), "ageb", ["ENTIDAD", "MUN", "LOC", "AGEB", "MZA", "POBTOT"])
        t = t[(t.MUN != "000") & (t.LOC != "0000") & (t.AGEB != "0000") & (t.MZA == "000")]
        rows.append(pd.DataFrame({"cvegeo": t.ENTIDAD + t.MUN + t.LOC + t.AGEB, "pop": _num(t.POBTOT)}))
    return pd.concat(rows, ignore_index=True)


def _mg(layer, columns):
    import pyogrio
    df = pyogrio.read_dataframe(f"/vsizip/{fetch(MG_URL)}/conjunto_de_datos/{layer}.shp", columns=columns)
    df = df.to_crs("EPSG:4326")
    df["geometry"] = df.geometry.make_valid()
    return df


def units():
    """The national unit caches in data/raw/mexico, built once:
    - mx_urban_ageb.gpkg: cvegeo (13), mun (5), pop, geometry, the urban AGEBs of the Marco Geoestadistico;
    - mx_mun.gpkg: cvegeo (5), name, pop (ITER), urban (its urban AGEBs' people), rural (its pointed localities'),
      rest (pop less both), geometry;
    - mx_rural_loc.csv: cvegeo (9), mun, pop, lat, lon, ITER's localities outside the AGEB files, with a point."""
    out = {k: raw(f"mx_{k}") for k in ("urban_ageb.gpkg", "mun.gpkg", "rural_loc.csv")}
    if all(os.path.exists(p) for p in out.values()):
        return out
    import pyogrio
    it = iter_table()
    mun = it[it["loc"] == "0000"]
    mun = mun.set_index(mun.ent + mun.mun)
    loc = it[~it["loc"].isin(["0000", "9998", "9999"])].copy()
    loc["cvegeo"] = loc.ent + loc.mun + loc["loc"]
    ag = ageb_table()
    a = _mg("00a", ["CVEGEO"])
    a = a[a.CVEGEO.str.len() == 13]
    pop = dict(zip(ag.cvegeo, ag["pop"]))
    miss_geo = sorted(set(pop) - set(a.CVEGEO))
    a["pop"] = a.CVEGEO.map(pop)
    print(f"  urban AGEB: {len(a):,} polygons, {len(ag):,} AGEB rows, {int(a['pop'].isna().sum()):,} polygons with no row "
          f"(0 people), {len(miss_geo):,} rows with no polygon ({ag.set_index('cvegeo').loc[miss_geo, 'pop'].sum():,.0f} "
          "people: their localities have no urban AGEB polygon, so count as rural localities at ITER's figure)", flush=True)
    a["pop"] = a["pop"].fillna(0.0)
    a["mun"] = a.CVEGEO.str[:5]
    a = a.rename(columns={"CVEGEO": "cvegeo"})
    urban_locs = set(a.cvegeo.str[:9])
    rural = loc[~loc.cvegeo.isin(urban_locs) & loc.lat.notna()]
    rural = pd.DataFrame({"cvegeo": rural.cvegeo, "mun": rural.cvegeo.str[:5], "pop": rural["pop"],
                          "lat": rural.lat, "lon": rural.lon})
    m = _mg("00mun", ["CVEGEO", "NOMGEO"]).rename(columns={"CVEGEO": "cvegeo", "NOMGEO": "name"})
    if set(m.cvegeo) != set(mun.index):
        raise SystemExit(f"municipios: {len(set(m.cvegeo) ^ set(mun.index))} differ between 00mun and ITER")
    m["pop"] = m.cvegeo.map(mun["pop"])
    m["urban"] = m.cvegeo.map(a.groupby("mun")["pop"].sum()).fillna(0.0)
    m["rural"] = m.cvegeo.map(rural.groupby("mun")["pop"].sum()).fillna(0.0)
    m["rest"] = m["pop"] - m["urban"] - m["rural"]
    neg = m[m["rest"] < 0]
    print(f"  municipios: {len(m):,}, {m['pop'].sum():,.0f} people (ITER national {NATIONAL:,}); urban AGEB "
          f"{m['urban'].sum():,.0f}, rural localities {m['rural'].sum():,.0f} ({len(rural):,}), rest {m['rest'].sum():,.0f}; "
          f"{len(neg)} municipios with units over ITER's total ({neg['rest'].sum():,.0f}, the units scaled down there)", flush=True)
    os.makedirs(RAW_MX, exist_ok=True)
    pyogrio.write_dataframe(a[["cvegeo", "mun", "pop", "geometry"]], out["urban_ageb.gpkg"] + ".part.gpkg")
    os.replace(out["urban_ageb.gpkg"] + ".part.gpkg", out["urban_ageb.gpkg"])
    pyogrio.write_dataframe(m, out["mun.gpkg"] + ".part.gpkg")
    os.replace(out["mun.gpkg"] + ".part.gpkg", out["mun.gpkg"])
    rural.to_csv(out["rural_loc.csv"] + ".part", index=False)
    os.replace(out["rural_loc.csv"] + ".part", out["rural_loc.csv"])
    return out


def _read_units(bbox):
    import pyogrio
    p = units()
    ag = pyogrio.read_dataframe(p["urban_ageb.gpkg"], bbox=bbox)
    mun = pyogrio.read_dataframe(p["mun.gpkg"], bbox=bbox)
    loc = pd.read_csv(p["rural_loc.csv"], dtype={"cvegeo": str, "mun": str})
    w, s, e, n = bbox
    pad = RURAL_KM / 100.0
    loc = loc[loc.lon.between(w - pad, e + pad) & loc.lat.between(s - pad, n + pad)].reset_index(drop=True)
    # where a municipio's units pass ITER's total, they are scaled to it
    k = (mun["pop"] / (mun["urban"] + mun["rural"]).where(mun["rest"] < 0)).fillna(1.0)
    k = dict(zip(mun.cvegeo, k))
    ag["pop"] = ag["pop"] * ag.mun.map(k).fillna(1.0)
    loc["pop"] = loc["pop"] * loc.mun.map(k).fillna(1.0)
    mun["rest"] = mun["rest"].clip(lower=0.0)
    return ag, mun, loc


# ---------------------------------------------------------------- population

def _xy(lat, lon, lat0):
    return np.column_stack([np.radians(lon) * np.cos(np.radians(lat0)), np.radians(lat)]) * (b.R_EARTH / 1000.0)


def _spread(idx, n_units, totals, weight, out, near=None):
    """Each unit's total over the points idx assigns it (-1 none), by weight, evenly where its weights are all zero;
    near: {unit: point} for units with no point."""
    ok = idx >= 0
    ws = np.bincount(idx[ok], weights=weight[ok], minlength=n_units)
    cnt = np.bincount(idx[ok], minlength=n_units)
    share = np.where(ws[idx[ok]] > 0, weight[ok] / np.maximum(ws[idx[ok]], 1e-30), 1.0 / np.maximum(cnt[idx[ok]], 1))
    np.add.at(out, np.flatnonzero(ok), totals[idx[ok]] * share)
    for u, p in (near or {}).items():
        out[p] += totals[u]


def plan_population(city, boundary, sig):
    """tier0's `population` hook for any planned Mexican row (the module's docstring); cells in no municipio keep
    GHSL's people."""
    import england
    from scipy.spatial import cKDTree
    import tier0
    lat, lon = np.asarray(sig["lat"], float), np.asarray(sig["lon"], float)
    ghsl = np.asarray(sig["pop_ghsl"], float)
    n = len(lat)
    out = np.zeros(n)
    shapely.prepare(boundary)
    ag, mun, loc = _read_units(boundary.bounds)
    # urban AGEBs, by representative point
    ag = ag[shapely.contains(boundary, ag.geometry.representative_point().to_numpy())].reset_index(drop=True)
    au = list(zip(ag.cvegeo, ag.geometry.values))
    idx, near = england.cell_units(lat, lon, au) if len(au) else (np.full(n, -1), {})
    _spread(idx, len(au), ag["pop"].to_numpy(float), ghsl, out, near)
    free = idx < 0
    # rural localities, each over the free cells nearer it than any other locality, within RURAL_KM
    lat0 = float(np.mean(boundary.bounds[1::2]))
    inside = shapely.contains_xy(boundary, loc.lon.to_numpy(), loc.lat.to_numpy())
    lidx = np.full(n, -1)
    if len(loc):
        d, j = cKDTree(_xy(loc.lat.to_numpy(), loc.lon.to_numpy(), lat0)).query(
            _xy(lat[free], lon[free], lat0), distance_upper_bound=RURAL_KM)
        got = np.isfinite(d)
        j = np.where(got, j, -1)
        j[got] = np.where(inside[j[got]], j[got], -1)
        lidx[np.flatnonzero(free)] = j
    have = np.bincount(lidx[lidx >= 0], minlength=len(loc)) if len(loc) else np.zeros(0, int)
    lone = np.flatnonzero(inside & (have == 0))
    lnear = {}
    if len(lone):
        pool = np.flatnonzero(free) if free.any() else np.arange(n)
        _, k = cKDTree(_xy(lat[pool], lon[pool], lat0)).query(_xy(loc.lat.to_numpy()[lone], loc.lon.to_numpy()[lone], lat0))
        lnear = dict(zip(lone.tolist(), pool[k].tolist()))
    _spread(lidx, len(loc), loc["pop"].to_numpy(float), ghsl, out, lnear)
    free &= lidx < 0
    # each municipio's rest, by its GHSL share inside the outline, over its free cells
    mu = list(zip(mun.cvegeo, mun.geometry.values))
    midx, mnear = england.cell_units(lat, lon, mu)
    within = shapely.within(mun.geometry.values, boundary)
    cut = np.flatnonzero(~within & shapely.intersects(mun.geometry.values, boundary))
    share = within.astype(float)
    cut = cut[mun["rest"].to_numpy()[cut] > 0]  # with the 2020 files every municipio's rest is 0
    if len(cut):
        whole, part = tier0.ghsl_people(mun.geometry.values[cut], inside=boundary)
        share[cut] = np.where(whole > 0, part / np.maximum(whole, 1e-9), 0.0)
    rest = mun["rest"].to_numpy(float) * share
    ridx = np.where(free, midx, -1)
    has_free = np.bincount(ridx[ridx >= 0], minlength=len(mu)) > 0
    ridx = np.where((midx >= 0) & ~has_free[np.maximum(midx, 0)], midx, ridx)  # no free cell: over all its cells
    _spread(ridx, len(mu), rest, ghsl, out, mnear)
    nomun = (midx < 0) & free
    out[nomun] += ghsl[nomun]
    print(f"  census 2020: {len(au):,} urban AGEBs {ag['pop'].sum():,.0f}, {int(inside.sum()):,} rural localities "
          f"{loc['pop'].to_numpy()[inside].sum():,.0f}, municipios' rest {rest.sum():,.0f} ({len(mu)} municipios, {len(cut)} cut); "
          f"{int(nomun.sum()):,} cells in no municipio keep GHSL's {ghsl[nomun].sum():,.0f}; {out.sum():,.0f} people "
          f"(GHSL {ghsl.sum():,.0f}, {ghsl.sum() / max(out.sum(), 1) - 1:+.1%})", flush=True)
    return out, POP_SOURCE


def cell_points(boundary):
    """tier0's `cell_points`: the rural localities' points, so each has a cell where GHSL has none."""
    _, _, loc = _read_units(boundary.bounds)
    shapely.prepare(boundary)
    sel = shapely.contains_xy(boundary, loc.lon.to_numpy(), loc.lat.to_numpy())
    return loc.lat.to_numpy()[sel], loc.lon.to_numpy()[sel]


# ---------------------------------------------------------------- jobs

def denue_cells():
    """DENUE's workers by res-9 cell, cached in data/raw/mexico/mx_denue_cells.npz: h3 (u8), jobs, and ent (the
    state of the cell's first unit); with the units counted and DENUE's edition."""
    path = raw("mx_denue_cells.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    import h3
    hs, js, es, n, bands, editions = [], [], [], 0, {}, set()
    for e in DENUE_PARTS:
        t = _csv(fetch(DENUE_URL.format(ent=e)), "denue", ["per_ocu", "latitud", "longitud"])
        n += len(t)
        for k, v in t.per_ocu.value_counts().items():
            bands[k] = bands.get(k, 0) + int(v)
        w = pd.to_numeric(t.per_ocu.str.extract(r"^(\d+)")[0], errors="coerce").map(DENUE_BAND)
        la, lo = pd.to_numeric(t.latitud, errors="coerce"), pd.to_numeric(t.longitud, errors="coerce")
        ok = w.notna() & la.notna() & lo.notna()
        hs.append(np.fromiter((h3.str_to_int(h3.latlng_to_cell(a, o, 9)) for a, o in zip(la[ok], lo[ok])), np.uint64, int(ok.sum())))
        js.append(w[ok].to_numpy(float))
        es.append(np.full(int(ok.sum()), int(e[:2]), np.uint8))
        with zipfile.ZipFile(raw(f"denue_{e}_csv.zip")) as z:
            meta = z.read(next(m for m in z.namelist() if m.startswith("metadatos"))).decode("utf-8", "replace")
        ed = re.search(r"\b(\d{2})_(20\d{2})\b", meta)
        if ed:
            editions.add(f"{ed.group(2)}-{ed.group(1)}")
        print(f"  DENUE {e}: {len(t):,} units, {w[ok].sum():,.0f} workers, {int((~ok).sum()):,} with no band or point", flush=True)
    unknown = {k: v for k, v in bands.items() if int(re.match(r"\d*", k).group() or -1) not in DENUE_BAND}
    if unknown:
        raise SystemExit(f"DENUE per_ocu bands not in DENUE_BAND: {unknown}")
    h, j, ent = np.concatenate(hs), np.concatenate(js), np.concatenate(es)
    u, first, inv = np.unique(h, return_index=True, return_inverse=True)
    out = {"h3": u, "jobs": np.bincount(inv, weights=j), "ent": ent[first], "units": np.array(n),
           "edition": np.array(max(editions, default="edition not given"))}
    np.savez_compressed(path + ".part.npz", **out)
    os.replace(path + ".part.npz", path)
    print(f"  DENUE: {n:,} units in {len(u):,} cells, {j.sum():,.0f} workers; bands {bands}", flush=True)
    return out


def denue_gap():
    """Jobs a resident that DENUE leaves out: tier 0's World Bank jobs a resident less DENUE's national workers over
    the census's people (never below 0)."""
    import tier0
    jpr, _, year = tier0.worldbank("MX")
    d = float(denue_cells()["jobs"].sum())
    return max(jpr - d / NATIONAL, 0.0), jpr, year, d


def plan_jobs(city, boundary, sig, weight):
    """tier0's `jobs` hook for any planned Mexican row: DENUE's workers in the cell holding each unit (the nearest
    cell where its own is not one of the row's), plus the gap (denue_gap) x the row's people, spread by weight."""
    import h3
    from scipy.spatial import cKDTree
    d = denue_cells()
    lat, lon = np.asarray(sig["lat"], float), np.asarray(sig["lon"], float)
    cells = np.asarray(sig["h3"], np.uint64)
    out = np.zeros(len(cells))
    order = np.argsort(cells)
    pos = np.searchsorted(cells[order], d["h3"])
    hit = (pos < len(cells)) & (cells[order][np.minimum(pos, len(cells) - 1)] == d["h3"])
    np.add.at(out, order[pos[hit]], d["jobs"][hit])
    rest = np.flatnonzero(~hit)
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in d["h3"][rest]]).reshape(-1, 2)
    w, s, e, n = boundary.bounds
    box = (ll[:, 0] >= s) & (ll[:, 0] <= n) & (ll[:, 1] >= w) & (ll[:, 1] <= e)
    rest, ll = rest[box], ll[box]
    shapely.prepare(boundary)
    ins = shapely.contains_xy(boundary, ll[:, 1], ll[:, 0])
    rest, ll = rest[ins], ll[ins]
    if len(rest):
        lat0 = float(np.mean(boundary.bounds[1::2]))
        _, k = cKDTree(_xy(lat, lon, lat0)).query(_xy(ll[:, 0], ll[:, 1], lat0))
        np.add.at(out, k, d["jobs"][rest])
    denue = out.sum()
    gap, jpr, year, nat = denue_gap()
    P = float(np.sum(sig["pop"]))
    wt = np.asarray(weight, float)
    out += gap * P * wt / max(wt.sum(), 1e-30)
    print(f"  DENUE {d['edition']}: {denue:,.0f} workers in the outline ({int(hit.sum()):,} cells; {len(rest):,} more to their "
          f"nearest), plus {gap:.3f} a resident x {P:,.0f} people by the proxy (World Bank {year} {jpr:.3f} a resident less "
          f"DENUE's {nat / NATIONAL:.3f}): {out.sum():,.0f} jobs, {out.sum() / max(P, 1):.3f} a resident", flush=True)
    return out, (f"INEGI DENUE ({d['edition']}), every economic unit at its point, workers at the middle of its per_ocu band, "
                 f"plus {gap:.3f} jobs a resident it leaves out (farm work, street trade, domestic work: World Bank {year} "
                 f"jobs a resident less DENUE's national total)")


def plan_hooks(row):
    """T-2080, T-2081: the country keys for any planned Mexican row (unit set MEX)."""
    return {"population": plan_population, "cell_points": cell_points, "jobs": plan_jobs,
            "jobs_url": DENUE_PAGE}


# ---------------------------------------------------------------- check

def _row_outline(row):
    """A stand-in for the row's outline: its geoBoundaries ADM2 units, dissolved (the world plan's file)."""
    import world_plan as wp
    with open(wp.RAW / "geoboundaries" / "MEX_ADM2.geojson", encoding="utf-8") as f:
        feats = json.load(f)["features"]
    want = set(row["units"]["MEX"])
    gs = [make_valid(shape(f["geometry"])) for f in feats if f["properties"]["shapeID"] in want]
    if len(gs) != len(want):
        raise SystemExit(f"{row['id']}: {len(want) - len(gs)} units not in geoBoundaries MEX ADM2")
    return shapely.union_all(gs).buffer(0)


def stand_in_sig(boundary, origin):
    """The cells and GHSL people tier0.signals would give (no land mask, no bounding, the rural localities' cells
    added); in memory only."""
    import h3
    import tier0
    pop, X0, Y0 = tier0.ghsl_window("pop", boundary.bounds, dtype=np.float32)
    g = tier0.grid_to_cells(pop, X0, Y0, origin, None, near=tier0.near_outline(boundary))
    la, lo = cell_points(boundary)
    extra = np.fromiter((h3.str_to_int(h3.latlng_to_cell(a, o, 9)) for a, o in zip(la, lo)), np.uint64, len(la))
    cells = np.unique(np.concatenate([np.fromiter(g.keys(), np.uint64), extra]))
    ok, ll = tier0.inside(cells, boundary)
    cells, ll = cells[ok], ll[ok]
    p = np.array([g.get(int(c), 0.0) for c in cells])
    return {"h3": cells, "lat": ll[:, 0], "lon": ll[:, 1], "pop_ghsl": p, "pop": p.copy()}


def check(args):
    """The hooks on stand-in outlines, people against INEGI and GHSL, jobs against DENUE: the plan's Mexican rows
    (ids, default all) or states ("states", or e.g. "states:09,20")."""
    import plan_config
    import pyogrio
    mun = pyogrio.read_dataframe(units()["mun.gpkg"])
    if args and args[0].startswith("states"):
        ents = args[0].split(":")[1].split(",") if ":" in args[0] else ENTS
        targets = []
        for e in ents:
            m = mun[mun.cvegeo.str[:2] == e]
            g = shapely.union_all(m.geometry.values).buffer(0)
            targets.append((f"ent {e}", g, float(m["pop"].sum()), tuple(g.representative_point().coords[0])))
    else:
        rows, _ = plan_config.load()
        rows = [r for r in rows if "MEX" in (r.get("units") or {}) and (not args or r["id"] in args)]
        targets = []
        for r in rows:
            g = _row_outline(r)
            shapely.prepare(g)
            sel = shapely.contains(g, mun.geometry.representative_point().to_numpy())
            targets.append((f"{r['id']} {r['name']}", g, float(mun["pop"][sel].sum()), tuple(r["origin"])))
    lines = []
    for name, g, census, origin in targets:
        print(f"{name}:", flush=True)
        sig = stand_in_sig(g, origin)
        pop, _ = plan_population(name, g, sig)
        sig["pop"] = pop
        jobs, _ = plan_jobs(name, g, sig, pop)  # the stand-in has no OSM or NRES signals: the gap goes by people
        lines.append((name, pop.sum(), census, sig["pop_ghsl"].sum(), jobs.sum()))
    print(f"{'outline':<28} {'hook':>12} {'INEGI':>12} {'hook/INEGI':>10} {'GHSL':>12} {'GHSL/hook':>10} {'jobs':>11} {'a res.':>6}")
    for name, p, c, gh, j in lines:
        print(f"{name:<28} {p:>12,.0f} {c:>12,.0f} {p / max(c, 1) - 1:>+10.2%} {gh:>12,.0f} {gh / max(p, 1) - 1:>+10.1%} "
              f"{j:>11,.0f} {j / max(p, 1):>6.3f}")
    P, C, G = (sum(x[i] for x in lines) for i in (1, 2, 3))
    print(f"{'all':<28} {P:>12,.0f} {C:>12,.0f} {P / max(C, 1) - 1:>+10.2%} {G:>12,.0f} {G / max(P, 1) - 1:>+10.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch_all()
    elif cmd == "units":
        units()
        d = denue_cells()
        print(f"DENUE {d['edition']}: {int(d['units']):,} units, {d['jobs'].sum():,.0f} workers in {len(d['h3']):,} cells; "
              "gap {:.3f} a resident (World Bank {:.3f}, {})".format(*denue_gap()[:3]))
    elif cmd == "check":
        check(sys.argv[2:])
    else:
        print(__doc__)
