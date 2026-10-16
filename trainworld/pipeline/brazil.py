"""T-458: Brazil's inputs for a tier 0 city (Sao Paulo first), through tier0's hooks.

- **Outline and units**: IBGE's 2022 census tracts (setores censitarios, the "malha com atributos"
  GeoPackage of the state), dissolved by municipality; the pack's municipalities by IBGE code.
- **Population**: Census 2022 people by tract (v0001), spread inside each tract by GHSL.
- **Jobs, work from home, curve** (a household travel survey, the general method any city with one
  can use): the Sao Paulo Metro's Pesquisa Origem-Destino 2017 microdata. Workers by survey zone of
  their first job (expanded by the person factor), scaled to the 2022 population, spread inside
  each zone by the jobs proxy; work from home is the share of workers whose job is at home; the
  curve is fitted to workers' home-to-job straight-line distances in bands (tier0.fit_curve_bands).

    python pipeline/brazil.py sao      (writes the units and survey caches and prints what they hold)
"""
import glob
import json
import os
import sys

import numpy as np
import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b
import england

# The 39 municipalities of the Regiao Metropolitana de Sao Paulo (IBGE codes), as the OD 2017 survey's
# Municipios_2017 layer lists them.
RMSP = [3503901, 3505708, 3506607, 3509007, 3509205, 3510609, 3513009, 3513801, 3515004, 3515103, 3515707,
        3516309, 3516408, 3518305, 3518800, 3522208, 3522505, 3523107, 3525003, 3526209, 3528502, 3529401,
        3530607, 3534401, 3539103, 3539806, 3543303, 3544103, 3545001, 3546801, 3547304, 3547809, 3548708,
        3548807, 3549953, 3550308, 3552502, 3552809, 3556453]
MUNIS = {"sao": RMSP}
SETORES_URL = ("https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/"
               "malha_com_atributos/setores/gpkg/UF/{uf}/{uf}_setores_CD2022.gpkg")
SURVEYS = {"sao": {"url": "https://transparencia.metrosp.com.br/sites/default/files/OD-2017.zip",
                   "name": "Metro de Sao Paulo, Pesquisa Origem-Destino 2017", "crs": "EPSG:22523"}}
# Home-to-job straight-line distance bands (km) for the curve fit; the survey's coordinates are exact
# to the address, so no road factor.
BAND_EDGES_KM = [0.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0, 1e9]


def raw(city, *p):
    return os.path.join(b.RAW, city, *p)


def setores_file(city, uf="SP"):
    """The pack's census tracts (code, municipality, people) as a GeoPackage; its path."""
    out = raw(city, f"{city}_setores.gpkg")
    if os.path.exists(out):
        return out
    import pyogrio
    src = raw(city, f"{uf}_setores_CD2022.gpkg")
    if not os.path.exists(src):
        b.fetch(SETORES_URL.format(uf=uf), src)
    codes = ",".join(f"'{c}'" for c in MUNIS[city])
    df = pyogrio.read_dataframe(src, columns=["CD_SETOR", "CD_MUN", "NM_MUN", "v0001"], where=f"CD_MUN IN ({codes})")
    df["v0001"] = df["v0001"].fillna(0).astype(float)
    df["geometry"] = df.geometry.make_valid()
    if df.CD_MUN.nunique() != len(MUNIS[city]):
        raise SystemExit(f"{city}: {df.CD_MUN.nunique()} municipalities of {len(MUNIS[city])} in the tracts")
    pyogrio.write_dataframe(df, out + ".part.gpkg")
    os.replace(out + ".part.gpkg", out)
    return out


def setores(city):
    import pyogrio
    return pyogrio.read_dataframe(setores_file(city))


def units_file(city):
    """The municipalities (tracts dissolved) with name, code and 2022 population; its path."""
    out = raw(city, f"{city}_units.geojson")
    if os.path.exists(out):
        return out
    df = setores(city)
    feats = []
    for code, g in df.groupby("CD_MUN"):
        geom = union_all(g.geometry.values).buffer(0)
        feats.append({"type": "Feature", "properties": {"code": code, "name": g.NM_MUN.iloc[0], "pop": int(g.v0001.sum())},
                      "geometry": mapping(geom)})
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    os.replace(out + ".part", out)
    return out


def units(city):
    with open(units_file(city), encoding="utf-8") as f:
        return [(ft["properties"], shape(ft["geometry"])) for ft in json.load(f)["features"]]


def boundary_file(city):
    """The municipalities dissolved, for tier0's `boundary_file`; its path."""
    out = raw(city, f"{city}_outline.geojson")
    if not os.path.exists(out):
        g = union_all([u[1] for u in units(city)]).buffer(0)
        with open(out + ".part", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"city": city},
                                                                   "geometry": mapping(g)}]}, f)
        os.replace(out + ".part", out)
    return out


def census_population(city, boundary, sig):
    """Census 2022 people by tract, spread inside each tract by GHSL; tier0's `population` hook."""
    df = setores(city)
    shapes = list(zip(df.CD_SETOR, df.geometry.values))
    out = england.spread_totals(sig["lat"], sig["lon"], shapes, dict(zip(df.CD_SETOR, df.v0001)), sig["pop_ghsl"])
    pts = shapely.points(sig["lon"], sig["lat"])
    for p, g in units(city):
        shapely.prepare(g)
        sel = shapely.contains(g, pts)
        have = sig["pop_ghsl"][sel].sum()
        print(f"  census {p['name']}: {p['pop']:,}; GHSL {have:,.0f} ({have / max(p['pop'], 1) - 1:+.1%})", flush=True)
    print(f"  census 2022 by tract: {len(df):,} tracts, {df.v0001.sum():,.0f} people; spread {out.sum():,.0f}; "
          f"GHSL {sig['pop_ghsl'].sum():,.0f}", flush=True)
    return out, "IBGE Census 2022 people by census tract (Agregados por setores censitarios), spread inside each by GHSL GHS-POP E2020"


# ---------------------------------------------------------------- the household travel survey

def _survey_dir(city):
    d = raw(city, "od")
    hit = glob.glob(os.path.join(d, "**", "OD_2017_v1.dbf"), recursive=True)
    if not hit:
        z = os.path.join(d, "OD-2017.zip")
        if not os.path.exists(z):
            b.fetch(SURVEYS[city]["url"], z)
        import zipfile
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d, [n for n in zf.namelist() if "Banco" in n or "Shape-OD2017" in n])
        hit = glob.glob(os.path.join(d, "**", "OD_2017_v1.dbf"), recursive=True)
    return os.path.dirname(os.path.dirname(hit[0]))


def _wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def survey(city):
    """The survey's derived tables (cached JSON): people, workers by zone of the first job, the work
    from home share, home-to-job distance bands, and the zones as lon/lat polygons."""
    out = raw(city, f"{city}_survey.json")
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    import pyogrio
    base = _survey_dir(city)
    df = pyogrio.read_dataframe(glob.glob(os.path.join(base, "Banco*", "OD_2017_v1.dbf"))[0], read_geometry=False,
                                columns=["F_PESS", "FE_PESS", "CD_ATIVI", "ZONATRA1", "TRAB1_RE", "CO_DOM_X", "CO_DOM_Y",
                                         "CO_TR1_X", "CO_TR1_Y"])
    p = df[df.F_PESS == 1]
    people = float(p.FE_PESS.sum())
    w = p[p.CD_ATIVI.isin([1, 2]) & p.ZONATRA1.between(1, 517)]
    jobs = w.groupby(w.ZONATRA1.astype(int)).FE_PESS.sum()
    home = float(w.FE_PESS[w.TRAB1_RE == 1].sum())
    c = w[(w.TRAB1_RE == 2) & w.CO_TR1_X.notna() & w.CO_DOM_X.notna()]
    d = np.hypot(c.CO_TR1_X - c.CO_DOM_X, c.CO_TR1_Y - c.CO_DOM_Y).to_numpy() / 1000.0
    band = np.searchsorted(BAND_EDGES_KM, d, side="right") - 1
    counts = np.bincount(band, weights=c.FE_PESS.to_numpy(), minlength=len(BAND_EDGES_KM) - 1)
    zf = pyogrio.read_dataframe(glob.glob(os.path.join(base, "Mapas*", "Shape*", "Zonas_2017_region.shp"))[0],
                                columns=["NumeroZona"]).to_crs("EPSG:4326")
    zones = {int(z): mapping(make_valid(g).buffer(0)) for z, g in zip(zf.NumeroZona, zf.geometry)}
    res = {"source": SURVEYS[city]["name"], "url": SURVEYS[city]["url"], "people": people,
           "workers_in_zones": float(w.FE_PESS.sum()), "jobs_by_zone": {str(k): float(v) for k, v in jobs.items()},
           "work_at_home": home / float(w.FE_PESS.sum()), "band_edges_km": BAND_EDGES_KM,
           "band_counts": counts.tolist(), "commute_median_km": _wmedian(d, c.FE_PESS.to_numpy()),
           "commute_mean_km": float(np.average(d, weights=c.FE_PESS.to_numpy())),
           "zones": zones}
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(out + ".part", out)
    return res


def survey_bands(city):
    """The bands with 0-2 and 2-4 km merged: between 2 km zones every neighbour pair sits at 2 or
    2.8 km, so the model cannot split them, and fitting the split drives the power to 2.3 (T-458)."""
    s = survey(city)
    e, c = s["band_edges_km"], s["band_counts"]
    return [e[0]] + e[2:], [c[0] + c[1]] + c[2:]


def survey_jobs(city, boundary, sig, weight):
    """Jobs per signal cell: the survey's workers by zone of their first job, scaled from the survey's
    population to the census's, spread inside each zone by weight; tier0's `jobs` hook."""
    s = survey(city)
    pop2022 = sum(p["pop"] for p, _ in units(city))
    k = pop2022 / s["people"]
    shapes = [(z, shape(g)) for z, g in s["zones"].items()]
    totals = {z: v * k for z, v in s["jobs_by_zone"].items()}
    out = england.spread_totals(sig["lat"], sig["lon"], shapes, totals, weight)
    print(f"  survey jobs: {len(totals)} zones, {sum(s['jobs_by_zone'].values()):,.0f} workers in 2017 x {k:.4f} "
          f"(census 2022 {pop2022:,} / survey {s['people']:,.0f} people) = {out.sum():,.0f}", flush=True)
    return out, f"{s['source']}, workers by zone of their first job (517 zones), scaled to the 2022 census population"


if __name__ == "__main__":
    city = sys.argv[1]
    us = units(city)
    g = union_all([u[1] for u in us])
    km2 = 111.32 * 110.57 * 0.917  # degrees squared to km2 near 23.5 S
    print(f"{city}: {len(us)} municipalities, census 2022 {sum(p['pop'] for p, _ in us):,}, outline {g.geom_type}, "
          f"{len(getattr(g, 'geoms', [g]))} parts, {sum(len(q.interiors) for q in getattr(g, 'geoms', [g]))} holes, "
          f"about {g.area * km2:,.0f} km2, bounds {tuple(round(v, 3) for v in g.bounds)}")
    s = survey(city)
    tot = sum(s["band_counts"])
    print(f"survey: {s['people']:,.0f} people, {s['workers_in_zones']:,.0f} workers with a job zone, "
          f"work at home {s['work_at_home']:.1%}; home-to-job bands {s['band_edges_km'][:-1]}: "
          + " ".join(f"{c / tot:.3f}" for c in s["band_counts"]))
