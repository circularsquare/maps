"""T-289: France's inputs for a tier 0 city (Paris first), through tier0's hooks.

- **Outline**: whole départements, dissolved from their communes.
- **Communes**: geo.api.gouv.fr (Etalab, from IGN Admin Express and INSEE's COG), contours and
  populations légales; Paris, Lyon and Marseille as their arrondissements municipaux, the unit
  INSEE's flows use. Population is the latest legal municipal population the API serves.
- **Jobs**: INSEE Mobilités professionnelles 2022 (census, base flux domicile-lieu de travail,
  every flow), employed people 15+ summed by commune of work: jobs held by residents of France,
  whatever their place of residence. Teleworkers count at their usual workplace.
- **Curve**: the same flows, commune to commune (curves/t289_curve.py).

All open (Licence Ouverte / Etalab). Cached in data/raw/tier0/france/.
"""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import requests
import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b
import england

GEO_API = ("https://geo.api.gouv.fr/communes?codeDepartement={dep}&fields=code,nom,population,codeDepartement"
           "&format=geojson&geometry=contour")
GEO_API_ARM = GEO_API.replace("communes?", "communes?type=arrondissement-municipal&")
PLM = {"75056", "69123", "13055"}  # Paris, Lyon, Marseille: INSEE's flows use their arrondissements
FLUX_URL = "https://www.insee.fr/fr/statistiques/fichier/8582949/base-flux-mobilite-domicile-lieu-travail-2022_csv.zip"


def raw(name):
    return os.path.join(b.RAW, "tier0", "france", name)


def _get_json(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, headers=b.UA, timeout=600)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def communes(deps):
    """[(code, name, département, polygon, population)] for the départements, Paris, Lyon and
    Marseille as arrondissements municipaux."""
    out = []
    for dep in deps:
        feats = _get_json(GEO_API.format(dep=dep), raw(f"communes_{dep}.geojson"))["features"]
        if any(ft["properties"]["code"] in PLM for ft in feats):
            feats = [ft for ft in feats if ft["properties"]["code"] not in PLM]
            feats += _get_json(GEO_API_ARM.format(dep=dep), raw(f"arm_{dep}.geojson"))["features"]
        for ft in feats:
            p = ft["properties"]
            out.append((p["code"], p["nom"], dep, make_valid(shape(ft["geometry"])), float(p.get("population") or 0)))
    return out


def departements_boundary(city, deps):
    """Write the départements, each dissolved from its communes, as one GeoJSON for tier0's
    `boundary_file`; its path."""
    out = raw(f"{city}_departements.geojson")
    if os.path.exists(out):
        return out
    cs = communes(deps)
    feats = []
    for dep in deps:
        g = union_all([c[3] for c in cs if c[2] == dep]).buffer(0)
        feats.append({"type": "Feature", "properties": {"code": dep}, "geometry": mapping(g)})
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    os.replace(out + ".part", out)
    return out


def commune_population(city, boundary, sig, deps):
    """Legal population by commune (arrondissement in Paris), spread over the signal cells inside
    each commune by GHSL; tier0's `population` hook."""
    cs = communes(deps)
    out = england.spread_totals(sig["lat"], sig["lon"], [(c[0], c[3]) for c in cs], {c[0]: c[4] for c in cs}, sig["pop_ghsl"])
    print(f"  populations légales by commune: {len(cs):,} communes, {out.sum():,.0f} people", flush=True)
    return out, ("INSEE populations légales (population municipale) by commune, Paris by arrondissement, via geo.api.gouv.fr, "
                 "spread inside each commune by GHSL GHS-POP E2020")


def flux():
    """INSEE 2022 home to work flows, commune (arrondissement) of residence to commune of work:
    DataFrame h, w, n (employed people 15+)."""
    path = raw("base-flux-mobilite-domicile-lieu-travail-2022_csv.zip")
    if not os.path.exists(path):
        england._get(FLUX_URL, path)
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    t = pd.read_csv(z.open(name), sep=";", dtype={"CODGEO": str, "DCLT": str},
                    usecols=["CODGEO", "DCLT", "NBFLUX_C22_ACTOCC15P"])
    t.columns = ["h", "w", "n"]
    return t


def workplace_jobs():
    """{commune of work: employed people 15+ working there} from the 2022 flows, cached small."""
    path = raw("insee_2022_jobs_by_commune_of_work.csv")
    if not os.path.exists(path):
        t = flux().groupby("w", as_index=False).n.sum()
        t.to_csv(path + ".part", index=False)
        os.replace(path + ".part", path)
    t = pd.read_csv(path, dtype={"w": str})
    return dict(zip(t.w, t.n))


DEPS_URL = "https://geo.api.gouv.fr/departements?fields=nom,code"
# Eurostat LFS lfsa_ehomp, France 2025, employed 15-64: 11.1% usually work from home, 24.4% sometimes;
# judged at about 0.6 and 0.2 of their days (T-289)
WFH = 0.12


def _name_key(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s).lower() if c.isalnum())


def nuts_departements(nuts_ids):
    """The département codes of NUTS-3 2024 units (each a département), matched by name."""
    import world_plan as wp
    names = {f["properties"]["NUTS_ID"]: f["properties"]["NAME_LATN"]
             for f in json.loads(wp.fetch(*wp.NUTS).read_text(encoding="utf-8"))["features"]}
    codes = {_name_key(d["nom"]): d["code"] for d in _get_json(DEPS_URL, raw("departements.json"))}
    out = [codes.get(_name_key(names.get(u, ""))) for u in nuts_ids]
    if None in out:
        raise ValueError(f"no département for NUTS {[u for u, c in zip(nuts_ids, out) if c is None]}")
    return out


def plan_hooks(row):
    """T-555: Paris's country keys for any planned row of French NUTS-3 units (départements). Not
    carried: the curve (Paris's own fit on the flows) and Charles de Gaulle's O&D share."""
    cid, units = row["id"], row["units"]["FRA"]
    deps = lambda: nuts_departements(units)  # noqa: E731
    return {
        "boundary_file": lambda: departements_boundary(cid, deps()),
        "population": lambda city, boundary, sig: commune_population(city, boundary, sig, deps()),
        "jobs": lambda city, boundary, sig, w: insee_jobs(city, boundary, sig, w, deps()),
        "jobs_url": "https://www.insee.fr/fr/statistiques/8582949",
        "wfh": WFH,
    }


def insee_jobs(city, boundary, sig, weight, deps):
    """Jobs per signal cell: INSEE 2022 workers by commune of work, spread inside each commune by
    weight; tier0's `jobs` hook."""
    cs = communes(deps)
    jobs = workplace_jobs()
    missing = [c[0] for c in cs if c[0] not in jobs]
    out = england.spread_totals(sig["lat"], sig["lon"], [(c[0], c[3]) for c in cs], jobs, weight)
    print(f"  INSEE 2022 workers by commune of work: {len(cs):,} communes ({len(missing)} with none), {out.sum():,.0f} jobs", flush=True)
    return out, "INSEE Mobilités professionnelles 2022 (census), employed people 15+ by commune of work, Paris by arrondissement"
