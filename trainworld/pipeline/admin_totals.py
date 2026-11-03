"""T-2084: census totals by admin unit for tier 0's `admin_totals` hook, read from one config entry
per country (CONFIGS); no code per country.

    python pipeline/admin_totals.py check <ISO2>          the join: codes unique, every unit a total, parents' sums
    python pipeline/admin_totals.py ghsl <ISO2>           GHSL's people per unit, cached (the shares mode needs it)
    python pipeline/admin_totals.py dry <ISO2> [ids..]    per built pack: GHSL's people, and the census's after scaling

A config names a polygon file (layer, code field, name field) and a totals file (an .xlsx sheet, a
.csv or a layer of the same geodatabase; code and population fields), so it covers COD-AB beside
COD-PS, USCB's geodatabases, and a statistics office's own geography service saved to a file.
`parent` is a regex whose one group is a unit code's parent code: the check sums units by it against
the parent rows of the totals file. Modes (research/next-countries-data.md "The general fix"):
- "factor": each unit gets its census total (tier0.scale_to_units then gives the pack the part of it
  inside the outline, by GHSL's share).
- "shares", for an old count: each unit gets GHSL's people over its `hold` parent (default `parent`)
  times the unit's census share of that parent, so the years of growth since the count are GHSL's.

A country module's plan_hooks puts `admin_totals: admin_totals.hook("XX")` in its rows' config; for a
country with no module, this file's own plan_hooks does (a zone rule's "module": "admin_totals").
Files: downloads in data/raw/admin_totals/<ISO2>/ (or a `mirror` path under the maps folder read in
place), GHSL per unit in data/work/admin_totals/.
"""
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import shapely
from shapely import make_valid

import build_city as b

RAW = os.path.join(b.RAW, "admin_totals")
WORK = os.path.join(b.WORK, "admin_totals")
MAPS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(b.RAW))))  # data may be a junction (agent trees)

USCB_ID = "https://data.humdata.org/dataset/6daa36c4-9ea3-4844-958b-042d2cc3d8b7/resource/"
USCB_PH = "https://data.humdata.org/dataset/809dfb22-77f4-482c-8560-79b07d20fc15/resource/"

CONFIGS = {
    "ID": {
        "source": "U.S. Census Bureau, Indonesia Subnational Population and Housing Data Tables (2020 census Long Form, "
                  "mid-2022, by kabupaten/kota), HDX, CC BY 4.0",
        "polygons": {"file": "ID/indonesia.gdb.zip", "url": USCB_ID + "b1763b8a-1c6b-4a2b-8f79-ddf92bfc0f82/download/indonesia.gdb.zip",
                     "layer": "ID_GEOG_ADM2_2020_uscb_202405", "code": "NSO_CODE", "name": "ADM2_NAME"},
        "totals": {"file": "ID/indonesia_uscb_202405.xlsx",
                   "url": USCB_ID + "2c8fce76-8ad7-4bce-ab02-374318302383/download/indonesia_uscb_202405.xlsx",
                   "sheet": "Age-Sex", "skiprows": [1], "code": "NSO_CODE", "pop": "BTOTL"},  # row 1 repeats the header in words
        "parent": r"^(\d\d)\d\d$",
        "national": "0",
        "no_total": ["1288"],  # Danau Toba: the lake
        "mode": "factor",
    },
    "PH": {
        "source": "U.S. Census Bureau, Philippines Subnational Population and Housing Data Tables (2020 census, by "
                  "municipality and city), HDX, CC BY 4.0",
        "polygons": {"file": "PH/philippines.gdb.zip", "url": USCB_PH + "f7835bf0-8168-4f4e-9961-108167c381a1/download/philippines.gdb.zip",
                     "mirror": "religiondots/data/raw/ph/philippines_uscb.gdb.zip",
                     "layer": "PH_GEOG_ADM3_2020_uscb_202402", "code": "GEO_MATCH", "name": "ADM3_NAME"},
        "totals": {"file": "PH/philippines.gdb.zip", "mirror": "religiondots/data/raw/ph/philippines_uscb.gdb.zip",
                   "layer": "PH_AGE_SEX_2020census_uscb_202402", "code": "GEO_MATCH", "pop": "BTOTL"},
        "parent": r"^(PHL_\d+_\d+)_\d+$",
        "national": "PHL_00",
        "mode": "factor",
    },
}


def source_path(spec):
    """The spec's file: in data/raw/admin_totals, else its mirror read in place, else fetched with plain
    curl (HDX answers a browser-like agent with an empty challenge, research/next-countries-data.md)."""
    path = os.path.join(RAW, spec["file"])
    if os.path.exists(path):
        return path
    if spec.get("mirror") and os.path.exists(os.path.join(MAPS, spec["mirror"])):
        return os.path.join(MAPS, spec["mirror"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.part"
    print(f"  downloading {spec['url']}", flush=True)
    subprocess.run(["curl", "-sS", "-L", "--fail", "-o", tmp, spec["url"]], check=True)
    if os.path.getsize(tmp) == 0 or (path.endswith(".zip") and not zipfile.is_zipfile(tmp)):
        os.remove(tmp)
        raise SystemExit(f"{spec['url']}: an empty or broken file (a bot check?); fetch it in a browser into {path}")
    os.replace(tmp, path)
    return path


def _codes(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def read_totals(cfg):
    """DataFrame code, pop: every row of the totals file, every level."""
    t = cfg["totals"]
    path = source_path(t)
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=t["sheet"], skiprows=t.get("skiprows"), dtype={t["code"]: str})
    elif path.endswith(".csv"):
        df = pd.read_csv(path, dtype={t["code"]: str}, skiprows=t.get("skiprows"))
    else:
        import pyogrio
        df = pyogrio.read_dataframe(path, layer=t["layer"], columns=[t["code"], t["pop"]], read_geometry=False)
    if not len(df):
        raise SystemExit(f"{path} {t.get('layer') or t.get('sheet')}: no rows")
    return pd.DataFrame({"code": _codes(df[t["code"]]), "pop": pd.to_numeric(df[t["pop"]], errors="coerce")})


def read_polygons(cfg):
    """DataFrame code, name, geometry (lon/lat shapely)."""
    import pyogrio
    p = cfg["polygons"]
    path = source_path(p)
    gdf = pyogrio.read_dataframe(path, layer=p["layer"], columns=[p["code"], p["name"]])
    if not len(gdf):  # a geodatabase can open cleanly and return no features (research, Indonesia)
        raise SystemExit(f"{path} {p['layer']}: no features")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return pd.DataFrame({"code": _codes(gdf[p["code"]]), "name": gdf[p["name"]].astype(str).str.strip(),
                         "geometry": make_valid(gdf.geometry.to_numpy())})


_UNITS = {}


def units(iso):
    """DataFrame code, name, geometry, census, parent for the config's polygons with a total."""
    if iso not in _UNITS:
        cfg = CONFIGS[iso]
        pol, tot = read_polygons(cfg), read_totals(cfg)
        df = pol.merge(tot.drop_duplicates("code"), on="code", how="left").rename(columns={"pop": "census"})
        df = df[~df.code.isin(cfg.get("no_total", []))].reset_index(drop=True)
        df["parent"] = df.code.str.extract(cfg["parent"], expand=False)
        _UNITS[iso] = df
    return _UNITS[iso]


def join_check(iso):
    """(ok, lines): codes unique on both sides, every polygon a total and every unit row a polygon,
    units summed by parent equal to the parent rows, and to the national row."""
    cfg = CONFIGS[iso]
    pol, tot = read_polygons(cfg), read_totals(cfg)
    lines, ok = [], True

    def say(good, text):
        nonlocal ok
        ok &= bool(good)
        lines.append(f"{'ok  ' if good else 'BAD '} {text}")

    dp, dt = pol.code[pol.code.duplicated()], tot.code[tot.code.duplicated()]
    say(dp.empty, f"{len(pol):,} polygons, codes unique{'' if dp.empty else ': repeated ' + ', '.join(dp[:5])}")
    say(dt.empty, f"{len(tot):,} total rows, codes unique{'' if dt.empty else ': repeated ' + ', '.join(dt[:5])}")
    have = dict(zip(tot.code, tot["pop"]))
    missing = [c for c in pol.code if not np.isfinite(have.get(c, np.nan)) and c not in cfg.get("no_total", [])]
    say(not missing, f"polygons without a total: {len(missing)} {missing[:5]} (config no_total: {cfg.get('no_total', [])})")
    unit_rows = tot[tot.code.str.fullmatch(cfg["parent"])]
    orphan = sorted(set(unit_rows.code) - set(pol.code))
    say(not orphan, f"{len(unit_rows):,} unit rows in the totals, without a polygon: {len(orphan)} {orphan[:5]}")
    u = units(iso)
    by = u.groupby("parent")["census"].sum()
    off = [(p, s, have.get(p)) for p, s in by.items() if have.get(p) is None or abs(s - have[p]) > 0.5]
    say(not off, f"units summed into {len(by)} parents equal the parent rows; off: {len(off)} "
                 + ", ".join(f"{p} {s:,.0f} vs {t if t is None else f'{t:,.0f}'}" for p, s, t in off[:5]))
    nat = have.get(cfg.get("national"))
    say(nat is not None and abs(u.census.sum() - nat) < 0.5, f"units sum to {u.census.sum():,.0f}; national row {nat if nat is None else f'{nat:,.0f}'}")
    return ok, lines


def ghsl_table(iso, epoch=2020):
    """GHSL's people per unit over its whole polygon, {code: people}; cached, made again when the
    config's units change."""
    path = os.path.join(WORK, f"{iso}.ghsl_e{epoch}.csv")
    u = units(iso)
    if os.path.exists(path):
        t = pd.read_csv(path, dtype={"code": str})
        if set(t.code) == set(u.code):
            return dict(zip(t.code, t.ghsl))
    import tier0
    g = tier0.ghsl_people(u.geometry.to_numpy(), epoch)
    os.makedirs(WORK, exist_ok=True)
    pd.DataFrame({"code": u.code, "ghsl": np.round(g, 1)}).to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)
    return dict(zip(u.code, g))


def held_shares(census, ghsl, parent):
    """Each unit's census share of its parent times GHSL's people over the parent: the totals of an old
    count with the parents' growth since left to GHSL (arrays, one per unit)."""
    census, ghsl = np.asarray(census, float), np.asarray(ghsl, float)
    df = pd.DataFrame({"c": census, "g": ghsl, "p": np.asarray(parent)})
    cs, gs = df.groupby("p")["c"].transform("sum"), df.groupby("p")["g"].transform("sum")
    return np.where(cs > 0, census * gs / cs.where(cs > 0, 1.0), 0.0)


def totals(iso, epoch=2020):
    """{code: the total the hook hands tier0} by the config's mode."""
    cfg, u = CONFIGS[iso], units(iso)
    if cfg["mode"] == "factor":
        return dict(zip(u.code, u.census))
    if cfg["mode"] != "shares":
        raise ValueError(f"{iso}: mode {cfg['mode']!r}")
    g = ghsl_table(iso, epoch)
    hold = u.code.str.extract(cfg.get("hold", cfg["parent"]), expand=False)
    return dict(zip(u.code, held_shares(u.census, [g.get(c, 0.0) for c in u.code], hold)))


def hook(iso, epoch=2020):
    """tier0's admin_totals for a country's config: boundary -> [(name, geometry, total)], the units
    touching the boundary."""
    def admin_totals(boundary):
        u = units(iso)
        tot = totals(iso, epoch)
        geoms = u.geometry.to_numpy()
        hit = shapely.STRtree(geoms).query(boundary, predicate="intersects")
        names, codes = u["name"].to_numpy(), u["code"].to_numpy()
        out = [(f"{names[i]} {codes[i]}", geoms[i], tot[codes[i]]) for i in sorted(hit) if np.isfinite(tot[codes[i]])]
        print(f"  {CONFIGS[iso]['source']}: {len(out)} units touch the outline, {sum(r[2] for r in out):,.0f} people", flush=True)
        return out
    cfg = CONFIGS[iso]
    admin_totals.source = cfg["source"] + (", as shares within each parent of GHSL's people" if cfg["mode"] == "shares" else "")
    return admin_totals


def plan_hooks(row):
    """The keys for a planned row of a country with a config, so a zone rule's `module` can be this
    file with no country module."""
    return {"admin_totals": hook(row["iso2"])}


# ---------------------------------------------------------------- the dry report

def outline(cid):
    import json
    from shapely.geometry import shape
    with open(os.path.join(b.WORK, f"{cid}.outline.geojson"), encoding="utf-8") as f:
        fc = json.load(f)
    return shapely.union_all([make_valid(shape(ft["geometry"])) for ft in (fc["features"] if "features" in fc else [fc])])


def dry_pack(iso, cid, epoch=2020, quiet=True):
    """What the hook would do to a built pack, from its signal cache and outline (nothing written):
    {id, ghsl (its people today), census (after scaling), old (the old rule: each touching unit's
    whole total into its cells), units, cut, flagged, province (the parent holding most of its people)}."""
    import contextlib
    import io
    import tier0
    z = np.load(os.path.join(b.WORK, f"{cid}.tier0_signals.npz"))
    out = {"lat": z["lat"], "lon": z["lon"], "pop": z["pop"].astype(float).copy(), "pop_ghs": z["pop_ghs"]}
    before = float(out["pop"].sum())
    boundary = outline(cid)
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        us = hook(iso, epoch)(boundary)
        rows = tier0.scale_to_units(out, boundary, us, epoch)
    old = before + sum(r["census"] - r["before"] for r in rows if r["before"] > 0)
    par = units(iso).set_index("code").parent
    prov = {}
    for (name, _, _), r in zip(us, rows):
        p = par.get(name.rsplit(" ", 1)[1])
        prov[p] = prov.get(p, 0.0) + r["after"]
    main = [r for r in rows if r["share"] >= 0.5]
    return {"id": cid, "ghsl": before, "census": float(out["pop"].sum()), "old": old, "units": len(us),
            "main_census": sum(r["census"] for r in main), "main_ghsl": sum(r["ghsl"] for r in main),
            "cut": sum(0.02 < r["share"] < 0.98 for r in rows), "flagged": sum(r["flag"] for r in rows),
            "province": max(prov, key=prov.get) if prov else None}


def plan_ids(iso):
    import plan_config
    rows, _ = plan_config.load()
    return [r["id"] for r in rows if r.get("iso2") == iso and os.path.exists(os.path.join(b.WORK, f"{r['id']}.tier0_signals.npz"))]


def main(argv):
    if len(argv) < 2 or argv[0] not in ("check", "ghsl", "dry"):
        raise SystemExit(__doc__)
    cmd, iso = argv[0], argv[1]
    if cmd == "check":
        ok, lines = join_check(iso)
        print("\n".join(lines))
        raise SystemExit(0 if ok else 1)
    if cmd == "ghsl":
        g = ghsl_table(iso)
        u = units(iso)
        print(f"{len(g):,} units: GHSL {sum(g.values()):,.0f}, census {u.census.sum():,.0f}")
        return
    ids = argv[2:] or plan_ids(iso)
    print("id\tGHSL\tcensus\told rule\tunits\tcut\tFLAG\tparent\tmain units census\tmain units GHSL", flush=True)
    for cid in ids:
        r = dry_pack(iso, cid)
        print(f"{r['id']}\t{r['ghsl']:,.0f}\t{r['census']:,.0f}\t{r['old']:,.0f}\t{r['units']}\t{r['cut']}\t{r['flagged']}\t{r['province']}"
              f"\t{r['main_census']:,.0f}\t{r['main_ghsl']:,.0f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:])
