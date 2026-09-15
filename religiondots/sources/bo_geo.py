"""Bolivia — boundaries for the 9 departamentos and 112 provincias, with the 2024 census count.

Writes, under data/geo/bo/:
    bo_departamentos.gpkg, bo_lookup.csv, bo_pop_2024.csv        departments
    bo_provincias.gpkg, bo_prov_pop_2024.csv                     provinces
    bo_municipios.csv      COD-AB's 339 municipality names, the witness for LAPOP's `municipio`

OCHA COD-AB Bolivia v02 (`cod-ab-bol`, Ministerio de Desarrollo Rural y Tierras, valid from
2024-09-16) for the polygons. **The people are the 2024 census, not COD-PS**: COD-PS for
Bolivia is a 2022 projection from the 2012 census that totals 12,006,031 against the 11,365,333
the census counted, and misses unevenly (Pando by a quarter), which is Ecuador's case (§9bn) at
twice the size. `sources/bo_census.py` takes the count from INE's REDATAM base.

## THE JOINS

COD's pcodes are INE's codes (`BO` + department, + province), in INE's order: 01 Chuquisaca, 02
La Paz, 03 Cochabamba, 04 Oruro, 05 Potosí, 06 Tarija, 07 Santa Cruz, 08 Beni, 09 Pando.
**LAPOP's `prov` is not in that order** (1001 La Paz, 1002 Santa Cruz, ...), so LAPOP's labels
join to COD by NAME here, all nine without an alias, and the resulting code map is written into
the lookup for `sources/bo.py` to assert. The census joins on INE's code and its names are
checked beside it: all nine departments by name, and the provinces with their differences
printed, since several COD province names are the long form of the census's.

Usage:
    python sources/bo_geo.py --fetch    COD-AB (22 MB) and COD-PS from HDX
    python sources/bo_geo.py            rebuild from data/raw/bo/
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bo")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "bo")
DEPT_OUT = os.path.join(OUT_DIR, "bo_departamentos.gpkg")
PROV_OUT = os.path.join(OUT_DIR, "bo_provincias.gpkg")
LOOKUP = os.path.join(OUT_DIR, "bo_lookup.csv")
POP_OUT = os.path.join(OUT_DIR, "bo_pop_2024.csv")
PROV_POP_OUT = os.path.join(OUT_DIR, "bo_prov_pop_2024.csv")
MUNIS_OUT = os.path.join(OUT_DIR, "bo_municipios.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
DOWNLOADS = {
    "bol_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/67333305-5479-4936-a7a3-1713ec3b7398/resource/"
        "caf0f066-84a3-491f-967b-e2c1476df2c8/download/bol_admin_boundaries.shp.zip",
    # only to print how far the projection is from the count
    "bol_admpop_adm1_2022.csv":
        "https://data.humdata.org/dataset/095b9a42-8880-477b-9fa9-e84054c6da98/resource/"
        "557d7e42-af6c-4138-960b-794570816bf7/download/bol_admpop_adm1_2022b.csv",
}

# LAPOP's `prov` value labels 2008-2018, verbatim ("Santa cruz" in 2012 folds the same).
LAPOP_DEPARTMENTS = {1001: "La Paz", 1002: "Santa Cruz", 1003: "Cochabamba", 1004: "Oruro",
                     1005: "Chuquisaca", 1006: "Potosí", 1007: "Pando", 1008: "Tarija",
                     1009: "Beni"}

CENSUS_TOTAL = 11_365_333
CODPS_TOTAL = 12_006_031

# The 2024 census counts the Territorio Indígena Multiétnico (Beni) as an area of its own, code
# 0809; COD-AB has no polygon for it. Ley 1497 of 2023-03-01 created it out of two municipalities,
# San Ignacio de Moxos (Moxos) and Santa Ana de Yacuma (Yacuma), per CIPCA's report of the law,
# and no split of its people between them is published. All 3,973 go to Moxos. Only the
# province layer can see the choice, and it moves 0.035% of the country either way.
TIOC_INTO = {"BO0809": ("BO0805", "Moxos")}
TIOC_REASON = ("Ley 1497 carved it from San Ignacio de Moxos (Moxos) and Santa Ana de Yacuma "
               "(Yacuma), split unpublished")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=1800) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def layer(level, geometry=True):
    return gpd.read_file(os.path.join(SHP_DIR, f"bol_admin{level}.shp"), engine="pyogrio",
                         ignore_geometry=not geometry)


def main():
    if "--fetch" in sys.argv:
        fetch()
    zpath = os.path.join(RAW, "bol_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    for p in ("cpv2024_pop_depto.csv", "cpv2024_pop_provin.csv", "cpv1992_religion_provin.csv"):
        if not os.path.exists(os.path.join(RAW, p)):
            raise SystemExit(f"{p} missing — run sources/bo_census.py --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if n.startswith(("bol_admin1.", "bol_admin2.",
                                                            "bol_admin3."))]
        if len(members) != 15:
            raise SystemExit(f"{len(members)} admin1-3 members in the COD-AB zip — it has been re-cut")
        for n in members:
            z.extract(n, SHP_DIR)

    # ---------------------------------------------------------------- departments
    g = layer(1)
    if len(g) != 9:
        raise SystemExit(f"{len(g)} ADM1 features, expected 9")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    g["name"] = g["adm1_name"].astype(str).str.strip()
    if sorted(g["pcode"]) != [f"BO0{i}" for i in range(1, 10)]:
        raise SystemExit(f"ADM1 pcodes {sorted(g['pcode'])}")
    by_name = dict(zip(g["name"].map(fold), g["pcode"]))
    missing = [n for n in LAPOP_DEPARTMENTS.values() if fold(n) not in by_name]
    if missing:
        raise SystemExit(f"LAPOP labels with no COD name: {missing}")
    lapop_map = {c: by_name[fold(n)] for c, n in LAPOP_DEPARTMENTS.items()}
    if len(set(lapop_map.values())) != 9:
        raise SystemExit("two LAPOP labels join to one department")
    print(f"read ADM1: 9 departments. LAPOP's prov joins by name, no alias: "
          + ", ".join(f"{c}->{p}" for c, p in lapop_map.items()))

    cen = pd.read_csv(os.path.join(RAW, "cpv2024_pop_depto.csv"), dtype={"code": str})
    cen["geo_id"] = "BO" + cen["code"].str.strip().str.zfill(2)
    if set(cen["geo_id"]) != set(g["pcode"]) or int(cen["pop"].sum()) != CENSUS_TOTAL:
        raise SystemExit("the 2024 department count does not cover COD's nine pcodes or does not "
                         f"total {CENSUS_TOTAL:,}")
    ab = dict(zip(g["pcode"], g["name"]))
    differ = [(p, n, ab[p]) for p, n in zip(cen["geo_id"], cen["name"]) if fold(n) != fold(ab[p])]
    if differ:
        raise SystemExit(f"census department names disagree with COD at the same code: {differ}")
    print(f"  census 2024 joins on INE's code and all nine names agree; {CENSUS_TOTAL:,} people")
    g = g.merge(cen[["geo_id", "pop"]], left_on="pcode", right_on="geo_id")

    cp = pd.read_csv(os.path.join(RAW, "bol_admpop_adm1_2022.csv"), encoding="utf-8-sig")
    if int(cp["T_TL"].sum()) != CODPS_TOTAL:
        raise SystemExit(f"COD-PS 2022 sums to {int(cp['T_TL'].sum()):,}")
    proj = dict(zip(cp["ADM1_PCODE"], cp["T_TL"]))
    print("  COD-PS 2022 (projected from 2012) against the count: "
          + ", ".join(f"{n} {proj[p] / c:.2f}x" for p, n, c in
                      sorted(zip(g["pcode"], g["name"], g["pop"]), key=lambda t: proj[t[0]] / t[2])))

    g["area_km2"] = g.to_crs(6933).geometry.area / 1e6
    g["density"] = g["pop"] / g["area_km2"]
    big, dense = g.loc[g["pop"].idxmax(), "name"], g.loc[g["density"].idxmax(), "name"]
    print(f"    largest {big}, densest {dense}, sparsest {g.loc[g['density'].idxmin(), 'name']}")
    if (fold(big), fold(dense)) != ("santacruz", "cochabamba"):
        raise SystemExit("Santa Cruz is not the largest or Cochabamba not the densest — the "
                         "population join is permuted")
    g["unit"] = g["pcode"]
    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]].to_file(
        DEPT_OUT, layer="departamentos", driver="GPKG")
    lut = pd.DataFrame({"geo_id": sorted(g["pcode"])})
    lut["unit"] = lut["geo_id"]
    lut["name"] = lut["geo_id"].map(ab)
    lut["lapop_prov"] = lut["geo_id"].map({p: c for c, p in lapop_map.items()})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    g[["geo_id", "name", "pop"]].sort_values("geo_id").to_csv(POP_OUT, index=False,
                                                              encoding="utf-8")
    print(f"wrote {DEPT_OUT}, {LOOKUP}, {POP_OUT}")

    # ---------------------------------------------------------------- provinces
    p2 = layer(2)
    if len(p2) != 112:
        raise SystemExit(f"{len(p2)} ADM2 features, expected 112")
    p2["pcode"] = p2["adm2_pcode"].astype(str).str.strip()
    p2["name"] = p2["adm2_name"].astype(str).str.strip()
    p2["parent"] = p2["adm1_pcode"].astype(str).str.strip()
    if (p2["pcode"].str[:4] != p2["parent"]).any():
        raise SystemExit("ADM2 pcodes do not start with their ADM1 pcode")
    names2 = dict(zip(p2["pcode"], p2["name"]))
    for label, fn in (("2024 count", "cpv2024_pop_provin.csv"),
                      ("1992 religion", "cpv1992_religion_provin.csv")):
        t = pd.read_csv(os.path.join(RAW, fn), dtype={"code": str})
        t["geo_id"] = "BO" + t["code"].str.strip().str.zfill(4)
        for tioc, (into, into_name) in TIOC_INTO.items():
            m = t["geo_id"] == tioc
            if not m.any():
                continue
            if fold(names2[into]) != fold(into_name):
                raise SystemExit(f"{into} is {names2[into]!r} in COD, not {into_name!r}")
            n = int(t.loc[m, "pop"].sum())
            print(f"  {label}: {t.loc[m, 'name'].iloc[0]} ({tioc}, {n:,} people) folded into "
                  f"{names2[into]} ({into}), {TIOC_REASON}")
            t.loc[t["geo_id"] == into, "pop"] += n
            t = t[~m].copy()
        if set(t["geo_id"]) != set(p2["pcode"]):
            raise SystemExit(f"{label}: province codes {sorted(set(t['geo_id']) ^ set(p2['pcode']))}"
                             " are on one side only")
        diff = [(c, n, names2[c]) for c, n in zip(t["geo_id"], t["name"])
                if fold(n) != fold(names2[c])]
        print(f"  {label}: all 112 province codes are COD's; names agree on {112 - len(diff)}, "
              f"differ on {len(diff)}: " + "; ".join(f"{c} {a!r}/{b!r}" for c, a, b in diff))
        if len(diff) > 20:
            raise SystemExit(f"{label}: {len(diff)} province names disagree at the same code — "
                             "too many to be spelling")
        if label == "2024 count":
            if int(t["pop"].sum()) != CENSUS_TOTAL:
                raise SystemExit("the 2024 province count does not total the census")
            p2 = p2.merge(t[["geo_id", "pop"]], left_on="pcode", right_on="geo_id")
    roll = p2.groupby("parent")["pop"].sum().reindex(g["pcode"]).to_numpy()
    if not (roll == g["pop"].to_numpy()).all():
        raise SystemExit("provinces do not sum to their departments' counts")
    top = p2.nlargest(3, "pop")
    print("    largest provinces: " + ", ".join(f"{n} {int(v):,}" for n, v in
                                                  zip(top["name"], top["pop"])))
    p2["unit"] = p2["pcode"]
    p2[["unit", "name", "pcode", "geo_id", "parent", "pop", "geometry"]].to_file(
        PROV_OUT, layer="provincias", driver="GPKG")
    p2[["geo_id", "name", "parent", "pop"]].sort_values("geo_id").to_csv(
        PROV_POP_OUT, index=False, encoding="utf-8")
    print(f"wrote {PROV_OUT}, {PROV_POP_OUT}")

    # ---------------------------------------------------------------- municipalities
    m3 = layer(3, geometry=False)
    if len(m3) != 339:
        raise SystemExit(f"{len(m3)} ADM3 features, expected 339")
    m3[["adm3_name", "adm3_pcode", "adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode"]].to_csv(
        MUNIS_OUT, index=False, encoding="utf-8")
    print(f"wrote {MUNIS_OUT} (339 municipalities)")


if __name__ == "__main__":
    main()
