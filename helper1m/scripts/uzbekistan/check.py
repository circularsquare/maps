"""Checks for the Uzbekistan build. Run after fetch.py and prep_boundaries.py.

1. Census switch off: totals against SIAT, every year.
2. Census switch on: each region in 2026 against the census.
3. Kontur 2023 (400 m hexes, centroid in polygon) against our 2023 district
   figures, both as shares of the national total.
4. Every boundary unit has population, every population code is a boundary unit.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import fetch  # noqa: E402

HELPER = fetch.HELPER
KONTUR = HELPER.parent / "religiondots" / "data" / "geo" / "kontur" / "kontur_population_UZ_20231101.gpkg"
KONTUR_CACHE = HELPER / "data" / "uzbekistan" / "kontur_adm2_2023.csv"


def kontur_by_district():
    if KONTUR_CACHE.exists():
        return pd.read_csv(KONTUR_CACHE, dtype={"code": str}).set_index("code")["kontur"]
    import geopandas as gpd
    k = gpd.read_file(KONTUR)
    a = gpd.read_file(HELPER / "data" / "uzbekistan" / "boundaries" / "adm2.gpkg").to_crs(k.crs)
    pts = k[["population"]].copy()
    pts = gpd.GeoDataFrame(pts, geometry=k.geometry.centroid, crs=k.crs)
    j = gpd.sjoin(pts, a[["code", "geometry"]], how="left", predicate="within")
    print(f"Kontur: {k['population'].sum():,.0f} people, {j.loc[j['code'].isna(), 'population'].sum():,.0f} with centroid outside every district")
    s = j.groupby("code")["population"].sum()
    s = s.reindex(a["code"]).fillna(0)
    s.rename("kontur").to_csv(KONTUR_CACHE, index_label="code")
    return s


def main():
    rows, _ = fetch.load_siat()
    siat = {r["Code"]: r for r in rows}
    units = pd.read_csv(HELPER / "data" / "uzbekistan" / "units.csv", dtype=str).set_index("code")

    print("== 1. SIAT only (CENSUS_LEVEL off)")
    raw = fetch.build(census_level=False, write=False, verbose=False)
    worst = 0
    for y in fetch.YEARS:
        tot = sum(v[y] for (lvl, c), v in raw.items() if lvl == 1)
        worst = max(worst, abs(tot - siat["1700"][str(y)] * 1000))
    print(f"  national, every year 2011-2026: largest gap to SIAT {worst:,.0f} people")
    for (lvl, c), v in sorted(raw.items()):
        if lvl != 1:
            continue
        diffs = [(y, v[y] - siat["17" + c[2:]][str(y)] * 1000) for y in fetch.YEARS]
        big = [(y, round(d)) for y, d in diffs if abs(d) > 100]
        if big:
            print(f"  {c} {units.loc[c, 'name']}: differs from SIAT region total in {big}")

    print("== 2. Census level (CENSUS_LEVEL on), 2026 against the census")
    cen = fetch.build(census_level=True, write=False, verbose=False)
    tot = 0
    for (lvl, c), v in sorted(cen.items()):
        if lvl != 1:
            continue
        cc = fetch.CENSUS_2026["17" + c[2:]]
        tot += v[2026]
        print(f"  {c} {units.loc[c, 'name'][:28]:28} ours {v[2026]:11,}  census {cc:11,}  diff {v[2026] - cc:+9,}")
    print(f"  national ours {tot:,} census {fetch.CENSUS_NATIONAL:,}")

    print("== 3. Kontur 2023 against our 2023 districts (shares of the national total)")
    k = kontur_by_district()
    for label, res in (("SIAT only", raw), ("census level", cen)):
        ours = pd.Series({c: v[2023] for (lvl, c), v in res.items() if lvl == 2 and 2023 in v})
        df = pd.DataFrame({"ours": ours, "kontur": k}).dropna()
        df["ratio"] = (df["kontur"] / df["kontur"].sum()) / (df["ours"] / df["ours"].sum())
        within = (df["ratio"].sub(1).abs() <= 0.10).mean()
        print(f"  {label}: {len(df)} districts, {within:.0%} within 10%, "
              f"{(df['ratio'].sub(1).abs() <= 0.25).mean():.0%} within 25%, median ratio {df['ratio'].median():.3f}")
        if label == "census level":
            df["name"] = df.index.map(units["name"])
            df["region"] = df.index.map(lambda c: units.loc[units.loc[c, "parent"], "name"])
            print("  worst, Kontur share over ours:")
            for c, r in df.reindex(df["ratio"].sub(1).abs().sort_values(ascending=False).index).head(15).iterrows():
                print(f"    {c} {r['name'][:40]:40} {r['region'][:18]:18} ours {r['ours']:9,.0f} kontur {r['kontur']:9,.0f} ratio {r['ratio']:.2f}")
            reg = df.groupby("region")[["ours", "kontur"]].sum()
            reg["ratio"] = (reg["kontur"] / reg["kontur"].sum()) / (reg["ours"] / reg["ours"].sum())
            print("  by region:")
            for n, r in reg.sort_values("ratio").iterrows():
                print(f"    {n[:28]:28} {r['ratio']:.2f}")
            print("  Tashkent region districts, SIAT-only basis (where would the census's extra 19% be?):")
            dr = pd.DataFrame({"ours": pd.Series({c: v[2023] for (lvl, c), v in raw.items() if lvl == 2}), "kontur": k})
            dr["ratio"] = (dr["kontur"] / dr["kontur"].sum()) / (dr["ours"] / dr["ours"].sum())
            for c, r in dr[dr.index.str.startswith("UZ27")].sort_values("ratio").iterrows():
                print(f"    {c} {units.loc[c, 'name'][:40]:40} ours {r['ours']:9,.0f} kontur {r['kontur']:9,.0f} ratio {r['ratio']:.2f}")

    print("== 3b. Same, each city folded into the district it shares most border with,")
    print("       Tashkent city and region left out (Kontur's own artefact there, religiondots uz.md s.8/10)")
    import geopandas as gpd
    a = gpd.read_file(HELPER / "data" / "uzbekistan" / "boundaries" / "adm2.gpkg").to_crs("ESRI:54009")
    a = a[~a["group"].isin(["UZ26", "UZ27"])].set_index("code")
    host = {}
    for c, r in a.iterrows():
        if c[4] != "4":          # SOATO 4xx = city of regional subordination
            continue
        best, blen = None, 0.0
        for c2, r2 in a[a["group"] == r["group"]].iterrows():
            if c2 == c or c2[4] == "4":
                continue
            inter = r.geometry.boundary.intersection(r2.geometry.buffer(50)).length
            if inter > blen:
                best, blen = c2, inter
        host[c] = best or c
    ours = pd.Series({c: v[2023] for (lvl, c), v in cen.items() if lvl == 2 and c in a.index})
    df = pd.DataFrame({"ours": ours, "kontur": k.reindex(ours.index)})
    df["unit"] = [host.get(c, c) for c in df.index]
    g = df.groupby("unit")[["ours", "kontur"]].sum()
    g["ratio"] = (g["kontur"] / g["kontur"].sum()) / (g["ours"] / g["ours"].sum())
    dev = g["ratio"].sub(1).abs()
    print(f"  {len(g)} units outside Tashkent: {(dev <= 0.10).mean():.0%} within 10%, {(dev <= 0.25).mean():.0%} within 25%")
    g["name"] = g.index.map(units["name"])
    print("  worst:")
    for c, r in g.reindex(dev.sort_values(ascending=False).index).head(12).iterrows():
        extra = [x for x, h in host.items() if h == c]
        print(f"    {c} {r['name'][:34]:34} +{','.join(extra) or '-':16} ours {r['ours']:9,.0f} kontur {r['kontur']:9,.0f} ratio {r['ratio']:.2f}")

    print("== 4. Coverage")
    for lvl in (1, 2):
        gj = json.loads((HELPER / "countries" / "uzbekistan" / f"adm{lvl}.geojson").read_text(encoding="utf-8"))
        codes = {f["properties"]["code"] for f in gj["features"]}
        empty = [f["properties"]["code"] for f in gj["features"] if not f["properties"]["populations"]]
        pop_codes = {c for (l, c) in cen if l == lvl}
        print(f"  adm{lvl}: {len(codes)} features, {len(empty)} without population, "
              f"{len(pop_codes - codes)} population codes with no feature")
        years = [len(f["properties"]["populations"]) for f in gj["features"]]
        print(f"         years per feature: min {min(years)}, max {max(years)}")


if __name__ == "__main__":
    main()
