"""Checks for helper1m Iran, run after fetch.py. Read-only.

1. Totals: national and province per year against SCI's published figures.
2. The 1390 carry: counties that kept their 1390 name, against the 1390 county rows; growth
   1390->1395 by county, extremes listed.
3. Every boundary unit has a population, every population row a unit; levels nest.
4. Kontur 2023 (religiondots' county table, read only) against the 2024 column.
5. A few big counties spot-checked against figures printed elsewhere.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
import census  # noqa: E402
from census import fold  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
DATA = HELPER / "data" / "iran"
KONTUR = REPO / "religiondots" / "data" / "geo" / "ir" / "ir_county_lookup.csv"
HEXES = REPO / "religiondots" / "data" / "geo" / "ir" / "ir_hexes.gpkg"
BOUNDS = {1: (REPO / "data/asia1m/iran/irn_admin1.shp", "adm1_pcode"),
          2: (REPO / "data/asia1m/iran/irn_admin2.shp", "adm2_pcode"),
          3: (DATA / "boundaries" / "adm3.gpkg", "code")}

# Printed elsewhere, for spot checks (English Wikipedia county articles cite the same censuses).
SPOT = {"شیراز": None, "مشهد": None, "تهران": None, "اصفهان": None, "کرج": None,
        "تبریز": None, "اهواز": None, "قم": None}


def unit_of_county_codes():
    """(SCI province, SCI county) -> COD adm2_pcode, from counties.csv."""
    c = pd.read_csv(DATA / "counties.csv", dtype=str)
    return {(s[:2], s[2:4]): p for s, p in zip(c["sci_code"], c["adm2_pcode"])}


def main():
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    cty = pd.read_csv(DATA / "counties.csv", dtype=str)
    for c in ("pop2011", "pop2016", "pop2024", "nomad2016"):
        cty[c] = cty[c].astype(int)

    print("== 1. totals")
    for (lv, y), g in pop.groupby(["level", "year"]):
        print(f"  level {lv} {y}: {g['pop'].sum():,} over {len(g)} units")

    print("\n== 2. the 1390 carry")
    # 1390 county rows, by province + folded name
    c90 = {}
    for i in range(31):
        nn = f"{i:02d}"
        for r in census.read90(nn):
            if r["kind"] == "county":
                c90[(nn, fold(r["name"]))] = r["pop"]
    cty["key"] = list(zip(cty["sci_code"].str[:2], cty["name_fa"].map(fold)))
    same = cty[cty["key"].isin(c90)].copy()
    same["c90"] = same["key"].map(c90)
    same["r"] = same["pop2011"] / same["c90"]
    exact = (same["pop2011"] == same["c90"]).sum()
    print(f"  {len(same)} of 429 counties carry a 1390 county's name; {exact} equal its 1390 row "
          f"exactly, {((same['r'] - 1).abs() < 0.01).sum()} within 1%")
    off = same[(same["r"] - 1).abs() >= 0.01].sort_values("r")
    for _, r in off.iterrows():
        print(f"    {r['sci_code']} {r['name_en']:<22} carried {r['pop2011']:>9,} vs 1390 row "
              f"{r['c90']:>9,}  x{r['r']:.3f}")
    g = (cty["pop2016"] / cty["pop2011"]) ** (1 / 5) - 1
    cty["g"] = g
    q = g.quantile([0, .05, .5, .95, 1])
    print(f"  annual growth 1390->1395 by county: min {q[0]:+.1%}, p5 {q[.05]:+.1%}, median "
          f"{q[.5]:+.1%}, p95 {q[.95]:+.1%}, max {q[1]:+.1%}")
    for _, r in pd.concat([cty.nsmallest(5, "g"), cty.nlargest(8, "g")]).iterrows():
        print(f"    {r['sci_code']} {r['name_en']:<22} {r['pop2011']:>9,} -> {r['pop2016']:>9,}"
              f"  {r['g']:+.1%}/yr")
    g2 = (cty["pop2024"] / cty["pop2016"]) ** (1 / 8) - 1
    q = g2.quantile([0, .05, .5, .95, 1])
    print(f"  annual growth 1395->1403 (forward cast): min {q[0]:+.1%}, p5 {q[.05]:+.1%}, "
          f"median {q[.5]:+.1%}, p95 {q[.95]:+.1%}, max {q[1]:+.1%}")

    print("\n== 3. coverage and nesting")
    for lv, (path, col) in BOUNDS.items():
        if not path.exists():
            print(f"  adm{lv}: no boundary file")
            continue
        b = pyogrio.read_dataframe(path, read_geometry=False)
        codes = set(b[col])
        have = set(pop[pop.level == lv]["code"])
        yrs = pop[pop.level == lv].groupby("code")["year"].nunique()
        print(f"  adm{lv}: {len(codes)} units; no population {len(codes - have)}, "
              f"no unit {len(have - codes)}, with <3 years {int((yrs < 3).sum())}")
    p1 = pop[pop.level == 1].set_index(["code", "year"])["pop"]
    p2 = pop[pop.level == 2].copy()
    p2["prov"] = p2["code"].str[:5]
    s2 = p2.groupby(["prov", "year"])["pop"].sum()
    bad = [(k, int(p1[k]), int(s2[k])) for k in p1.index if p1[k] != s2.get(k)]
    print(f"  counties sum to provinces: {'all' if not bad else bad}")
    if (pop.level == 3).any():
        p3 = pop[pop.level == 3].copy()
        par = pyogrio.read_dataframe(BOUNDS[3][0], read_geometry=False).set_index("code")["parent"]
        p3["par"] = p3["code"].map(par)
        s3 = p3.groupby(["par", "year"])["pop"].sum()
        p2i = pop[pop.level == 2].set_index(["code", "year"])["pop"]
        bad = [(k, int(p2i[k]), int(s3.get(k, -1))) for k in p2i.index if p2i[k] != s3.get(k)]
        print(f"  districts sum to counties: {'all' if not bad else bad[:10]}")

    print("\n== 4. Kontur 2023 against the 2024 column (county)")
    k = pd.read_csv(KONTUR, dtype={"county": str}).set_index("county")["kontur_pop_2023"]
    df = cty.set_index("adm2_pcode")[["name_en", "pop2016", "pop2024"]].join(k)
    nat = df["kontur_pop_2023"].sum() / df["pop2024"].sum()
    df["rel"] = df["kontur_pop_2023"] / df["pop2024"] / nat
    r = df["rel"]
    print(f"  national Kontur/2024 x{nat:.3f}; after that, counties within 10% "
          f"{((r - 1).abs() <= .1).mean():.0%}, within 25% {((r - 1).abs() <= .25).mean():.0%}, "
          f"quartiles {r.quantile(.25):.2f}-{r.quantile(.75):.2f}")
    print("  (religiondots found Kontur puts 18% of Iran in the wrong county, with false cities at "
          "its density cap, so this is a weak witness)")
    for c, row in df.assign(dev=r.apply(lambda v: max(v, 1 / v))).nlargest(8, "dev").iterrows():
        print(f"    {c} {row['name_en']:<22} 2024 {row['pop2024']:>9,} Kontur "
              f"{row['kontur_pop_2023']:>11,.0f} x{row['rel']:.2f}")

    if (pop.level == 3).any():
        print("\n== 4b. districts: religiondots' county-calibrated Kontur hexes against 2016")
        import geopandas as gpd
        hx = gpd.read_file(HEXES, columns=["county", "pop"])
        hx["geometry"] = hx.geometry.centroid
        a3 = gpd.read_file(BOUNDS[3][0])[["code", "name", "parent", "geometry"]].to_crs(hx.crs)
        j = gpd.sjoin(hx, a3, how="inner", predicate="within")
        kk = j.groupby("code")["pop"].sum()
        p16 = pop[(pop.level == 3) & (pop.year == 2016)].set_index("code")["pop"]
        d = pd.DataFrame({"census": p16, "kontur": kk}).fillna(0)
        d["name"] = a3.set_index("code")["name"]
        # only districts in counties with 2+ units say anything (else it is 1.0 by construction)
        multi = a3.groupby("parent")["code"].transform("size") > 1
        d = d[d.index.isin(a3.loc[multi, "code"])]
        r = d["kontur"] / d["census"]
        print(f"  {len(d)} districts in counties with more than one: within 10% "
              f"{((r - 1).abs() <= .1).mean():.0%}, within 25% {((r - 1).abs() <= .25).mean():.0%}"
              f", weighted by people within 25% "
              f"{d.loc[(r - 1).abs() <= .25, 'census'].sum() / d['census'].sum():.0%}")
        big = d[d["census"] > 20000].assign(r=r).assign(dev=lambda x: x["r"].apply(
            lambda v: max(v, 1 / v) if v > 0 else 99))
        for c, row in big.nlargest(12, "dev").iterrows():
            print(f"    {c:<16} {row['name']:<30} census {int(row['census']):>9,} Kontur "
                  f"{row['kontur']:>11,.0f} x{row['r']:.2f}")

    if (pop.level == 3).any():
        print("\n== 4c. districts: 1395 settlements located by OSM place nodes")
        import geopandas as gpd
        sys.path.insert(0, str(Path(__file__).parent))
        import bakhsh
        a3 = gpd.read_file(BOUNDS[3][0]).to_crs(4326)
        unit_of = {}
        for _, r in a3.iterrows():
            for s in r["sci_districts"].split("|"):
                unit_of[(s[:2], s[2:4], s[4:6])] = r["code"]
        a3["pcs_id"] = range(len(a3))
        comp = bakhsh.located_composition(a3, unit_of_county_codes(), gpd.read_file(
            BOUNDS[2][0]).rename(columns={"code": "adm2_pcode"}))
        own = tot = 0
        per = []
        for i, (_, r) in enumerate(a3.iterrows()):
            c = comp[i]
            t = sum(c.values())
            o = sum(v for T, v in c.items() if unit_of.get(T) == r["code"])
            own += o
            tot += t
            per.append((r["code"], r["name"], o, t))
        print(f"  {own / tot:.1%} of located 1395 people fall inside their own district's polygon")
        bad = sorted([p for p in per if p[3] > 5000 and p[2] / p[3] < 0.8], key=lambda p: p[2] / p[3])
        print(f"  units where under 80% of the located people inside belong to them: {len(bad)}")
        for code, name, o, t in bad[:12]:
            print(f"    {code:<16} {name:<30} {o:>8,.0f} of {t:>8,.0f} ({o / t:.0%})")

    print("\n== 5. biggest counties")
    for _, r in cty.nlargest(10, "pop2016").iterrows():
        print(f"    {r['adm2_pcode']} {r['name_en']:<14} 2011 {r['pop2011']:>9,}  2016 "
              f"{r['pop2016']:>9,}  2024 {r['pop2024']:>9,}")


if __name__ == "__main__":
    main()
