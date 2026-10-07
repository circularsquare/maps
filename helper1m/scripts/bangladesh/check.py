"""Checks for data/bangladesh/population.csv. Prints the numbers quoted in README.md.

  1. national totals against BBS's published figures
  2. 2022 divisions and zilas against the census district table on HDX (UNRCO
     upload of BBS's PHC 2022 dataset), which is an independent transcription
  3. 2011 zilas and divisions against the USCB rows (what the union re-cut moved)
  4. level 3: 2022 against Kontur 2023 (hex centres per v03 polygon, scaled to
     the census total), and the 2022/2011 ratio
  5. every boundary unit has both years, every row lands on a unit

Usage:  C:\\Python39\\python.exe helper1m/scripts/bangladesh/check.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "helper1m/data/bangladesh"
V03 = REPO / "data/asia1m/bangladesh/bgd_admin{n}.shp"
USCB_XLSX = REPO / "religiondots/data/raw/bd/bangladesh_uscb_202107.xlsx"
UNRCO = DATA / "raw/unrco_phc2022_admin02.xlsx"
KONTUR = DATA / "raw/kontur_population_BD_20231101.gpkg"

OFFICIAL = {2011: 144_043_696, 2022: 165_158_616}  # enumerated; 2022 adjusted is 169,828,911

# Zila spellings that differ between the UNRCO sheet / USCB and v03.
ZILA_ALIAS = {"chapainawabganj": "chapainababganj", "jessore": "jashore",
              "jhalokathi": "jhalokati", "bogra": "bogura", "comilla": "cumilla",
              "chittagong": "chattogram", "barisal": "barishal", "kishorganj": "kishoreganj",
              "maulvibazar": "moulvibazar", "netrokona": "netrakona", "shariyatpur": "shariatpur",
              "nator": "natore", "jaipurhat": "joypurhat", "jhalakati": "jhalokati",
              "khagrachari": "khagrachhari"}


def norm(s):
    # Some USCB names carry literal "Ā" escapes rather than the character.
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), str(s))
    s = s.lower().replace("ā", "a").replace("ī", "i").replace("’", "")
    s = re.sub(r"[^a-z]", "", s)
    return ZILA_ALIAS.get(s, s)


def pct(a, b):
    return 100.0 * (a - b) / b


def main():
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    v1, v2, v3 = (gpd.read_file(str(V03).format(n=n)) for n in (1, 2, 3))
    wide = {lv: pop[pop.level == lv].pivot(index="code", columns="year", values="pop")
            for lv in (1, 2, 3)}

    print("== 1. national")
    for y in (2011, 2022):
        t = wide[3][y].sum()
        print(f"  {y}: {t:,} vs official {OFFICIAL[y]:,} ({t - OFFICIAL[y]:+,}, {pct(t, OFFICIAL[y]):+.4f}%)")
    for lv in (1, 2):
        for y in (2011, 2022):
            assert wide[lv][y].sum() == wide[3][y].sum()

    print("== 2. 2022 zilas and divisions vs the UNRCO/BBS district table")
    u = pd.read_excel(UNRCO, sheet_name=" Population by Sex, Dist & Loca")
    u["k"] = u.District.map(norm)
    u["mf"] = u.Population_Male + u.Population_Female
    z = v2.assign(k=v2.adm2_name.map(norm)).merge(u, on="k", how="outer", indicator=True)
    assert (z._merge == "both").all(), z[z._merge != "both"][["adm2_name", "District"]]
    z["ours"] = z.adm2_pcode.map(wide[2][2022])
    print(f"  zilas: {len(z)} matched; ours == male+female in {(z.ours == z.mf).sum()}; "
          f"ours == total incl. hijra in {(z.ours == z.Population_Total).sum()}; "
          f"max |diff| vs total {int((z.ours - z.Population_Total).abs().max())}")
    d = z.groupby("adm1_pcode")[["ours", "Population_Total", "mf"]].sum()
    d["name"] = d.index.map(v1.set_index("adm1_pcode").adm1_name)
    d["diff_total"] = d.ours - d.Population_Total
    print(d.to_string())

    print("== 3. 2011 zilas and divisions vs USCB rows (union re-cut leakage)")
    x = pd.read_excel(USCB_XLSX, sheet_name="Age-Sex", skiprows=[1])
    x["lev"] = x.ADM_LEVEL.astype(str)
    x2 = x[x.lev == "2"].assign(k=lambda t: t.ADM2_NAME.map(norm))
    z = v2.assign(k=v2.adm2_name.map(norm)).merge(x2, on="k", how="outer", indicator=True)
    assert (z._merge == "both").all(), z[z._merge != "both"][["adm2_name", "ADM2_NAME"]]
    z["ours"] = z.adm2_pcode.map(wide[2][2011])
    z["d"] = z.ours - z.BTOTL
    print(f"  zilas exact: {(z.d == 0).sum()} of {len(z)}; moved across zila lines: "
          f"{int(z.d.abs().sum() / 2):,} people")
    print(z[z.d != 0][["adm2_name", "BTOTL", "ours", "d"]].to_string(index=False))
    x1 = x[x.lev == "1"].assign(k=lambda t: t.ADM1_NAME.map(norm))
    z1 = v1.assign(k=v1.adm1_name.map(norm)).merge(x1, on="k")
    z1["ours"] = z1.adm1_pcode.map(wide[1][2011])
    print(z1[["adm1_name", "BTOTL", "ours"]].assign(d=z1.ours - z1.BTOTL).to_string(index=False))

    print("== 4. level 3")
    w3 = wide[3].copy()
    meta = v3.set_index("adm3_pcode")
    w3["name"] = meta.adm3_name
    w3["zila"] = meta.adm2_name
    w3["ratio"] = w3[2022] / w3[2011]
    hexes = gpd.read_file(KONTUR)[["population", "geometry"]]
    pts = hexes.copy()
    pts["geometry"] = hexes.geometry.centroid
    pts = gpd.sjoin(pts.to_crs(4326), v3[["adm3_pcode", "geometry"]], predicate="within")
    k = pts.groupby("adm3_pcode").population.sum()
    print(f"  Kontur 2023 inside v03 units: {k.sum():,.0f} "
          f"(of {hexes.population.sum():,.0f}); scaled to the 2022 census total")
    w3["kontur"] = k.reindex(w3.index).fillna(0) * w3[2022].sum() / k.sum()
    w3["k_vs_census"] = pct(w3.kontur, w3[2022])
    within = (w3.k_vs_census.abs() <= 10).mean() * 100
    within20 = (w3.k_vs_census.abs() <= 20).mean() * 100
    print(f"  Kontur within 10% of the census: {within:.1f}% of units; within 20%: {within20:.1f}%")
    popw = (w3[2022] * (w3.k_vs_census.abs() <= 10)).sum() / w3[2022].sum() * 100
    print(f"  (population-weighted within 10%: {popw:.1f}%)")
    cols = ["name", "zila", 2011, 2022, "kontur", "k_vs_census", "ratio"]
    print("  worst Kontur below census:")
    print(w3.sort_values("k_vs_census")[cols].head(10).round(1).to_string())
    print("  worst Kontur above census:")
    print(w3.sort_values("k_vs_census")[cols].tail(10).round(1).to_string())
    print(f"  2022/2011 ratio: national {w3[2022].sum() / w3[2011].sum():.3f}; "
          f"units {w3.ratio.describe()[['min', '25%', '50%', '75%', 'max']].round(3).to_dict()}")
    print("  lowest and highest 2022/2011 ratios:")
    print(w3.sort_values("ratio")[cols].head(8).round(3).to_string())
    print(w3.sort_values("ratio")[cols].tail(12).round(3).to_string())

    print("== 4b. level 3: 2022 census vs the USCB 2022 projection (HDX COD-PS)")
    # Only units whose 2011 lineage is exactly one 2011 upazila, untouched: every
    # union of that upazila stayed home and nothing else landed there. The COD-PS
    # projection is a 2011-based cohort projection, so it shares the 2011 base
    # but nothing from the 2022 census: a unit far off it is either a wrong
    # join or real growth the projection did not foresee.
    lin = pd.read_csv(DATA / "unions_2011_to_v03.csv", dtype={"home": str})
    g = lin.groupby("adm3_pcode")
    pure = [c for c, t in g if (t.how == "home").all() and t.home.nunique() == 1
            and not ((lin.home == c) & (lin.adm3_pcode != c)).any()]
    nso = lin.drop_duplicates("home").set_index("home")
    ps = pd.read_excel(DATA / "raw/bgd_admpop_2022.xlsx", sheet_name="bgd_admpop_adm3_2022")
    ps = ps.set_index("ADM3_PCODE").T_TL
    gm2nso = gpd.read_file(REPO / "religiondots/data/raw/bd/Bangladesh.gdb",
                           layer="BD_GEOG_ADM3_2011_uscb_202107", ignore_geometry=True)
    home2ps = {"BD" + n[:4] + n[4:].zfill(4): "BD" + n for n in gm2nso.NSO_CODE}
    t = w3.loc[pure, ["name", "zila", 2011, 2022]].copy()
    t["uscb_2022"] = [ps.get(home2ps.get(c)) for c in t.index]
    t = t.dropna(subset=["uscb_2022"])
    t["census_vs_uscb"] = pct(t[2022], t.uscb_2022)
    print(f"  {len(pure)} units with a one-to-one 2011 lineage, {len(t)} found in COD-PS")
    print(f"  census within 10% of the projection: {(t.census_vs_uscb.abs() <= 10).mean() * 100:.1f}%; "
          f"within 20%: {(t.census_vs_uscb.abs() <= 20).mean() * 100:.1f}%")
    print(t.sort_values("census_vs_uscb").head(6).round(1).to_string())
    print(t.sort_values("census_vs_uscb").tail(6).round(1).to_string())

    level4(pop, v3, wide, hexes)

    print("== 5. coverage")
    for lv, g, col in ((1, v1, "adm1_pcode"), (2, v2, "adm2_pcode"), (3, v3, "adm3_pcode")):
        codes = set(g[col])
        rows = set(wide[lv].index)
        both = wide[lv].dropna()
        print(f"  adm{lv}: {len(codes)} units, {len(rows)} codes in csv, "
              f"{len(codes - rows)} units without data, {len(rows - codes)} rows without a unit, "
              f"{len(both)} with both years")


def level4(pop, v3, wide, hexes):
    """Level 4 (level4.py): sums, published totals, an independent print of
    one district's unions, Kontur, and the 2022/2011 ratios."""
    import fitz
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import level4_tables

    print("== 6. level 4 (unions, paurashavas, city corporation wards/thanas)")
    w4 = pop[pop.level == 4].pivot(index="code", columns="year", values="pop")
    g4 = gpd.read_file(DATA / "level4.gpkg").set_index("code")
    assert set(g4.index) == set(w4.index), "level4.gpkg and population.csv disagree"
    assert w4.notna().all().all(), "a level-4 unit lacks a year"
    print(f"  {len(w4)} units, every one with 2011 and 2022")
    for y in (2011, 2022):
        s = w4[y].groupby(g4.adm3_pcode).sum()
        bad = (s - wide[3][y].reindex(s.index)).abs()
        print(f"  {y}: sums to level 3 exactly in {(bad == 0).sum()} of {len(s)} level-3 units")
        assert (bad == 0).all()

    t = pd.DataFrame(level4_tables.parse())
    un = t[(t.table == "U01") & ~t.name.isin(["", "TOTAL"])]
    print(f"  2022 published totals: unions {un['pop'].sum():,} (Union Statistics 'Union Total' "
          f"{int(t[(t.table == 'U01') & (t.name == 'TOTAL')]['pop'].iloc[0]):,}); paurashavas "
          f"{t[(t.table == 'P34') & (t.name != 'TOTAL')]['pop'].sum():,} (P34 total "
          f"{int(t[(t.table == 'P34') & (t.name == 'TOTAL')]['pop'].iloc[0]):,})")

    # Thakurgaon's District Report (BBS 2024), Table 01, prints the same unions
    # again; an independent typesetting of the same count. Its Total includes
    # hijra, so it may be a few above. Kontur is checked against the raw 2011
    # union polygons too: it is as far off there (median 34%), so a poor
    # level-4 Kontur score is Kontur at this scale, not the fitted polygons.
    pdf = DATA / "raw/zila2022/thakurgaon.pdf"
    if pdf.exists():
        doc = fitz.open(pdf)
        lines = []
        for p in range(154, 158):
            lines += [l.strip() for l in doc[p].get_text().splitlines()]
        dr = {}
        for i, l in enumerate(lines):
            if l.endswith(" Union") and i + 2 < len(lines) and lines[i + 2].isdigit() \
                    and norm(l[:-6]) not in dr:  # Table 02 follows on page 157
                dr[norm(l[:-6])] = int(lines[i + 2])
        us = un[un.district == "Thakurgaon"].set_index(un[un.district == "Thakurgaon"].name.map(norm))["pop"]
        both = us.index.intersection(list(dr))
        d = pd.Series({k: dr[k] - us[k] for k in both})
        print(f"  Thakurgaon District Report vs Union Statistics: {len(both)} of {len(us)} unions "
              f"found; equal {(d == 0).sum()}, district report higher by 1-15 (hijra) "
              f"{((d > 0) & (d <= 15)).sum()}, other {((d < 0) | (d > 15)).sum()}")

    # Kontur 2023 hex centres per level-4 polygon
    pts = hexes.copy()
    pts["geometry"] = hexes.geometry.centroid
    pts = gpd.sjoin(pts.to_crs(4326), g4.reset_index()[["code", "geometry"]], predicate="within")
    k = pts.groupby("code").population.sum().reindex(w4.index).fillna(0)
    k = k * w4[2022].sum() / k.sum()
    kd = pct(k, w4[2022])
    print(f"  Kontur within 10% of the 2022 count: {(kd.abs() <= 10).mean() * 100:.1f}% of units; "
          f"within 20%: {(kd.abs() <= 20).mean() * 100:.1f}%; median |diff| {kd.abs().median():.1f}%")
    r = w4[2022] / w4[2011]
    print(f"  2022/2011 ratio quantiles: {r.quantile([.01, .05, .25, .5, .75, .95, .99]).round(2).to_dict()}")
    print(f"  units below 0.6: {(r < .6).sum()}, above 2: {(r > 2).sum()}")
    x = g4.join(w4).assign(ratio=r)[["name", "adm3_name", 2011, 2022, "ratio"]]
    print(x.sort_values("ratio").head(8).round(2).to_string())
    print(x.sort_values("ratio").tail(8).round(2).to_string())
    big = x[x[2022] > 500000].sort_values(2022)
    print("  units over 500,000 in 2022:")
    print(big.to_string())
    lin = pd.read_csv(DATA / "level4_lineage.csv")
    how = lin.how.fillna("").str.split("; ").explode().value_counts()
    print("  members by rule:", how.to_dict())


if __name__ == "__main__":
    main()
