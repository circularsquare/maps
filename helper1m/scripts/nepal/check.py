"""Checks for helper1m Nepal, run after fetch.py. Read-only.

1. National and province totals per year against NSO's published figures.
2. Palika level against an independent source: Kontur Population 2023-11-01
   (400 m hexes), already keyed to COD palika codes by religiondots
   (religiondots/data/geo/np/np_hexes.gpkg, read only). Compared with NSO's
   projection for 2023, the year of the Kontur release.
3. Every boundary unit has a population and every population row has a unit.
4. The spread of 2021->2026 palika growth in the projection.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
RAW = HELPER / "data" / "nepal" / "raw"
POP = HELPER / "data" / "nepal" / "population.csv"
HEXES = REPO / "religiondots" / "data" / "geo" / "np" / "np_hexes.gpkg"
BOUNDS = {1: (REPO / "data/asia1m/nepal/npl_admin1.shp", "adm1_pcode", "adm1_name"),
          2: (REPO / "data/asia1m/nepal/npl_admin2.shp", "adm2_pcode", "adm2_name"),
          3: (HELPER / "data/nepal/boundaries/adm3.gpkg", "adm3_pcode", "adm3_name")}

sys.path.insert(0, str(Path(__file__).parent))
import fetch  # noqa: E402


def main():
    pop = pd.read_csv(POP, dtype={"code": str})
    names = {}
    units, districts, provinces = fetch.read_census()
    fetch.spread_institutional(units, districts)
    fetch.join_cod(units)
    # province rows as printed in the census workbook, keyed by COD code
    p1_of = {u["prov_n"]: u["cod"][3] for u in units}
    CENSUS_PROV = {p1_of[p]: v["pop"] for p, v in provinces.items()}
    print("\n== 1. totals")
    for y in sorted(pop["year"].unique()):
        n = pop[(pop.level == 1) & (pop.year == y)]["pop"].sum()
        print(f"  {y}: national {n:,}")
    p21 = pop[(pop.level == 1) & (pop.year == 2021)].set_index("code")["pop"]
    bad = {c: (int(p21[c]), v) for c, v in CENSUS_PROV.items() if p21[c] != v}
    print(f"  2021 provinces vs census province rows: {7 - len(bad)}/7 exact {bad or ''}")
    for c in sorted(CENSUS_PROV):
        print(f"    {c} {CENSUS_PROV[c]:>10,}")
    pp = {}
    for p in range(1, 8):
        rows = json.loads((RAW / "projection" / f"p_{p}.json").read_text())
        pp[f"NP0{p}"] = {r["year"]: r["total"] for r in rows}
    for y in (2026, 2031):
        got = pop[(pop.level == 1) & (pop.year == y)].set_index("code")["pop"]
        bad = [c for c in pp if got[c] != pp[c][y]]
        print(f"  {y} provinces vs NSO province projection rows: {7 - len(bad)}/7 exact")

    print("\n== 3. boundary <-> population coverage")
    for lv, (path, code_col, name_col) in BOUNDS.items():
        b = pyogrio.read_dataframe(path, read_geometry=False)
        codes = set(b[code_col])
        names.update(dict(zip(b[code_col], b[name_col])))
        have = set(pop[pop.level == lv]["code"])
        yrs = pop[pop.level == lv].groupby("code")["year"].nunique()
        print(f"  adm{lv}: {len(codes)} units, {len(have)} coded rows; "
              f"no population {len(codes - have)}, no unit {len(have - codes)}, "
              f"units with <3 years {int((yrs < 3).sum())}")

    print("\n== 2. palika vs Kontur 2023 (independent grid)")
    hx = pyogrio.read_dataframe(HEXES, read_geometry=False, columns=["unit", "pop"])
    k = hx.groupby("unit")["pop"].sum()
    proj23 = {}
    # 2023 for each palika: the projection cache is keyed by NSO's numbering
    seq = sorted(districts)
    for u in units:
        dn = seq.index((u["prov_n"], u["dist_n"])) + 1
        rows = json.loads((RAW / "projection" / f"m_{dn:02d}_{u['local_n']:02d}.json").read_text())
        proj23[u["cod"][1]] = {r["year"]: r["total"] for r in rows}[2023]
    df = pd.DataFrame({"nso": pd.Series(proj23), "kontur": k}).dropna()
    df["name"] = [names.get(c, c) for c in df.index]
    nat = df["kontur"].sum() / df["nso"].sum()
    df["ratio"] = df["kontur"] / df["nso"]
    df["rel"] = df["ratio"] / nat
    print(f"  {len(df)} palikas compared; Kontur total {df['kontur'].sum():,.0f} vs NSO 2023 "
          f"{df['nso'].sum():,} (x{nat:.3f})")
    for lab, col in (("raw", "ratio"), ("after the national factor", "rel")):
        r = df[col]
        print(f"  {lab}: within 10% {((r - 1).abs() <= 0.10).mean():.1%}, within 25% "
              f"{((r - 1).abs() <= 0.25).mean():.1%}, median {r.median():.2f}, "
              f"quartiles {r.quantile(.25):.2f}-{r.quantile(.75):.2f}")
    print("  furthest out (Kontur/NSO after the national factor):")
    w = df.assign(dev=(df["rel"].apply(lambda v: max(v, 1 / v)))).sort_values("dev", ascending=False)
    for c, r in w.head(14).iterrows():
        print(f"    {c} {r['name']:<24} NSO {int(r['nso']):>8,}  Kontur {int(r['kontur']):>8,}  "
              f"x{r['rel']:.2f}")
    # pooled with district: is the gap a boundary effect inside the district?
    df["d"] = df.index.str[:6]
    dd = df.groupby("d")[["nso", "kontur"]].sum()
    dr = dd["kontur"] / dd["nso"] / nat
    print(f"  districts (77): within 10% {((dr - 1).abs() <= 0.10).mean():.1%}, "
          f"within 25% {((dr - 1).abs() <= 0.25).mean():.1%}")

    print("\n== 4. projected palika growth 2021->2026 (NSO projection years only)")
    p3 = pop[pop.level == 3].pivot(index="code", columns="year", values="pop")
    g = p3[2031] / p3[2026] - 1
    print(f"  2026->2031: median {g.median():+.1%}, p5 {g.quantile(.05):+.1%}, "
          f"p95 {g.quantile(.95):+.1%}, min {g.min():+.1%} ({names[g.idxmin()]}), "
          f"max {g.max():+.1%} ({names[g.idxmax()]})")
    g2 = p3[2026] / p3[2021] - 1
    print(f"  census 2021 -> projection 2026: median {g2.median():+.1%}, "
          f"min {g2.min():+.1%} ({names[g2.idxmin()]}), max {g2.max():+.1%} "
          f"({names[g2.idxmax()]})")


if __name__ == "__main__":
    main()
