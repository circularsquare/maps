"""Checks for the Kyrgyzstan build (run after fetch.py and prep_boundaries.py).

1. National and oblast totals against the NSC's own rows, every year.
2. Level 3 against an independent count: Kontur Population (2023-11-01, 400 m
   H3 hexagons from GHSL and Meta settlement data), hex centres summed in each
   unit's polygon and scaled to the 2026 national total.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys  # noqa: E402
import warnings  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crosswalk as cw  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "helper1m", "data", "kyrgyzstan")
KONTUR = os.path.join(REPO, "religiondots", "data", "raw", "kg", "kontur_population_KG_20231101.gpkg")


def main():
    pop = pd.read_csv(os.path.join(OUT, "population.csv"), dtype={"code": str})
    units = pd.read_csv(os.path.join(OUT, "units.csv"), dtype=str)
    d = cw.load_years()
    print("1. totals against the NSC rows")
    for y, t in d.items():
        off = t.set_index("code")["pop"]
        l1 = pop[(pop.level == 1) & (pop.year == y)].set_index("code")["pop"]
        print(f"  {y}: national {l1.sum():,} vs NSC {int(off['41700000000000']):,}")
        for c, v in l1.items():
            o = off.get(c)
            flag = "" if o == v else f"   (NSC row {o:,.0f}: {'old boundary' if y == 2024 else 'DIFFERS'})"
            if y == 2026 or flag:
                print(f"     {c} {units[units.level == '1'].set_index('code').name[c]:<16} {v:>10,}{flag}")

    print("2. level 3 against Kontur 2023")
    g3 = gpd.read_file(os.path.join(OUT, "boundaries", "adm3.gpkg"))
    k = gpd.read_file(KONTUR).to_crs("EPSG:4326")
    k["geometry"] = k.geometry.centroid
    j = gpd.sjoin(k[["population", "geometry"]], g3[["code", "geometry"]], predicate="within")
    ks = j.groupby("code").population.sum()
    print(f"  Kontur total {k.population.sum():,.0f}; inside level-3 polygons {ks.sum():,.0f}")
    p26 = pop[(pop.level == 3) & (pop.year == 2026)].set_index("code")["pop"]
    df = pd.DataFrame({"nsc": p26, "kontur": ks}).fillna(0)
    df = df[df.nsc > 0]
    df["kontur"] *= df.nsc.sum() / df.kontur.sum()
    df["ratio"] = df.kontur / df.nsc
    df["name"] = units[units.level == "3"].set_index("code").name.reindex(df.index)
    w10 = (df.ratio.sub(1).abs() <= 0.10)
    w25 = (df.ratio.sub(1).abs() <= 0.25)
    print(f"  {len(df)} populated units: {w10.mean():.0%} within 10%, {w25.mean():.0%} within 25%; "
          f"by people {df.nsc[w10].sum() / df.nsc.sum():.0%} / {df.nsc[w25].sum() / df.nsc.sum():.0%}")
    big = df[df.nsc >= 20000]
    print(f"  units of 20,000+: {len(big)}, {(big.ratio.sub(1).abs() <= 0.10).mean():.0%} within 10%, "
          f"{(big.ratio.sub(1).abs() <= 0.25).mean():.0%} within 25%")
    df["absdiff"] = (df.kontur - df.nsc).abs()
    print("  largest gaps (people):")
    for c, r in df.sort_values("absdiff", ascending=False).head(15).iterrows():
        print(f"    {c} {r['name']:<34} NSC {r.nsc:>9,.0f}  Kontur {r.kontur:>9,.0f}  x{r.ratio:.2f}")
    # same comparison one level up, where the regrouping hardly matters
    u3 = units[units.level == "3"].set_index("code")
    k2 = ks.groupby(ks.index.map(u3.adm2)).sum()
    p2 = pop[(pop.level == 2) & (pop.year == 2026)].set_index("code")["pop"]
    r2 = (k2.reindex(p2.index).fillna(0) * p2.sum() / k2.sum()) / p2
    print(f"  level 2 ({len(r2)} units): {(r2.sub(1).abs() <= 0.10).mean():.0%} within 10%, "
          f"{(r2.sub(1).abs() <= 0.25).mean():.0%} within 25%")
    land = pop[(pop.level == 3) & (pop.year == 2026) & (pop["pop"] == 0)].code
    print(f"  Kontur people in the {len(land)} zero units (land): {ks.reindex(land).fillna(0).sum():,.0f}")
    df.to_csv(os.path.join(OUT, "check_kontur.csv"), encoding="utf-8")


if __name__ == "__main__":
    main()
