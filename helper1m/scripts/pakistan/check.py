"""Checks on the built Pakistan data: joins both ways, and Kontur 2023-11 against the census.

    C:\\Python39\\python.exe helper1m\\scripts\\pakistan\\check.py

Kontur's population grid (H3 hexes, 2023-11-01) is summed on hex centroids inside each unit and
compared with the 2023 census count. It is an independent spread of people, not a second count:
in Pakistan it runs well below the census in Balochistan and dense Karachi/Quetta, so a
national-level scale is taken out first (ratio / national ratio) and the per-province medians are
printed beside it.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kontur

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "helper1m", "data", "pakistan")


def main():
    pop = pd.read_csv(os.path.join(DATA, "population.csv"), dtype={"code": str})
    pts = kontur.centroids(DATA)
    print(f"Kontur 2023-11: {pts['pop'].sum():,.0f} people in {len(pts):,} hexes")
    for level in (1, 2, 3):
        g = gpd.read_file(os.path.join(DATA, "boundaries", f"adm{level}.gpkg"))
        p = pop[(pop.level == level)]
        codes_b, codes_p = set(g.code), set(p.code)
        print(f"\nadm{level}: {len(g)} polygons; population rows for {len(codes_p)} codes; "
              f"polygons without data {sorted(codes_b - codes_p)[:5]}, data without polygon "
              f"{sorted(codes_p - codes_b)[:5]}")
        two = p.groupby("code").year.nunique()
        print(f"  units with both 2017 and 2023: {(two == 2).sum()} of {len(two)}")
        j = gpd.sjoin(pts, g[["code", "geometry"]], predicate="within")
        k = j.groupby("code")["pop"].sum()
        c = p[p.year == 2023].set_index("code")["pop"]
        df = pd.DataFrame({"census": c, "kontur": k}).fillna(0)
        df = df.join(g.set_index("code")[["name", "group"]])
        nat = df.kontur.sum() / df.census.sum()
        df["ratio"] = df.kontur / df.census
        df["rel"] = df.ratio / nat
        within = (df.rel.sub(1).abs() <= 0.10).mean()
        within_raw = (df.ratio.sub(1).abs() <= 0.10).mean()
        print(f"  Kontur / census 2023: national {nat:.3f}; units within 10% raw {100 * within_raw:.0f}%, "
              f"after the national scale {100 * within:.0f}%; within 25% after scale "
              f"{100 * (df.rel.sub(1).abs() <= 0.25).mean():.0f}%")
        if level >= 2:
            med = df.groupby("group").ratio.median()
            print("  median ratio by province: " + ", ".join(f"{k} {v:.2f}" for k, v in med.items()))
            # within 10% of the unit's own province median
            df["relp"] = df.ratio / df.group.map(med)
            print(f"  within 10% of own province's median ratio: "
                  f"{100 * (df.relp.sub(1).abs() <= 0.10).mean():.0f}%, within 25%: "
                  f"{100 * (df.relp.sub(1).abs() <= 0.25).mean():.0f}%")
            big = df[df.census >= 500_000].sort_values("relp")
            print("  units >= 500k furthest below their province's ratio:")
            for code, r in big.head(6).iterrows():
                print(f"     {r.relp:5.2f}  {r['name'][:50]:50s} census {int(r.census):>10,} kontur {r.kontur:>10,.0f}")
            print("  ... furthest above:")
            for code, r in big.tail(6).iterrows():
                print(f"     {r.relp:5.2f}  {r['name'][:50]:50s} census {int(r.census):>10,} kontur {r.kontur:>10,.0f}")
        else:
            for code, r in df.iterrows():
                print(f"     {r['name']:20s} census {int(r.census):>12,} kontur {r.kontur:>12,.0f}  {r.ratio:.2f}")
        df.to_csv(os.path.join(DATA, f"check_kontur_adm{level}.csv"))


if __name__ == "__main__":
    main()
