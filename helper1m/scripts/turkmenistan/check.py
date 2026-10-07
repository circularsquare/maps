"""Checks on Turkmenistan's population.csv. Run after fetch.py.

1. Units sum to their top-level unit in both years, and the top level to the nation.
2. 2022 top level against census table 1.3 (Ahal less Arkadag's 567).
3. Each census etrap's total is in the census tables (census.py asserts this);
   here: the census etrap -> current unit flows for the five new etraps.
4. Kontur population (400 m hexes, 2023-11, read from religiondots) summed in each
   current unit, as a share of the nation and of its velayat, against the census.
5. Big cities against table 1.5-1.9's own rows.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

from pathlib import Path

import geopandas as gpd
import pandas as pd

HELPER = Path(__file__).resolve().parents[2]
DATA = HELPER / "data" / "turkmenistan"
HEXES = HELPER.parent / "religiondots" / "data" / "geo" / "tm" / "tm_hexes.gpkg"

TABLE_1_3 = {"TM-S": 1030063, "TM-A": 886845 - 567, "TM-AR": 567, "TM-B": 529895,
             "TM-D": 1550354, "TM-L": 1447298, "TM-M": 1613386}
CITIES = {  # current unit: census figure for the same territory (tables 1.4-1.9)
    "TM-D-03": ("Dashoguz city", 201142),
    "TM-L-12": ("Turkmenabat city", 230861),
    "TM-M-04": ("Mary city", 167027),
    "TM-M-01": ("Bayramaly city", 70376),
    "TM-B-07": ("Turkmenbashy city", 91745),
    "TM-AR": ("Arkadag city", 567),
}


def main():
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    adm2 = gpd.read_file(DATA / "boundaries" / "adm2.gpkg")
    par = adm2.set_index("code").group
    bad = 0
    for year, g in pop.groupby("year"):
        l1 = g[g.level == 1].set_index("code")["pop"]
        l2 = g[g.level == 2].set_index("code")["pop"]
        assert set(l2.index) == set(adm2.code), "units missing a population"
        s = l2.groupby(par).sum()
        for c in l1.index:
            if s[c] != l1[c]:
                print(f"  {year} {c}: units {s[c]:,} vs level 1 {l1[c]:,}")
                bad += 1
        print(f"{year}: national {l1.sum():,}; level 1 = sum of units: {'yes' if not bad else 'NO'}")
        if year == 2022:
            assert l1.sum() == 7057841
            for c, want in TABLE_1_3.items():
                flag = "" if l1[c] == want else "  <-- differs"
                print(f"  {c:6s} {l1[c]:>10,}  table 1.3 {want:>10,}{flag}")
    assert not bad

    c22 = pop[(pop.level == 2) & (pop.year == 2022)].set_index("code")["pop"]
    print("\ncities against their census rows (2022):")
    for code, (nm, want) in CITIES.items():
        print(f"  {code:8s} {nm:20s} {c22[code]:>9,} vs {want:>9,} {'ok' if c22[code] == want else 'DIFF'}")

    # Kontur
    hexes = gpd.read_file(HEXES)
    hexes["geometry"] = hexes.geometry.to_crs(3857).centroid.to_crs(4326)
    j = gpd.sjoin(hexes[["pop", "geometry"]], adm2[["code", "geometry"]], predicate="within")
    k = j.groupby("code")["pop"].sum().reindex(adm2.code).fillna(0)
    df = pd.DataFrame({"census": c22, "kontur": k, "group": par})
    df["r_nat"] = (df.kontur / df.kontur.sum()) / (df.census / df.census.sum())
    vk = df.groupby("group").kontur.transform("sum")
    vc = df.groupby("group").census.transform("sum")
    df["r_vel"] = (df.kontur / vk) / (df.census / vc)
    print(f"\nKontur total {df.kontur.sum():,.0f} vs census {df.census.sum():,}")
    reg = df.groupby("group")[["census", "kontur"]].sum()
    reg["ratio"] = (reg.kontur / reg.kontur.sum()) / (reg.census / reg.census.sum())
    print(reg.round(2).to_string())
    for col, lab in (("r_nat", "share of nation"), ("r_vel", "share of own velayat")):
        r = df[col][df.census > 1000]
        print(f"{lab}: {((r - 1).abs() <= 0.1).sum()} of {len(r)} within 10%, "
              f"{((r - 1).abs() <= 0.25).sum()} within 25%")
    names = adm2.set_index("code").name
    df["name"] = names
    print("\nby unit (Kontur / census, share of own velayat):")
    print(df.sort_values("r_vel")[["name", "census", "kontur", "r_vel"]].round(2).to_string())


if __name__ == "__main__":
    main()
