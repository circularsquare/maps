"""Independent check: Kontur population hexagons (400 m, 2023-11-01) summed
inside each shipped rayon, against the bulletin's 1 Jan 2025 figure.

Kontur's national total runs ~2% above the bulletin, so each unit's ratio is
divided by the national ratio before it is judged; "within 10%" means the
normalised ratio is in 0.9-1.1. Hexes are assigned by centroid.

Reads religiondots' Kontur copy (read-only) and caches a decompressed .gpkg in
helper1m/data/kazakhstan/. Writes data/kazakhstan/kontur_check.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import gzip
import shutil

import geopandas as gpd
import pandas as pd

from common import DATA, REPO

SRC = REPO / "religiondots" / "data" / "raw" / "kz" / "kontur_population_KZ_20231101.gpkg.gz"
GPKG = DATA / "kontur_population_KZ_20231101.gpkg"
YEAR = 2025


def main():
    if not GPKG.exists():
        with gzip.open(SRC, "rb") as fi, GPKG.open("wb") as fo:
            shutil.copyfileobj(fi, fo)
    hexes = gpd.read_file(GPKG)
    hexes = hexes.to_crs("EPSG:4326")
    hexes["geometry"] = hexes.geometry.centroid
    adm2 = gpd.read_file(DATA / "boundaries" / "adm2.gpkg")
    j = gpd.sjoin(hexes[["population", "geometry"]], adm2[["code", "geometry"]],
                  how="left", predicate="within")
    j = j[~j.index.duplicated()]
    k = j.groupby("code").population.sum()
    dropped = j[j.code.isna()].population.sum()
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    p = pop[(pop.level == 2) & (pop.year == YEAR)].set_index("code")["pop"]
    df = adm2[["code", "name"]].set_index("code").join(p.rename("pop")).join(k.rename("kontur"))
    df["kontur"] = df.kontur.fillna(0)
    nat = df.kontur.sum() / df["pop"].sum()
    df["ratio"] = df.kontur / df["pop"] / nat
    within = ((df.ratio >= 0.9) & (df.ratio <= 1.1)).mean()
    within20 = ((df.ratio >= 0.8) & (df.ratio <= 1.25)).mean()
    print(f"Kontur {df.kontur.sum():,.0f} vs bulletin {YEAR} {df['pop'].sum():,} "
          f"(ratio {nat:.3f}); hexes outside every rayon: {dropped:,.0f}")
    print(f"units within 10% (normalised): {within:.1%}; within 0.8-1.25: {within20:.1%}")
    df = df.sort_values("ratio")
    print("lowest:\n" + df.head(12).to_string())
    print("highest:\n" + df.tail(12).to_string())
    df.to_csv(DATA / "kontur_check.csv")

    # Kontur smears city people into the rayons around them (religiondots'
    # kz_geo.md found Astana 0.64 and Akmola 1.54 against the 2021 census).
    # So pool every city with the rayons it touches, and judge the pools and
    # the rayons that touch no city separately.
    def is_city(code):
        return code[:2] in ("71", "75", "79") or int(code[2:4]) < 30
    a = adm2.set_index("code")
    cities = [c for c in a.index if is_city(c)]
    parent = {c: c for c in a.index}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    sindex = a.sindex
    for c in cities:
        for i in sindex.query(a.geometry[c], predicate="intersects"):
            other = a.index[i]
            if other != c:
                parent[find(other)] = find(c)
    df["pool"] = [find(c) for c in df.index]
    pools = df.groupby("pool")[["pop", "kontur"]].sum()
    sizes = df.groupby("pool").size()
    pools["ratio"] = pools.kontur / pools["pop"] / nat
    multi = pools[sizes > 1]
    lone = pools[sizes == 1]
    lone_rural = lone[[not is_city(c) for c in lone.index]]

    def share(s, lo, hi):
        return ((s >= lo) & (s <= hi)).mean()
    print(f"\n{len(multi)} pools of a city and the rayons round it: within 10% "
          f"{share(multi.ratio, 0.9, 1.1):.0%}, within 0.8-1.25 {share(multi.ratio, 0.8, 1.25):.0%}")
    print(f"{len(lone_rural)} rayons touching no city: within 10% "
          f"{share(lone_rural.ratio, 0.9, 1.1):.0%}, within 0.8-1.25 "
          f"{share(lone_rural.ratio, 0.8, 1.25):.0%}")
    med = lone_rural.ratio.median()
    rel = lone_rural.ratio / med
    print(f"  their median ratio is {med:.3f} (Kontur puts city people in the "
          f"countryside); against that median: within 10% {share(rel, 0.9, 1.1):.0%}, "
          f"within 0.8-1.25 {share(rel, 0.8, 1.25):.0%}")
    names = df.name.to_dict()
    show = multi.assign(units=[", ".join(names[c] for c in df.index[df.pool == p]) for p in multi.index])
    print(show.sort_values("ratio").to_string())
    print(lone_rural.assign(name=[names[c] for c in lone_rural.index]).sort_values("ratio").to_string())


if __name__ == "__main__":
    main()
