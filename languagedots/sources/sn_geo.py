"""Senegal: religiondots' Kontur hexes re-keyed to the 14 régions of 2023 -> data/geo/sn/sn_hexes.gpkg.

    python sources/sn_geo.py

religiondots draws Senegal at its 1988 units (sources/sn_geo.py there): nine 1988 régions and
Diourbel's three départements, built from COD-AB Senegal v02 by pcode, with Kontur 2023-11 400 m
hexes keyed to them (coastal centroids just offshore snapped to the nearest unit, within 500 m).
RGPH-5's language tables are by today's 14 régions. Every 1988 unit is a union of today's
régions, or a part of one (Diourbel's départements), so most hexes carry their région already:

    1988 unit           2023 région(s)
    Dakar ... Fatick    the same région (6 units)
    Bambey, Diourbel,   Diourbel
    Mbacké
    Saint-Louis         Saint-Louis + Matam        <- split by COD-AB admin1 polygon
    Tambacounda         Tambacounda + Kédougou     <- split
    Kaolack             Kaolack + Kaffrine         <- split
    Kolda               Kolda + Sédhiou            <- split

In the four split units each hex goes to the région polygon (of the two) its centroid falls in;
a centroid in neither (a coastal hex religiondots snapped, or a sliver) goes to the nearer of
the two. Everything is read from religiondots, read-only; nothing is downloaded.

CHECKS: the 14 COD pcodes and names; every hex keyed; population kept exactly; per-région
Kontur against the census (RGPH-5 Tableau I-21 resident population, and the language table's
3+ totals), printed, with a shuffled-join control on the log correlation.
"""
import math
import random
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

RD_HEX = RD / "data" / "geo" / "sn" / "sn_hexes.gpkg"
RD_ADM1 = RD / "data" / "raw" / "sn" / "shp" / "sen_admin1.shp"
OUT = HERE / "data" / "geo" / "sn" / "sn_hexes.gpkg"

ADM1_NAMES = {
    "SN01": "Dakar", "SN02": "Diourbel", "SN03": "Fatick", "SN04": "Kaffrine", "SN05": "Kaolack",
    "SN06": "Kédougou", "SN07": "Kolda", "SN08": "Louga", "SN09": "Matam", "SN10": "Saint-Louis",
    "SN11": "Sédhiou", "SN12": "Tambacounda", "SN13": "Thiès", "SN14": "Ziguinchor",
}
DIRECT = {"Dakar": "Dakar", "Ziguinchor": "Ziguinchor", "Thiès": "Thiès", "Louga": "Louga",
          "Fatick": "Fatick", "Bambey": "Diourbel", "Diourbel": "Diourbel", "Mbacké": "Diourbel"}
SPLIT = {"Saint-Louis": ("SN10", "SN09"), "Tambacounda": ("SN12", "SN06"),
         "Kaolack": ("SN05", "SN04"), "Kolda": ("SN07", "SN11")}

# RGPH-5 chapter 1, Tableau I-21: résident population by région (all ages)
POP_2023 = {
    "Dakar": 4_004_426, "Ziguinchor": 617_567, "Diourbel": 2_080_333, "Saint-Louis": 1_202_441,
    "Tambacounda": 987_152, "Kaolack": 1_336_720, "Thiès": 2_463_677, "Louga": 1_125_908,
    "Fatick": 906_918, "Kolda": 914_798, "Matam": 831_630, "Kaffrine": 820_405,
    "Kédougou": 245_147, "Sédhiou": 589_266,
}
assert sum(POP_2023.values()) == 18_126_388  # the table prints 18,126,390: rounding, 2 people


def main():
    hexes = gpd.read_file(RD_HEX)
    a1 = gpd.read_file(RD_ADM1)
    got = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    if got != ADM1_NAMES:
        raise SystemExit(f"COD admin1 pcodes/names changed: {got}")
    if set(hexes["unit"]) != set(DIRECT) | set(SPLIT):
        raise SystemExit(f"religiondots' sn_hexes units changed: {sorted(set(hexes['unit']))}")
    n0, p0 = len(hexes), hexes["pop"].sum()

    region = hexes["unit"].map(DIRECT)
    m = gpd.GeoDataFrame(geometry=a1.geometry, crs=a1.crs).to_crs(3857)
    m["pcode"] = a1["adm1_pcode"].to_numpy()
    cent = gpd.GeoDataFrame(geometry=hexes.to_crs(3857).geometry.centroid, crs=3857)
    fallback = 0
    for unit, pcs in SPLIT.items():
        idx = hexes.index[hexes["unit"] == unit]
        polys = m[m["pcode"].isin(pcs)]
        j = gpd.sjoin(cent.loc[idx], polys[["pcode", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        miss = j["pcode"].isna()
        if miss.any():
            near = gpd.sjoin_nearest(cent.loc[j.index[miss]], polys[["pcode", "geometry"]], how="left")
            near = near[~near.index.duplicated(keep="first")]
            j.loc[miss, "pcode"] = near["pcode"]
            fallback += int(miss.sum())
        region.loc[idx] = j["pcode"].map(ADM1_NAMES)
        for pc in pcs:
            sel = j["pcode"] == pc
            print(f"  {unit:12s} -> {ADM1_NAMES[pc]:12s} {int(sel.sum()):>6,} hexes "
                  f"{hexes.loc[j.index[sel], 'pop'].sum():>12,.0f} people")
    if region.isna().any():
        raise SystemExit(f"{int(region.isna().sum())} hexes with no région")
    print(f"  {fallback} hexes in a split unit outside both régions' polygons, given the nearer")

    out = gpd.GeoDataFrame({"unit": region.to_numpy(), "pop": hexes["pop"].to_numpy()},
                           geometry=hexes.geometry.to_numpy(), crs=hexes.crs)
    assert len(out) == n0 and abs(out["pop"].sum() - p0) < 1
    per = out.groupby("unit")["pop"].sum()
    if set(per.index) != set(POP_2023):
        raise SystemExit(f"régions in the layer: {sorted(per.index)}")

    norm = HERE / "data" / "normalized" / "sn.csv"
    t3 = None
    if norm.exists():
        df = pd.read_csv(norm)
        t3 = df[df["geo_level"] == "region"].groupby("geo_id")["count"].sum()
    ratio = per.sum() / sum(POP_2023.values())
    print(f"  Kontur {per.sum():,.0f} / census {sum(POP_2023.values()):,} = {ratio:.3f}")
    print(f"  {'région':12s} {'Kontur':>11s} {'census':>11s} {'norm':>6s} {'3+ table':>11s}")
    for u in sorted(POP_2023, key=POP_2023.get, reverse=True):
        r = per[u] / POP_2023[u] / ratio
        extra = f" {int(t3[u]):>11,}" if t3 is not None else ""
        print(f"  {u:12s} {per[u]:>11,.0f} {POP_2023[u]:>11,} {r:>6.2f}{extra}")
        if not (1 / 1.6 <= r <= 1.6):
            raise SystemExit(f"{u}: Kontur against the census {r:.2f}, outside 1.6x")

    lc = [math.log(POP_2023[u]) for u in POP_2023]
    lk = [math.log(per[u]) for u in POP_2023]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("the join is not carrying information")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
