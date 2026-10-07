"""Myanmar placement layer: Kontur 400 m hexes keyed to the 325 townships GAD's ethnicity table covers.

    python sources/mm_geo.py      -> data/geo/mm/mm_hexes.gpkg  (unit, pop)

Units are USCB's ADM3 layer from the same HDX geodatabase as the table (sources/mm_gad.py), which
is MIMU's 2019 township boundaries (330 townships, the 2014 census set). The table and the
polygons share USCB's GEO_MATCH code (MMR_ss_dd_tt), so the join is on that code, asserted 1:1
with the five Wa SAD / Mongla townships, which have no ethnicity table, left out by name. Their
hexes fall outside every unit and are not drawn. religiondots draws Myanmar on 15 states, so its
layer is not reused. The witness for the join is the Kontur-against-GAD correlation per township
and the shuffled control hex_layer prints.
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from _grid import hex_layer  # noqa: E402

GDB = HERE / "data" / "raw" / "mm" / "burma.gdb.zip"
LAYER = "MM_GEOG_ADM3_2014_uscb_202003"
NO_TABLE = {"MONGMAO", "PANGWAUN", "PANGSANG", "NARPHAN", "MONGLA"}


def main():
    g = gpd.read_file(f"zip://{GDB.as_posix()}!Burma.gdb", layer=LAYER)
    assert len(g) == 330 and g["GEO_MATCH"].is_unique, f"{len(g)} ADM3 features"
    assert not g.geometry.is_empty.any() and g.geometry.notna().all(), "empty township geometry"
    df = pd.read_csv(HERE / "data" / "normalized" / "mm.csv")
    census = df.groupby("geo_id")["count"].sum().astype(int).to_dict()
    left = set(g["GEO_MATCH"]) - set(census)
    assert set(census) <= set(g["GEO_MATCH"]), sorted(set(census) - set(g["GEO_MATCH"]))
    assert set(g.loc[g["GEO_MATCH"].isin(left), "ADM3_NAME"]) == NO_TABLE, sorted(left)
    print(f"  {len(census)} townships joined on GEO_MATCH; left out (no table): "
          f"{sorted(g.loc[g['GEO_MATCH'].isin(left), 'ADM3_NAME'])}")
    units = g[g["GEO_MATCH"].isin(set(census))].rename(columns={"GEO_MATCH": "unit"})
    layer = hex_layer("mm", units[["unit", "geometry"]], census=census)

    # Downtown Yangon townships smaller than a hex (Pabedan, Seikkan, Pazundaung) get no hex
    # centroid. Their own polygon is appended, with GAD's count as its weight (playbook: "A unit
    # missing from the place layer is not drawn on its polygon").
    have = set(layer.loc[layer["pop"] > 0, "unit"])
    miss = units[~units["unit"].isin(have)][["unit", "geometry"]].copy()
    if len(miss):
        miss["pop"] = miss["unit"].map(census).astype(float)
        print(f"  appended own polygons for {len(miss)} units: "
              f"{sorted(g.set_index('GEO_MATCH').loc[miss['unit'], 'ADM3_NAME'])}")
        layer = pd.concat([layer[layer["unit"].isin(have)], miss.to_crs(4326)], ignore_index=True)
        layer = gpd.GeoDataFrame(layer, geometry="geometry", crs=4326)
    assert set(layer["unit"]) == set(census), "a counted township has no placement"
    out = HERE / "data" / "geo" / "mm" / "mm_hexes.gpkg"
    layer.to_file(out, layer="hexes", driver="GPKG")

    import math
    import random
    per = layer.groupby("unit")["pop"].sum()
    u = sorted(census)
    lc = [math.log(census[k]) for k in u]
    lk = [math.log(per[k]) for k in u]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation Kontur against GAD per township r = {r:.3f}, best of 500 "
          f"shuffles {best:.3f}")
    assert r > 0.8 and r > best, "the join is not carrying information"
    print(f"  wrote {out} ({len(layer):,} features)")


if __name__ == "__main__":
    main()
