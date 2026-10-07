"""Estonia: placement layer for the 127 parts RL21434 publishes (sources/ee_census.py).

    python sources/ee_geo.py     -> data/geo/ee/ee_units.gpkg, data/geo/ee/ee_hexes.gpkg

Reads Maa-amet's municipality and settlement-unit shapefiles (EHAK, 2024-12-01) from
religiondots' data/geo/ee/ READ-ONLY (its sources/ee_geo.py fetched them). Builds one polygon
per drawn part:

  a municipality with no parts       the municipality              OKOOD = code[4:8]
  a town "as a settlement unit"      the settlement unit (linn)    AKOOD = code[8:12]
  a city district (Tallinn 8,        the linnaosa                  AKOOD = code[8:12]
    Kohtla-Järve 5)
  "<municipality>, excl. <towns>"    the municipality less those towns (a bare serial code
                                     such as "4"; the parent is read from the census tree)

THE KEYS. Four municipality codes were retired between the census (2021) and the boundary file
(religiondots' sources/ee_geo.md §4): 0142->0145 Antsla, 0514->0515 Narva-Jõesuu,
0735->0736 Sillamäe, 0855->0857 Valga. They are re-joined by name, only where one candidate is
left on each side, and witnessed: a re-joined municipality's census settlement units must lie in
the new code. Every settlement-unit and district code must be found, inside its municipality.

CHECKS: 127 parts, each with area; each municipality's parts tile it (area within 0.5%); then
_grid.hex_layer's Kontur checks against the census total of every part.

KONTUR IS WRONG BETWEEN TOWNS HERE, AND IT DOES NOT MATTER (measured 2026-10-05). Kontur EE puts
900,302 people in Tallinn against the census's 437,817 (2.09x the national ratio) and 9,089 in
Tartu town against 95,190 (0.10x); Narva, Sillamäe, Rakvere, Viljandi, Pärnu and Keila all read
0.11-0.17. The shuffle control still passes (log r 0.810 against a best of 0.367). The dots'
counts per part are the census's; Kontur only spreads a part's dots over its own hexes, where
its footprint (which hexes are built up) is what counts, not its level. So the band failure is
printed and accepted. The 962 hexes outside every unit (15,127 people, 1.1%) have centroids at
sea or over the border (Kontur's extract runs past both); they are dropped, which only shifts
weight inland within a coastal part.
"""
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _grid import hex_layer  # noqa: E402

RD_EE = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "geo", "ee")
OMA = os.path.join(RD_EE, "omavalitsus", "omavalitsus.shp")
ASU = os.path.join(RD_EE, "asustusyksus", "asustusyksus.shp")
NORM = os.path.join(ROOT, "data", "normalized", "ee.csv")
OUT = os.path.join(ROOT, "data", "geo", "ee")
EXPECTED_PARTS = 127
EXPECTED_RETIRED = {"0142": "0145", "0514": "0515", "0735": "0736", "0855": "0857"}
TILE_TOL = 0.005


def bare(name):
    """Drop the unit-type words, English (census) or Estonian (Maa-amet)."""
    n = name.lower()
    for w in ("rural municipality", "city", "vald", "linn"):
        n = re.sub(rf"\b{w}\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def read_shp(path):
    g = gpd.read_file(path, encoding="cp1257")
    if not g["ONIMI"].str.contains("õ|ä|ö|ü").any():
        raise SystemExit(f"{path}: Estonian letters did not decode")
    return g


def main():
    df = pd.read_csv(NORM, dtype={"geo_id": str, "parent": str}, keep_default_na=False)
    tot = df[df["source_category"] == "Mother tongue total"]
    munis = tot[tot["geo_level"] == "municipality"][["geo_id", "geo_name"]].drop_duplicates()
    parts = tot[tot["geo_level"] == "part"][["geo_id", "geo_name", "parent", "count"]]
    if len(parts) != EXPECTED_PARTS:
        raise SystemExit(f"{len(parts)} parts, expected {EXPECTED_PARTS}")

    oma = read_shp(OMA)
    asu = read_shp(ASU)
    if len(oma) != 79:
        raise SystemExit(f"{len(oma)} municipalities in the boundary file, expected 79")

    # --- municipality code join, retired codes re-joined by name
    code_of = {g: g[4:8] for g in munis["geo_id"]}
    have = set(oma["OKOOD"])
    left_c = {g: n for g, n in zip(munis["geo_id"], munis["geo_name"]) if code_of[g] not in have}
    used = {code_of[g] for g in munis["geo_id"] if code_of[g] in have}
    left_p = oma[~oma["OKOOD"].isin(used)]
    alias = {}
    for g, n in left_c.items():
        cand = left_p[left_p["ONIMI"].map(bare) == bare(n)]
        if len(cand) != 1:
            raise SystemExit(f"{g} {n}: {len(cand)} name candidates among the unmatched polygons")
        alias[code_of[g]] = cand["OKOOD"].iloc[0]
    if alias != EXPECTED_RETIRED:
        raise SystemExit(f"retired-code aliases {alias}, expected {EXPECTED_RETIRED}")
    for g in munis["geo_id"]:
        code_of[g] = alias.get(code_of[g], code_of[g])
    if sorted(code_of.values()) != sorted(oma["OKOOD"]):
        raise SystemExit("municipality join is not one to one")
    print(f"  OK  79 municipalities joined, {len(alias)} by name after a code change: {alias}")

    oma_geom = dict(zip(oma["OKOOD"], oma.geometry))
    asu_by = {a: (o, geom, t) for a, o, geom, t in zip(asu["AKOOD"], asu["OKOOD"], asu.geometry, asu["TYYP"])}

    rows = []
    with_parts = {p for g, p in zip(parts["geo_id"], parts["parent"]) if g != p}
    for g, n, p, c in parts.itertuples(index=False):
        if g in set(munis["geo_id"]):          # a municipality drawn whole
            rows.append((g, n, p, code_of[g], oma_geom[code_of[g]], "municipality"))
    sub, retired_su = {}, {}
    for g, n, p, c in parts.itertuples(index=False):
        if g in set(munis["geo_id"]):
            continue
        ok_code = code_of[p]
        if len(g) == 14:
            ak = g[8:12]
            if ak not in asu_by:
                town = bare(n.replace("as a settlement unit", ""))
                cand = asu[(asu["OKOOD"] == ok_code) & (asu["ANIMI"].map(bare) == town)]
                if len(cand) != 1:
                    raise SystemExit(f"{g} {n}: settlement unit {ak} not in the boundary file, "
                                     f"and {len(cand)} name candidates in {ok_code}")
                print(f"  settlement unit {ak} {n!r} retired; re-joined by name to "
                      f"{cand['AKOOD'].iloc[0]} {cand['ANIMI'].iloc[0]}")
                retired_su[ak] = cand["AKOOD"].iloc[0]
                ak = retired_su[ak]
            o, geom, t = asu_by[ak]
            if o != ok_code:
                raise SystemExit(f"{g} {n}: settlement unit {ak} lies in {o}, census says {ok_code}")
            sub.setdefault(p, []).append(geom)
            rows.append((g, n, p, ok_code, geom, "district" if t == "6" else "town"))
    # witness for the re-joined codes: their towns sit inside the new code (checked above, since
    # code_of[p] is the aliased code); say which re-joins that witnessed
    witnessed = sorted(old for old, new in alias.items()
                       if any(code_of[p] == new for p in sub))
    print(f"  OK  every settlement unit and district found inside its municipality; "
          f"re-joins witnessed by a town: {witnessed}")
    for g, n, p, c in parts.itertuples(index=False):
        if g in set(munis["geo_id"]) or len(g) == 14:
            continue
        m = oma_geom[code_of[p]]
        towns = gpd.GeoSeries(sub.get(p, []), crs=oma.crs).union_all()
        rows.append((g, n, p, code_of[p], m.difference(towns), "remainder"))

    units = gpd.GeoDataFrame(pd.DataFrame(rows, columns=["unit", "name", "parent", "okood",
                                                         "geometry", "kind"]),
                             geometry="geometry", crs=oma.crs)
    if len(units) != EXPECTED_PARTS or units["unit"].duplicated().any():
        raise SystemExit(f"{len(units)} unit polygons for {EXPECTED_PARTS} parts")
    if (units.area <= 0).any():
        raise SystemExit(f"empty polygons: {units.loc[units.area <= 0, 'name'].tolist()}")
    print("  parts by kind:", units["kind"].value_counts().to_dict())

    # each municipality's parts tile it
    worst = 0.0
    for p in with_parts:
        a = units.loc[units["parent"] == p].area.sum()
        b = oma_geom[code_of[p]].area
        worst = max(worst, abs(a / b - 1))
        if abs(a / b - 1) > TILE_TOL:
            raise SystemExit(f"{p}: parts cover {a / b:.4f} of the municipality")
    print(f"  OK  the parts of all {len(with_parts)} split municipalities tile them "
          f"(worst area miss {100 * worst:.3f}%)")
    km2 = units.area / 1e6
    print(f"  part area km2: median {km2.median():.1f}, smallest "
          + ", ".join(f"{n} {a:.1f}" for n, a in sorted(zip(units['name'], km2), key=lambda x: x[1])[:5]))

    os.makedirs(OUT, exist_ok=True)
    units.to_crs(4326).to_file(os.path.join(OUT, "ee_units.gpkg"), layer="units", driver="GPKG")
    census = dict(zip(parts["geo_id"], parts["count"].astype(int)))
    hex_layer("ee", units[["unit", "geometry"]], census=census)


if __name__ == "__main__":
    main()
