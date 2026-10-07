"""Russia placement layer: religiondots' 3km Kontur hexes, each one marked urban or rural.

    python sources/ru_geo.py        -> data/geo/ru/ru_grid_3km.gpkg  (unit = "RU-TA/u", "RU-TA/r")

Rosstat prints native language by federal subject, split into urban and rural population
(sources/ru_census.py). The split matters on this map: nationally 37% of Tatar speakers are
rural against 24% of Russian speakers, and 62% of Chechen speakers. So each subject is
two units, and the hexes of a subject are divided between them.

THE HEXES are religiondots' `data/geo/ru/ru_grid_3km.gpkg` (read-only): Kontur H3 r6 population
hexes, each assigned and clipped to one of geoBoundaries' 83 subjects, with `unit` (ISO 3166-2)
and `pop`. religiondots' sources/ru_geo.py did the join and its checks (Kontur reproduces the
census to 0.984x nationally, within a factor of two in every subject).

CRIMEA AND SEVASTOPOL (UA-43, UA-40) are drawn from this census (Anita, 2026-10-05), but
geoBoundaries' Russia, and so religiondots' layer, stops at the 83 subjects. Their units are the
U.S. Census Bureau's 2001 polygons that Ukraine's layer uses (sources/ua_geo.py): the Autonomous
Republic's 25 raions and cities merge to UA-43 and Sevastopol's one polygon is UA-40. Their hexes
are Kontur UA's 400 m hexes as ua_geo.py assigned them over all 672 Ukrainian units and then cut
out (data/geo/ua/crimea_hexes.gpkg), so the line at Perekop and Chonhar is the one Ukraine's
mainland layer stops at. Asserted here: the merged subjects' areas equal their members' (no
overlap inside), they share no area with Ukraine's mainland units and touch them along the
isthmuses (no gap), no hex is in both Crimea's and Ukraine's mainland layer, and no religiondots
Russia hex reaches into Crimea's polygons across the Kerch Strait.

URBAN OR RURAL. Within a subject, hexes are ranked by population density (Kontur people per km² of
the clipped hex) and the densest are urban until their population reaches the census's urban share
of the subject (urban / (urban + rural) over everyone, native language stated or not). The hex
that crosses the line goes to whichever side leaves the share closer. A subject whose census has
rural people always gets at least one rural hex, and vice versa. Cities and towns are the dense
hexes, so this is a fair stand-in for Rosstat's administrative urban/rural line; it is not that
line, and a 36 km² hex holding a small town and its villages is all one or the other. Printed per
subject: the census's urban share and the share the hexes got.
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

SRC = RD_GEO / "ru" / "ru_grid_3km.gpkg"
NORM = ROOT / "data" / "normalized" / "ru.csv"
OUT = ROOT / "data" / "geo" / "ru" / "ru_grid_3km.gpkg"
UA_GDB = ("zip://" + str(ROOT / "data" / "raw" / "ua" / "ukraine.gdb.zip").replace("\\", "/")
          + "!Ukraine.gdb")
UA_LAYER = "UA_GEOG_ADM2_2001_uscb_201905"
CRIMEA_HEXES = ROOT / "data" / "geo" / "ua" / "crimea_hexes.gpkg"
UA_HEXES = ROOT / "data" / "geo" / "ua" / "ua_hexes.gpkg"


def crimea_subject(unit):
    """A 2001 Ukrainian unit id -> the Russian census subject it lies in, or None (mainland)."""
    if unit.startswith("UKR_01_"):
        return "UA-43"
    if unit == "UKR_02_01":
        return "UA-40"
    return None


def crimea(rd):
    """Crimea's and Sevastopol's hexes, keyed UA-43 / UA-40, with the boundary checks."""
    import pyogrio
    units = pyogrio.read_dataframe(UA_GDB, layer=UA_LAYER)[["GEO_MATCH", "geometry"]]
    units["geometry"] = units.geometry.make_valid()
    units["subj"] = units["GEO_MATCH"].map(crimea_subject)
    m = units.to_crs(6933)
    cr, mainland = m[m["subj"].notna()], m[m["subj"].isna()]
    if (cr["subj"] == "UA-43").sum() != 25 or (cr["subj"] == "UA-40").sum() != 1:
        raise SystemExit(f"Crimean units: {cr['subj'].value_counts().to_dict()} (want 25 + 1)")
    merged = cr.dissolve("subj")
    for s, geom in merged.geometry.items():
        parts = cr.loc[cr["subj"] == s].area.sum()
        if abs(geom.area - parts) > 1e-4 * parts:
            raise SystemExit(f"{s}: merged {geom.area / 1e6:,.1f} km2 vs members {parts / 1e6:,.1f}")
    land = mainland.geometry.union_all() if hasattr(mainland.geometry, "union_all") \
        else mainland.geometry.unary_union
    both = merged.geometry.union_all() if hasattr(merged.geometry, "union_all") \
        else merged.geometry.unary_union
    overlap = both.intersection(land).area / 1e6
    if overlap > 0.01:
        raise SystemExit(f"Crimea overlaps Ukraine's mainland units by {overlap:.3f} km2")
    shared = both.boundary.intersection(land.buffer(10)).length / 1e3
    if shared < 5:
        raise SystemExit(f"Crimea touches the mainland units along only {shared:.1f} km: a gap?")
    a43, a40 = merged.geometry["UA-43"], merged.geometry["UA-40"]
    if a43.intersection(a40).area > 1e4 or a43.distance(a40) > 1:
        raise SystemExit("UA-43 and UA-40 overlap or do not touch")
    print(f"  Crimea boundaries: UA-43 {a43.area / 1e6:,.0f} km2 (25 units), UA-40 "
          f"{a40.area / 1e6:,.0f} km2; {overlap:.4f} km2 shared with Ukraine's mainland units, "
          f"{shared:,.1f} km of boundary along them")

    h = gpd.read_file(CRIMEA_HEXES)
    h["unit"] = h["unit"].map(crimea_subject)
    if h["unit"].isna().any():
        raise SystemExit("a mainland unit's hex in crimea_hexes.gpkg")
    ua_main = gpd.read_file(UA_HEXES)
    if ua_main["unit"].map(crimea_subject).notna().any():
        raise SystemExit("Ukraine's mainland layer still holds Crimean hexes")
    dup = set(h.geometry.to_wkb()) & set(ua_main.geometry.to_wkb())
    if dup:
        raise SystemExit(f"{len(dup)} hexes in both Crimea's and Ukraine's mainland layer")
    # religiondots' Russia (Krasnodar across the Kerch Strait) must not reach into Crimea
    rdm = rd.to_crs(6933)
    near = rdm[rdm.intersects(both)]
    reach = near.intersection(both).area.sum() / 1e6
    if reach > 1.0:
        raise SystemExit(f"religiondots' Russia hexes cover {reach:.2f} km2 of Crimea "
                         f"({sorted(near['unit'].unique())})")
    print(f"  Crimea hexes: {len(h):,} ({h.groupby('unit')['pop'].sum().round().to_dict()}), "
          f"none in Ukraine's mainland layer; religiondots' Russia hexes cover {reach:.3f} km2 of "
          f"Crimea's polygons ({len(near)} hexes touching)")
    return h[["unit", "pop", "geometry"]]


def main():
    rd = gpd.read_file(SRC)
    print(f"  religiondots hexes: {len(rd):,}, {rd['unit'].nunique()} subjects, "
          f"{rd['pop'].sum():,.0f} people")
    g = gpd.GeoDataFrame(pd.concat([rd[["unit", "pop", "geometry"]], crimea(rd).to_crs(rd.crs)],
                                   ignore_index=True), crs=rd.crs)
    df = pd.read_csv(NORM)
    df = df[df["geo_level"] == "subject"]
    ur = df[df["area"] != "total"].pivot_table(index="geo_id", columns="area", values="count",
                                                aggfunc="sum").fillna(0)
    if set(ur.index) != set(g["unit"]):
        raise SystemExit(f"subjects differ: census only {sorted(set(ur.index) - set(g['unit']))}, "
                         f"hexes only {sorted(set(g['unit']) - set(ur.index))}")
    share = ur["urban"] / (ur["urban"] + ur["rural"])

    area = g.geometry.to_crs(6933).area.to_numpy() / 1e6
    g["dens"] = g["pop"].to_numpy() / np.maximum(area, 1e-6)
    g["ur"] = ""
    rows = []
    for iso, idx in g.groupby("unit").groups.items():
        sub = g.loc[idx].sort_values("dens", ascending=False)
        pop = sub["pop"].to_numpy(dtype=float)
        target = share[iso] * pop.sum()
        cum = np.cumsum(pop)
        before = cum - pop
        # urban while the running total before the hex is short of the target; the crossing hex
        # goes to the side that leaves the share closer
        urban = before < target
        k = int(urban.sum())                      # hexes [0, k) urban
        if 0 < k <= len(pop) and abs(cum[k - 1] - target) > abs(before[k - 1] - target):
            k -= 1
        if ur.loc[iso, "urban"] > 0:
            k = max(k, 1)
        if ur.loc[iso, "rural"] > 0:
            k = min(k, len(pop) - 1)
        flags = np.array(["r"] * len(pop), dtype=object)
        flags[:k] = "u"
        g.loc[sub.index, "ur"] = flags
        got = pop[:k].sum() / pop.sum() if pop.sum() else 0
        rows.append((iso, share[iso], got, k, len(pop)))
    g["unit"] = g["unit"] + "/" + g["ur"]

    rep = pd.DataFrame(rows, columns=["iso", "census_urban", "hex_urban", "urban_hexes", "hexes"])
    off = (rep["hex_urban"] - rep["census_urban"]).abs()
    print(f"  urban share, census vs hexes: mean gap {off.mean():.3f}, worst "
          f"{rep.loc[off.idxmax(), 'iso']} {off.max():.3f}")
    for r in rep.sort_values("iso").itertuples():
        print(f"    {r.iso:7} census {r.census_urban:5.3f}  hexes {r.hex_urban:5.3f}  "
              f"{r.urban_hexes:5,}/{r.hexes:,} hexes urban")
    if off.max() > 0.10:
        raise SystemExit("a subject's hexes miss the census urban share by more than 10 points")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["unit", "pop", "geometry"]].to_file(OUT, layer="grid3km", driver="GPKG")
    print(f"  wrote {OUT.relative_to(ROOT)}: {len(g):,} hexes, {g['unit'].nunique()} units")


if __name__ == "__main__":
    main()
