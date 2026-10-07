"""Seychelles: the 2022 census's 25 districts and 18 islands, and a Kontur placement layer on them.

Writes data/geo/sc/sc_units.gpkg (the units) and data/geo/sc/sc_hexes.gpkg (placement pieces with
`unit` and `pop`). Reads religiondots' copy of OCHA COD-AB `cod-ab-syc` (`syc_admbnda_adm3_nbs2010`)
and of Kontur's 2023 Seychelles extract, read-only.

THE UNITS. Table B3.1a (sources/sc_census.py) prints the 2010 census's districts for Mahé and
Praslin, plus Ile Perseverance, which became a district in 2022, and then two regions island by
island: `La Digue & Inner Islands` (Bird, Denis, Fregate, La Digue, North, Silhouette) and `Outer
Islands` (12 islands). So the units are:

  * 24 COD districts as drawn, with English River WITHOUT Perseverance Island (religiondots merged
    the two for 2010, when the island was not yet a district);
  * COD's `Perseverance Island` feature as Ile Perseverance;
  * La Digue: COD's La Digue plus Félicité, Marianne, the Soeurs and Cocos, which religiondots
    showed the 2010 census counts with La Digue (its printed 36.4 km2 closes only with them). The
    2022 table has no rows for them, so they stay with La Digue;
  * 17 more islands cut by position out of COD's `Other Islands` multipolygon (from GAUL): every
    part whose centroid lies within a radius of the island's position. Each named island must
    catch at least one part and land inside an area band; no part may be caught twice.

COD parts left over (Cosmoledo, Astove, St Pierre, Desnoeufs, St François and Bijoutier near
Alphonse, the Mahé islets, Cousin and Cousine, the uninhabited islets of D'Arros' atoll) are in no
2022 row and are dropped; the script prints them.

THE PLACEMENT is religiondots' sc_grid.py method on these units: Kontur hexes CUT BY OVERLAP
(Victoria's districts are 1.2-1.7 km2 against a 0.74 km2 hex, too small for a centroid rule),
a part-sea hex keeping its whole population on its land pieces, strays with no land under COD's
shoreline snapped to the nearest unit within 700 m. Perseverance Island's hexes are KEPT here:
religiondots dropped them because the island was empty in 2010, but the 2022 census counts 5,410
people on it. An island Kontur gives no populated hex gets its own polygon as one piece with
pop 1, so its dots spread evenly over it rather than vanish.

Usage:
    python sources/sc_geo.py
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd                       # noqa: E402
import pandas as pd                           # noqa: E402
from shapely.geometry import MultiPolygon     # noqa: E402
from shapely.ops import unary_union           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
RD_RAW = ROOT.parent / "religiondots" / "data" / "raw" / "sc"
COD = RD_RAW / "cod_shp" / "syc_admbnda_adm3_nbs2010.shp"
KONTUR = RD_RAW / "kontur_population_SC_20231101.gpkg"
NORM = ROOT / "data" / "normalized" / "sc.csv"
GEO = ROOT / "data" / "geo" / "sc"
UNITS_OUT = GEO / "sc_units.gpkg"
HEX_OUT = GEO / "sc_hexes.gpkg"

AREA_CRS = "ESRI:102022"       # Africa Albers Equal Area Conic, COD's own recommendation
SNAP_M = 700.0
MAX_SPAN_DEG = 12.0            # Aldabra at 46°E to Mahé at 55.5°E

# COD ADM3 name -> unit id, for the districts drawn as COD draws them.
DISTRICTS = {
    "Anse Aux Pins": "SC-01", "Anse Boileau": "SC-02", "Anse Etoile": "SC-03", "Au Cap": "SC-04",
    "Anse Royale": "SC-05", "Baie Lazare": "SC-06", "Baie Sainte Anne": "SC-07",
    "Beau Vallon": "SC-08", "Bel Air": "SC-09", "Belombre": "SC-10", "Cascade": "SC-11",
    "Glacis": "SC-12", "Grand Anse Mahe": "SC-13", "Grand Anse Praslin": "SC-14",
    "English River": "SC-16", "Mont Buxton": "SC-17", "Mont Fleuri": "SC-18",
    "Plaisance": "SC-19", "Pointe Larue": "SC-20", "Port Glaud": "SC-21", "Saint Louis": "SC-22",
    "Takamaka": "SC-23", "Les Mamelles": "SC-24", "Roche Caiman": "SC-25",
    "Perseverance Island": "SC-PI",
}
# The ISO number in COD's pcode must agree with the unit id (as religiondots asserts).
LA_DIGUE = "SC-I-LADIGUE"
# Parts of Other Islands that go with La Digue: Félicité, Marianne, the Soeurs, Cocos.
LA_DIGUE_BOX = (-4.37, 55.84, -4.27, 55.95)            # lat_min, lon_min, lat_max, lon_max

# (name in Table B3.1a, unit id, lat, lon, radius km, min km2, max km2). Positions read off the
# parts COD draws (sources/sc.md lists them) and checked against each island's known place.
ISLANDS = [
    ("Bird", "SC-I-BIRD", -3.721, 55.205, 3, 0.4, 1.2),
    ("Denis", "SC-I-DENIS", -3.806, 55.667, 3, 0.9, 1.8),
    ("Fregate", "SC-I-FREGATE", -4.586, 55.941, 3, 1.5, 2.5),
    ("North", "SC-I-NORTH", -4.393, 55.245, 3, 1.5, 2.5),
    ("Silhouette", "SC-I-SILHOUETTE", -4.487, 55.230, 3, 15.0, 25.0),
    ("Aldabra", "SC-I-ALDABRA", -9.42, 46.35, 25, 100.0, 200.0),
    ("Alphonse", "SC-I-ALPHONSE", -7.006, 52.727, 3, 1.2, 2.0),
    ("Assumption", "SC-I-ASSUMPTION", -9.732, 46.511, 4, 8.0, 14.0),
    ("Coetivy", "SC-I-COETIVY", -7.129, 56.280, 6, 7.0, 11.0),
    ("Darros", "SC-I-DARROS", -5.416, 53.298, 2, 1.2, 2.0),
    ("Desroches", "SC-I-DESROCHES", -5.690, 53.668, 4, 3.0, 4.5),
    ("Farquhar", "SC-I-FARQUHAR", -10.15, 51.15, 12, 5.0, 9.0),
    ("Marie-Louise", "SC-I-MARIELOUISE", -6.179, 53.144, 3, 0.4, 0.8),
    ("Platte", "SC-I-PLATTE", -5.864, 55.385, 3, 0.3, 0.6),
    ("Poivre", "SC-I-POIVRE", -5.760, 53.306, 3, 2.0, 2.8),
    ("Providence", "SC-I-PROVIDENCE", -9.224, 51.031, 4, 1.2, 1.9),
    ("Remire", "SC-I-REMIRE", -5.117, 53.312, 3, 0.15, 0.35),
]


def _parts(geom):
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]


def build_units():
    g = gpd.read_file(COD)
    if len(g) != 27 or g.crs.to_epsg() != 4326:
        raise SystemExit(f"COD ADM3: {len(g)} features, CRS {g.crs}; expected 27 in EPSG:4326")
    names = list(g["ADM3_EN"])
    if sorted(set(names) - set(DISTRICTS)) != ["La Digue", "Other Islands"]:
        raise SystemExit(f"COD names outside the district list: {sorted(set(names) - set(DISTRICTS))}")
    rows = []
    for _, r in g.iterrows():
        n = r["ADM3_EN"]
        if n in DISTRICTS:
            uid = DISTRICTS[n]
            if uid != "SC-PI" and r["ADM3_PCODE"][4:6] != uid[3:5]:
                raise SystemExit(f"{n}: pcode {r['ADM3_PCODE']} disagrees with {uid}")
            rows.append((uid, n, r["ADM3_PCODE"], r.geometry))

    oi = g.loc[g["ADM3_EN"] == "Other Islands"].geometry.iloc[0]
    parts = gpd.GeoSeries(_parts(oi), crs=4326)
    area = (parts.to_crs(AREA_CRS).area / 1e6).to_numpy()
    cen = parts.to_crs(AREA_CRS).centroid.to_crs(4326)
    owner = [None] * len(parts)

    lat0, lon0, lat1, lon1 = LA_DIGUE_BOX
    for i, c in enumerate(cen):
        if lat0 <= c.y <= lat1 and lon0 <= c.x <= lon1:
            owner[i] = LA_DIGUE
    n_ld = sum(1 for o in owner if o == LA_DIGUE)
    a_ld = sum(a for a, o in zip(area, owner) if o == LA_DIGUE)
    print(f"  La Digue: COD's own feature + {n_ld} Other Islands parts ({a_ld:.2f} km2: Félicité, "
          "Marianne, the Soeurs, Cocos)")
    if not 4.0 <= a_ld <= 5.5:
        raise SystemExit("the La Digue inner parts are not the expected 4-5.5 km2")
    ld = g.loc[g["ADM3_EN"] == "La Digue"].iloc[0]
    rows.append((LA_DIGUE, "La Digue", ld["ADM3_PCODE"] + "+inner parts of SC4726OI",
                 unary_union([ld.geometry] + [p for p, o in zip(parts, owner) if o == LA_DIGUE])))

    print("  islands cut out of COD's Other Islands:")
    for name, uid, lat, lon, rad, lo, hi in ISLANDS:
        hits = []
        for i, c in enumerate(cen):
            dy = (c.y - lat) * 111.0
            dx = (c.x - lon) * 111.0 * math.cos(math.radians(lat))
            if math.hypot(dx, dy) <= rad:
                hits.append(i)
        taken = [i for i in hits if owner[i] is not None]
        if taken:
            raise SystemExit(f"{name}: parts already given to {owner[taken[0]]}")
        a = sum(area[i] for i in hits)
        print(f"      {name:<14}{len(hits):>4} parts {a:8.2f} km2")
        if not hits or not lo <= a <= hi:
            raise SystemExit(f"{name}: {len(hits)} parts, {a:.2f} km2, expected {lo}-{hi}")
        for i in hits:
            owner[i] = uid
        rows.append((uid, name, "part of SC4726OI", unary_union([parts[i] for i in hits])))

    left = [(round(cen[i].y, 2), round(cen[i].x, 2), area[i]) for i in range(len(parts))
            if owner[i] is None]
    print(f"  {len(left)} Other Islands parts in no 2022 row, dropped, "
          f"{sum(x[2] for x in left):.1f} km2; the larger ones:")
    for y, x, a in sorted(left, key=lambda t: -t[2])[:8]:
        print(f"      {y:7.2f} {x:7.2f} {a:6.2f} km2")

    units = gpd.GeoDataFrame({"unit": [r[0] for r in rows], "name": [r[1] for r in rows],
                              "cod_pcode": [r[2] for r in rows]},
                             geometry=[r[3] for r in rows], crs=4326)
    if units["unit"].nunique() != 43:
        raise SystemExit(f"{units['unit'].nunique()} units, expected 43")
    return units


def main():
    units = build_units()
    df = pd.read_csv(NORM)
    census = df.groupby("geo_id")["count"].sum()
    a, b = set(census.index), set(units["unit"])
    if a != b:
        raise SystemExit(f"sc.csv and the units disagree: only in csv {sorted(a - b)}, "
                         f"only in units {sorted(b - a)}")
    print(f"  join: the 43 units of sc.csv and the 43 polygons are the same set, both ways")
    GEO.mkdir(parents=True, exist_ok=True)
    units.to_file(UNITS_OUT, layer="units", driver="GPKG")

    hexes = gpd.read_file(KONTUR)
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    hexes = hexes.rename(columns={popcol: "kpop"})[["kpop", "geometry"]]
    hexes = hexes[hexes["kpop"] > 0].to_crs(AREA_CRS).reset_index(drop=True)
    hexes["hid"] = hexes.index
    span = hexes.to_crs(4326).total_bounds
    if span[2] - span[0] > MAX_SPAN_DEG:
        raise SystemExit("the grid is torn [[reference_antimeridian]]")
    print(f"\n  Kontur 2023: {len(hexes)} populated hexes, {hexes['kpop'].sum():,.0f} people")

    u = units[["unit", "geometry"]].to_crs(AREA_CRS)
    pieces = gpd.overlay(hexes, u, how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.geometry.area
    pieces = pieces[pieces["a"] > 1.0]
    pieces["pop"] = pieces["kpop"] * pieces["a"] / pieces.groupby("hid")["a"].transform("sum")
    strays = hexes[~hexes["hid"].isin(set(pieces["hid"]))]
    snapped = []
    if len(strays):
        pts = gpd.GeoDataFrame({"hid": strays["hid"], "kpop": strays["kpop"]},
                               geometry=strays.geometry.centroid, crs=AREA_CRS)
        near = gpd.sjoin_nearest(pts, u, how="left", max_distance=SNAP_M, distance_col="_d")
        near = near[~near.index.duplicated(keep="first")]
        ok = near["unit"].notna()
        print(f"  {len(strays)} hexes touch no unit ({strays['kpop'].sum():,.0f} people): "
              f"{int(ok.sum())} snapped within {SNAP_M:.0f} m ({near.loc[ok, 'kpop'].sum():,.0f}), "
              f"{int((~ok).sum())} dropped ({near.loc[~ok, 'kpop'].sum():,.0f})")
        keep = strays.set_index("hid").loc[near.loc[ok, "hid"]]
        snapped = [gpd.GeoDataFrame({"unit": near.loc[ok, "unit"].to_numpy(),
                                     "pop": keep["kpop"].to_numpy()},
                                    geometry=keep.geometry.to_numpy(), crs=AREA_CRS)]
    out = pd.concat([pieces[["unit", "pop", "geometry"]]] + snapped, ignore_index=True)
    out["src"] = "kontur"
    have = set(out.loc[out["pop"] > 0, "unit"])
    bare = units[~units["unit"].isin(have)].to_crs(AREA_CRS)
    if len(bare):
        print(f"  {len(bare)} units with no Kontur population get their own polygon, pop 1: "
              + ", ".join(f"{n} ({int(census[x]):,})" for x, n in zip(bare["unit"], bare["name"])))
        out = pd.concat([out, gpd.GeoDataFrame({"unit": bare["unit"].to_numpy(), "pop": 1.0,
                                                "src": "polygon"},
                                               geometry=bare.geometry.to_numpy(), crs=AREA_CRS)],
                        ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=AREA_CRS).to_crs(4326)

    # Kontur against the census, Kontur-placed units only
    per = out[out["src"] == "kontur"].groupby("unit")["pop"].sum()
    rows = [(x, float(census[x]), float(per[x])) for x in per.index if census.get(x, 0) > 0]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    norm = sorted((k / c / ratio, x) for x, c, k in rows)
    print(f"\n  Kontur / census (3+) {ratio:.3f} over {len(rows)} units; normalised "
          f"lowest {', '.join(f'{x} {r:.2f}' for r, x in norm[:4])}; "
          f"highest {', '.join(f'{x} {r:.2f}' for r, x in norm[-4:])}")
    mahe = [(r, x) for r, x in norm if not x.startswith("SC-I-") or x == LA_DIGUE]
    bad = [(x, round(r, 2)) for r, x in mahe if not 0.5 <= r <= 2.0]
    print(f"  districts and La Digue outside a factor of 2: {bad or 'none'}")
    if len(bad) > 2:
        raise SystemExit("more than two districts outside a factor of 2; look at the join")
    import random
    sel = [(c, k) for x, c, k in rows if not x.startswith("SC-I-") or x == LA_DIGUE]
    lc = [math.log(c) for c, _ in sel]
    lk = [math.log(k) for _, k in sel]

    def pear(p, q):
        mp, mq = sum(p) / len(p), sum(q) / len(q)
        num = sum((s - mp) * (t - mq) for s, t in zip(p, q))
        return num / math.sqrt(sum((s - mp) ** 2 for s in p) * sum((t - mq) ** 2 for t in q))
    r = pear(lc, lk)
    rng = random.Random(0)
    beat = sum(1 for _ in range(2000) if pear(lc, rng.sample(lk, len(lk))) >= r)
    print(f"  log census against log Kontur over {len(sel)} districts: r = {r:.3f}; "
          f"{beat} of 2,000 shuffles reach it")
    if beat > 20:
        raise SystemExit("the join is not carrying information")

    tmp = HEX_OUT.with_name("sc_hexes.part.gpkg")
    if tmp.exists():
        tmp.unlink()
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, HEX_OUT)
    print(f"\nwrote {UNITS_OUT} (43 units) and {HEX_OUT} ({len(out):,} pieces)")


if __name__ == "__main__":
    main()
