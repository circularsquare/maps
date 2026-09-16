"""Uruguay — the placement layer: Kontur 400 m population hexagons, keyed to department.

Writes data/geo/uy/uy_hexes.gpkg.

**URUGUAY IS THE MOST LOPSIDED COUNTRY THIS MAP HAS DRAWN AT ADM1 IN THE AMERICAS.**
Montevideo is 0.3% of the land and 37% of the people; the seventeen departments of the
interior are between 20 and 40 people per km² and their population sits almost entirely in
one departmental capital each, with grazing country in between. Spread a department's dots
evenly over its polygon and Uruguay's religion is painted across the estancias.

The second reason is the coast. Maldonado and Rocha are long thin departments whose people
are on a 200 km strip of Atlantic shoreline — Punta del Este, La Paloma, Chuy — and whose
interiors are almost empty; those two are also the departments with the highest unaffiliated
shares in the country, so getting their dots onto the coast is getting the map's most
striking claim into the right place.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a department line belongs wholly to
one side and no population is double-counted. Hexes whose centroid falls outside every
department (Kontur's UY extract overruns into Brazil and Argentina, and the Río de la Plata
and Uruguay river boundaries are wide) are dropped and reported.

## MONTEVIDEO IS CUT INTO ITS 62 BARRIOS, ON INE'S LINE AND NOT COD'S (2026-09-15)

Montevideo's dots are placed inside INE's barrios (`sources/uy_geo.py`). **COD-AB's Montevideo
polygon is not INE's Montevideo**: against INE's own 2011 department layer, whose Montevideo
is the union of the barrios to IoU 0.9999, COD's comes in at IoU 0.835. 70 km2 of INE's
Montevideo (the north of Villa García and Colón) is in COD's Canelones, holding about 19,700
Kontur people, and 18 km2 of COD's Montevideo (by Paso Carrasco, and across the Santa Lucía)
is outside every barrio, holding about 12,100. A centroid join on COD would have drawn
Montevideo's dots in Canelones and San José and left two barrios' northern halves empty.

So every hex that touches INE's Montevideo, or whose centroid COD puts there, is cut: its
pieces inside a barrio go to that barrio, its pieces outside go to the COD department they
lie in, and a piece of COD's Montevideo outside INE's line goes to the nearest other
department. The hex's people are shared over the area of its land pieces (`sources/mt_geo.py`'s
rule, so the river takes none). Every other hex keeps its centroid join. This moves Kontur
weight inside Canelones and San José near the line; it changes no department's count.

Usage:
    python sources/uy_grid.py --fetch    one ~2 MB gzipped gpkg from Kontur
    python sources/uy_grid.py            rebuild from data/raw/uy/ and data/geo/uy/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uy")
GEO = os.path.join(ROOT, "data", "geo", "uy")
DEPARTMENTS = os.path.join(GEO, "uy_departamentos.gpkg")
POP = os.path.join(GEO, "uy_pop_2023.csv")
OUT = os.path.join(GEO, "uy_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_UY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_UY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_UY_20231101.gpkg"

EXPECTED_DEPARTMENTS = 19
CAPITAL = "Montevideo"

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-department weight, so
# what matters is that it is not wildly out — assert the RELATIONSHIP.
INE_POPULATION = 3_496_400          # INE's estimate at 30 June 2023
KONTUR_TOLERANCE = 0.35

BARRIOS = os.path.join(GEO, "uy_barrios.gpkg")
BARRIO_POP = os.path.join(GEO, "uy_barrios_pop.csv")
MONTEVIDEO_PCODE = "UY10"
EXPECTED_BARRIOS = 62
UTM = 32721                         # UTM 21S, the barrio layer's own projection
# Both set on 2026-09-15 BEFORE the numbers were read. People in hexes on the Montevideo edge
# that end up with no land piece at all are dropped, and may not be more than this share of
# the edge's people. The barrio Kontur counts must rank the census's barrio counts better than
# every one of SHUFFLES random pairings.
EDGE_DROP_MAX = 0.005
SHUFFLES = 1000

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(DEPARTMENTS):
        raise SystemExit(f"missing {DEPARTMENTS} — run sources/uy_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    deps = gpd.read_file(DEPARTMENTS)
    if len(deps) != EXPECTED_DEPARTMENTS:
        raise SystemExit(f"{DEPARTMENTS} has {len(deps)} departments, "
                         f"expected {EXPECTED_DEPARTMENTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(deps.crs)
    hex_utm = hexes.to_crs(UTM)
    hexes = hexes.to_crs(deps.crs)

    joined = gpd.sjoin(pts, deps[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every department: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's UY extract overruns into Brazil and Argentina; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=deps.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(deps["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"departments with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"departments whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DEPARTMENTS} departments has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    cap = deps.loc[deps["name"] == CAPITAL, "unit"]
    if len(cap) != 1:
        raise SystemExit(f"{CAPITAL} is not one department in {DEPARTMENTS}")
    cap = cap.iloc[0]
    cap_share = per.loc[cap, "sum"] / out["pop"].sum()
    print(f"    {CAPITAL} ({cap}) holds {cap_share:.1%} of Kontur's Uruguayan population "
          f"in {int(per.loc[cap, 'size']):,} hexes")
    if not 0.25 < cap_share < 0.50:
        raise SystemExit(f"{CAPITAL} holds {cap_share:.1%} of the grid, which is not the "
                         "37% INE counts there — the spatial join is wrong")

    tot = float(out["pop"].sum())
    ratio = tot / INE_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs INE 2023 {INE_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and INE disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-department weight, so the level does not matter and "
          "the\n     shape does.")

    print("\n  per-department Kontur/INE ratio (the shape check):")
    ine = pd.read_csv(POP, encoding="utf-8-sig")
    ine = dict(zip(ine["geo_id"].astype(str).str.strip(), ine["pop_2023"]))
    rows = [(deps.loc[deps["unit"] == u, "name"].iloc[0], int(r["size"]), r["sum"] / ine[u])
            for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<20} {n:>8,} hexes   {r:5.2f}x")

    # ---- Montevideo's edge and its barrios (module docstring, last section)
    hx = gpd.GeoDataFrame({"hid": range(len(hex_utm)),
                           "unit": joined["unit"].to_numpy(),
                           "pop": pts[popcol].to_numpy(dtype=float)},
                          geometry=hex_utm.geometry.to_numpy(), crs=UTM)
    final = montevideo_edge(hx, deps)
    barrio_check(final)

    units = sorted(final["unit"].unique())
    n_dept = sum(1 for u in units if "-" not in u)
    n_barrio = sum(1 for u in units if u.startswith(MONTEVIDEO_PCODE + "-B"))
    if (n_dept, n_barrio, len(units)) != (EXPECTED_DEPARTMENTS - 1, EXPECTED_BARRIOS,
                                          EXPECTED_DEPARTMENTS - 1 + EXPECTED_BARRIOS):
        raise SystemExit(f"the place layer has {n_dept} departments and {n_barrio} barrios "
                         f"({len(units)} units)")
    final = final.to_crs(deps.crs)
    w, s, e, n = final.total_bounds
    if not (-58.6 < w < e < -53.0 and -35.1 < s < n < -30.0):
        raise SystemExit(f"bbox {w:.2f} {s:.2f} {e:.2f} {n:.2f} is not Uruguay")

    os.makedirs(GEO, exist_ok=True)
    final[["unit", "pop", "geometry"]].to_file(OUT + ".part.gpkg", layer="hexes", driver="GPKG")
    os.replace(OUT + ".part.gpkg", OUT)
    print(f"\nwrote {OUT} ({len(final):,} hexes and pieces, {n_dept} departments and "
          f"{n_barrio} barrios)")


def montevideo_edge(hx, deps):
    """Cut the hexes on Montevideo's edge by INE's line; keep every other hex whole.

    `hx` is every Kontur hex in EPSG:32721 with `unit` (its COD department by centroid, NaN
    outside every department) and `pop`. Returns the placement layer in EPSG:32721.
    """
    import geopandas as gpd
    import pandas as pd

    bar = gpd.read_file(BARRIOS).to_crs(UTM)
    if len(bar) != EXPECTED_BARRIOS:
        raise SystemExit(f"{BARRIOS} has {len(bar)} barrios, expected {EXPECTED_BARRIOS}")
    line = bar.union_all() if hasattr(bar, "union_all") else bar.unary_union
    d = deps.to_crs(UTM)

    near = hx["unit"].eq(MONTEVIDEO_PCODE).to_numpy() | hx.intersects(line).to_numpy()
    nh = hx[near]
    far = hx[~near & hx["unit"].notna().to_numpy()]
    cod_mvd = float(hx.loc[hx["unit"] == MONTEVIDEO_PCODE, "pop"].sum())
    print(f"\n  Montevideo's edge: {len(nh):,} hexes touch INE's Montevideo or are centred in "
          f"COD's, {nh['pop'].sum():,.0f} people; COD's centroid join put {cod_mvd:,.0f} in "
          "Montevideo")

    inside = gpd.overlay(nh[["hid", "pop", "geometry"]], bar[["unit", "geometry"]],
                         how="intersection", keep_geom_type=True)
    rest = gpd.overlay(nh[["hid", "pop", "geometry"]],
                       gpd.GeoDataFrame(geometry=[line], crs=UTM),
                       how="difference", keep_geom_type=True)
    rest = gpd.overlay(rest, d[["unit", "geometry"]], how="intersection", keep_geom_type=True)

    rest = rest.reset_index(drop=True)
    stray = rest["unit"] == MONTEVIDEO_PCODE
    if stray.any():
        # keep the pieces' own index: a GeoDataFrame built from an array plus an indexed
        # geometry Series aligns the two and leaves most centroids empty
        c = gpd.GeoDataFrame(geometry=rest.loc[stray].geometry.centroid, crs=UTM)
        nb = gpd.sjoin_nearest(c, d.loc[d["unit"] != MONTEVIDEO_PCODE, ["unit", "geometry"]],
                               how="left")
        nb = nb[~nb.index.duplicated(keep="first")]
        rest.loc[nb.index, "unit"] = nb["unit"]
        print(f"    {int(stray.sum()):,} pieces of COD's Montevideo lie outside INE's line and go "
              f"to the nearest other department: {sorted(set(nb['unit']))}")

    pieces = gpd.GeoDataFrame(pd.concat([inside[["hid", "pop", "unit", "geometry"]],
                                         rest[["hid", "pop", "unit", "geometry"]]],
                                        ignore_index=True), geometry="geometry", crs=UTM)
    if pieces["unit"].isna().any():
        raise SystemExit(f"{int(pieces['unit'].isna().sum())} edge pieces came out with no unit")
    land = pieces.area.groupby(pieces["hid"]).transform("sum")
    pieces["pop"] = pieces["pop"] * pieces.area / land
    pieces = pieces[pieces["pop"] > 0]

    dropped = float(nh.loc[~nh["hid"].isin(set(pieces["hid"])), "pop"].sum())
    share = dropped / float(nh["pop"].sum())
    print(f"    {dropped:,.0f} people ({share:.3%}) are in edge hexes with no land piece and "
          f"are dropped (bar {EDGE_DROP_MAX:.1%})")
    if share > EDGE_DROP_MAX:
        raise SystemExit("too many of the edge's people fall on no land piece; the cut is wrong")

    was = nh.assign(unit=nh["unit"].fillna("(outside)")).groupby("unit")["pop"].sum()
    now = pieces.assign(dept=pieces["unit"].str.slice(0, 4)).groupby("dept")["pop"].sum()
    print("    Kontur people on the edge, by department: COD centroid join -> INE's line")
    for u in sorted(set(was.index) | set(now.index)):
        print(f"      {u:<10} {was.get(u, 0.0):>12,.0f} -> {now.get(u, 0.0):>12,.0f}")

    mvd_now = float(pieces.loc[pieces["unit"].str.startswith(MONTEVIDEO_PCODE + "-"), "pop"].sum())
    print(f"    Montevideo holds {mvd_now:,.0f} Kontur people on INE's line against "
          f"{cod_mvd:,.0f} on COD's")
    out = pd.concat([far[["unit", "pop", "geometry"]], pieces[["unit", "pop", "geometry"]]],
                    ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=UTM)


def barrio_check(layer):
    """Kontur per barrio against the census's barrio count: the join's witness.

    The barrio counts come from INE, joined to the polygons by name (`sources/uy_geo.py`), so
    Kontur people per polygon are a measurement neither key determines. Normalised by
    Montevideo's own Kontur/census ratio; asserted as a rank correlation that beats every
    shuffle, set before the numbers were read. A barrio with no piece stops the build.
    """
    import numpy as np
    import pandas as pd

    bp = pd.read_csv(BARRIO_POP, dtype={"geo_id": str}).set_index("geo_id")
    b = layer[layer["unit"].str.startswith(MONTEVIDEO_PCODE + "-B")]
    k = b.groupby("unit")["pop"].sum().reindex(bp.index)
    npieces = b.groupby("unit").size().reindex(bp.index).fillna(0)
    empty = sorted(npieces.index[npieces == 0])
    if empty:
        raise SystemExit(f"barrios with no Kontur piece: {empty}")
    rel = (k / bp["census_2023"]) / (k.sum() / bp["census_2023"].sum())
    q = rel.quantile([0.1, 0.5, 0.9])
    print(f"\n  barrios: Kontur share over census 2023 share, p10 {q[0.1]:.2f} median "
          f"{q[0.5]:.2f} p90 {q[0.9]:.2f}; pieces per barrio median {npieces.median():.0f}, "
          f"min {int(npieces.min())}")
    for label, s in (("lowest", rel.nsmallest(4)), ("highest", rel.nlargest(4))):
        print(f"    {label}: " + ", ".join(f"{bp.loc[g, 'name']} {v:.2f}" for g, v in s.items()))
    obs = k.rank().corr(bp["census_2023"].rank())
    rng = np.random.default_rng(0)
    cr = bp["census_2023"].rank().to_numpy()
    kr = k.rank().to_numpy()
    null = np.array([np.corrcoef(kr, rng.permutation(cr))[0, 1] for _ in range(SHUFFLES)])
    print(f"    Spearman Kontur vs census over {len(bp)} barrios {obs:+.3f}; best of "
          f"{SHUFFLES} shuffles {null.max():+.3f}")
    if not obs > null.max():
        raise SystemExit("Kontur does not rank the barrios' census counts better than chance; "
                         "the name join or the polygons are wrong")


if __name__ == "__main__":
    main()
