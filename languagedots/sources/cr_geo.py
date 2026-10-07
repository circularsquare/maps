"""Costa Rica placement layer: Kontur 2023-11 400 m hexes keyed to the 472 distritos of the 2011
census -> data/geo/cr/cr_hexes.gpkg (and cr_units.gpkg, the rebuilt 2011 distritos).

    python sources/cr_geo.py          (needs data/raw/cr/ from sources/cr_censo.py --fetch)

BOUNDARIES. OCHA COD-AB Costa Rica (valid_on 2024-12-03), admin3, 492 distritos, read from
religiondots' download (religiondots/data/raw/cr/gdb/, read-only; religiondots draws Costa Rica
by province). Read with engine="fiona": pyogrio returns zero features from this .gdb
(religiondots/sources/cr_geo.py found the same).

THE 2011 DISTRITOS ARE REBUILT FROM THE 2024 ONES. The census has 472; COD has 492. A COD pcode is
CR + the five-digit DTA code, and 468 of the census's codes are COD codes. The rest:

  - three distritos became cantons of their own after 2011 and were renumbered (NEW_CANTONS):
    Grecia's Rio Cuarto (20306) is canton 216's three distritos, Puntarenas's Monteverde (60109)
    is 61201, Golfito's Puerto Jimenez (60702) is 61301;
  - seventeen distritos were carved out of 2011 distritos (CARVED). Each goes to the 2011 distrito(s)
    of the same canton that lost that much area, read from the census base's own area per distrito
    (EXTTER) against COD's area; where two lost land (Caldera from Espiritu Santo and San Juan
    Grande; Gutierrez Braun from San Vito and Sabalito; Jaris and Quitirrisi from four of Mora's)
    the new polygon is cut on a 250 m lattice, each cell to the nearest of the parents, weighted
    (WEIGHTED) so each parent gets back about the land it lost;
  - Isla del Coco (60110, no residents) is left out: 2011's Puntarenas area does not include it.
  - Puriscal's 10405 and 10408 take each other's COD polygon (POLYGON_FOR).
  - outside hexes: snapped on the coast, dropped inside Nicaragua or Panama (outside_hexes).

THE JOIN IS ON CODE, AND THE WITNESS IS AREA. REDATAM's own labels are unreliable here: 60111-60116
are labelled one place out (60111 "Chacarita" is 316.6 km2, which is Cobano, as COD's 60111 is),
and the two sources swap the names of Puriscal's 10405 and 10408. The census base carries an area
per distrito, independent of both keys, so every rebuilt distrito's area is compared with it.
Boundary moves between existing distritos since 2011 (no new unit, so nothing to undo) show up as
area failures and are pinned in AREA_MOVED with the place they went; their people moved are few.
The Puriscal pair also gets a Kontur witness (printed).

PLACEMENT is sources/_grid.py's hex_layer: each Kontur hex to the distrito its centroid falls in.
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

GDB = RD / "data" / "raw" / "cr" / "gdb" / "cri_admin_boundaries.gdb"
OUT_UNITS = HERE / "data" / "geo" / "cr" / "cr_units.gpkg"
N = 472
CRS_M = 8908      # CR-SIRGAS / CRTM05, metres

NEW_CANTONS = {"21601": "20306", "21602": "20306", "21603": "20306",
               "61201": "60109", "61301": "60702"}
# Isla del Coco, a distrito since 2011 with no residents: Puntarenas (60101) is 34.1 km2 in COD
# against 35.6 in EXTTER without it, so the 2011 base counted its area in no distrito. Left out.
COCO = {"60110": None}
# post-2011 distrito -> the 2011 distrito(s) it was carved from: same-canton distritos whose COD
# area is short of the census's EXTTER. Several parents: cut on the nearest parent, and where
# WEIGHTED, with each parent's piece sized to the land it lost.
WEIGHTED = {"10707", "60206", "60806"}
CARVED = {
    "10706": ["10701"],                              # Jaris (Mora) from Colon
    "10707": ["10701", "10702", "10703", "10704"],   # Quitirrisi from Colon, Guayabo, Tabarcia,
                                                     # Piedras Negras
    "11912": ["11907"],                              # La Amistad from Pejibaye (Perez Zeledon)
    "20214": ["20208"],                              # San Lorenzo from Angeles (San Ramon)
    "20404": ["20403"],                              # Labrador from Jesus Maria (San Mateo)
    "21308": ["21301"],                              # Canalete from Upala
    "30206": ["30201"],                              # Birrisito from Paraiso
    "30404": ["30401"],                              # La Victoria from Juan Vinas
    "40207": ["40202"],                              # Puente Salas from San Pedro (Barva)
    "50808": ["50802"],                              # Cabeceras from Quebrada Grande (Tilaran)
    "51105": ["51101"],                              # Matambu from Hojancha (nearest, by a test cut)
    "60206": ["60201", "60202"],                     # Caldera from Espiritu Santo, San Juan Grande
    "60506": ["60503"],                              # Bahia Drake from Sierpe
    "60806": ["60801", "60802"],                     # Gutierrez Braun from San Vito, Sabalito
    "61103": ["61102"],                              # Lagunillas from Tarcoles
    "70207": ["70201", "70202", "70203", "70204", "70205", "70206"],  # La Colonia (Pococi)
    "70307": ["70301"],                              # Reventazon from Siquirres
}
# census code -> COD pcode, where COD's polygon for a code is another distrito's. COD has Puriscal's
# 10405 as "San Antonio" and 10408 as "San Rafael"; the official DTA order (and REDATAM) is 10405
# San Rafael, 10408 San Antonio. The polygons follow COD's NAMES: Kontur puts 3,927 people (at the
# national ratio) in COD's 10405 and 2,159 in its 10408, against the census's 1,730 for San Rafael
# and 3,889 for San Antonio. Swapped, both areas stay inside the area bar (13.9 vs 15.5 km2,
# 16.9 vs 14.6). The Puriscal witness in main() is asserted.
POLYGON_FOR = {"10405": "10408", "10408": "10405"}
EXPECT_NO_HEX = {"10802"}
SNAP_M = 2_000
AREA_TOL = 0.15
AREA_FLOOR = 2.0   # km2
# 2011 distritos whose rebuilt area is off by more than AREA_TOL because a line between EXISTING
# distritos moved after 2011, each looked at: code -> what moved.
_POCOCI = "Pococi-Guacimo line moved ~220 km2 of the Caribbean lowland to Rio Jimenez"
_ORIENTE = "Paraiso-Jimenez-Turrialba lines moved: Orosi +61 km2, Tayutic, Santa Rosa, Pejibaye -77"
_SANCARLOS = "a line moved inside San Carlos: Aguas Zarcas +26 km2, Palmera and Buenavista -36"
_SARCHI = "lines moved inside Sarchi (Valverde Vega): Sarchi Norte, Toro Amarillo +22, San Pedro -5"
AREA_MOVED = {
    "70206": _POCOCI, "70604": _POCOCI,
    "30203": _ORIENTE, "30508": _ORIENTE, "30509": _ORIENTE, "30403": _ORIENTE,
    "21003": _SANCARLOS, "21009": _SANCARLOS, "21004": _SANCARLOS,
    "21201": _SARCHI, "21203": _SARCHI, "21204": _SARCHI,
    "50702": "La Sierra (Abangares) -30 km2; COD's line with Montes de Oro differs",
    "60402": "Union (Montes de Oro) +18 km2, the other side of La Sierra's line",
    "50502": "Palmira (Carrillo) -6 km2 to a neighbour",
    "20608": "Palmitos (Naranjo) +2.3 km2, just over the floor",
}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def census_tables():
    import cr_censo
    names = cr_censo.district_names()
    areas = cr_censo.district_areas()
    df = pd.read_csv(HERE / "data" / "normalized" / "cr.csv", dtype={"geo_id": str})
    pop = df.groupby("geo_id")["count"].sum().to_dict()
    return names, areas, pop


def read_cod():
    import geopandas as gpd
    g = gpd.read_file(GDB, layer="cri_admin3", engine="fiona")
    if len(g) != 492:
        raise SystemExit(f"COD admin3: {len(g)} features, expected 492")
    g["code"] = g["adm3_pcode"].str.replace("CR", "", regex=False)
    if not g["code"].str.fullmatch(r"\d{5}").all() or g["code"].duplicated().any():
        raise SystemExit("COD admin3: pcodes are not unique CR + five digits")
    return g.to_crs(CRS_M)


def split_nearest(poly, parents, targets, step=250.0):
    """Cut `poly` into pieces, each cell of a `step` m lattice to the nearest parent polygon, the
    distances offset per parent (a weighted Voronoi) until each parent's piece is the land it
    lost: `targets` {parent: km2 lost}, scaled to the new polygon's area."""
    import numpy as np
    import shapely
    from shapely.geometry import box
    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0, x1 + step, step)
    ys = np.arange(y0, y1 + step, step)
    cells = [box(x, y, x + step, y + step) for x in xs for y in ys]
    cells = [c for c in cells if c.intersects(poly)]
    cell_km2 = np.array([c.intersection(poly).area / 1e6 for c in cells])
    pts = shapely.points([(c.centroid.x, c.centroid.y) for c in cells])
    codes = list(parents)
    dist = np.array([[p.distance(parents[k]) for k in codes] for p in pts])
    if targets is None:                  # plain nearest parent
        targets = {k: 1.0 for k in codes}
        tries = 0
    else:
        tries = 400
    want = np.array([max(targets[k], 0.0) for k in codes])
    want = want / want.sum() * cell_km2.sum()
    w = np.zeros(len(codes))
    for _ in range(tries):
        pick = (dist - w).argmin(axis=1)
        got = np.array([cell_km2[pick == i].sum() for i in range(len(codes))])
        if np.abs(got - want).max() < 0.02 * cell_km2.sum():
            break
        w += 1500.0 * (want - got) / cell_km2.sum()
    pick = (dist - w).argmin(axis=1)
    out = {}
    for k_i, k in enumerate(codes):
        mine = [c for c, j in zip(cells, pick) if j == k_i]
        if mine:
            piece = shapely.union_all(mine).intersection(poly)
            if not piece.is_empty:
                out[k] = piece
    return out


def build_units(names, areas):
    import geopandas as gpd
    import shapely
    cod = read_cod()
    geom = dict(zip(cod["code"], cod.geometry))
    census = set(names)
    pieces = {c: [geom[POLYGON_FOR.get(c, c)]] for c in cod["code"] if c in census}
    for new, old in NEW_CANTONS.items():
        pieces.setdefault(old, []).append(geom[new])
    for new, parents in CARVED.items():
        if len(parents) == 1:
            pieces[parents[0]].append(geom[new])
    for new, parents in CARVED.items():
        if len(parents) == 1:
            continue
        lost = {p: areas[p] - sum(g.area for g in pieces[p]) / 1e6 for p in parents}
        cut = split_nearest(geom[new], {p: geom[p] for p in parents},
                            lost if new in WEIGHTED else None)
        print(f"  {new} split: " + ", ".join(f"{p} {g.area / 1e6:.1f} km2 (lost {lost[p]:.1f})"
                                             for p, g in cut.items()))
        for p, g in cut.items():
            pieces[p].append(g)
    used = set(pieces) | set(NEW_CANTONS) | set(COCO) | set(CARVED)
    left = sorted(set(cod["code"]) - used - census)
    if left:
        raise SystemExit(f"COD distritos given to no 2011 distrito: {left}")
    if set(pieces) != census:
        raise SystemExit(f"rebuilt {len(pieces)} units; missing {sorted(census - set(pieces))[:8]}, "
                         f"extra {sorted(set(pieces) - census)[:8]}")
    units = gpd.GeoDataFrame({"unit": list(pieces)},
                             geometry=[shapely.union_all(v) for v in pieces.values()], crs=CRS_M)
    print(f"  {len(units)} units rebuilt from {len(cod)} COD distritos")
    if len(units) != N:
        raise SystemExit(f"{len(units)} units, expected {N}")

    # the area witness
    units["km2"] = units.area / 1e6
    units["extter"] = units["unit"].map(areas)
    units["r"] = units["km2"] / units["extter"]
    # EXTTER is printed to 0.1 km2, so a small unit needs an absolute floor as well
    off = units[((units["r"] - 1).abs() > AREA_TOL)
                & ((units["km2"] - units["extter"]).abs() > AREA_FLOOR)].sort_values("r")
    print(f"  area against the census base's EXTTER: median ratio {units['r'].median():.3f}; "
          f"{len(off)} of {N} off by more than {AREA_TOL:.0%}")
    for _, r in off.iterrows():
        tag = AREA_MOVED.get(r["unit"], "!! not pinned")
        print(f"     {r['unit']} {names[r['unit']]:<28} {r['km2']:8.1f} km2 vs {r['extter']:8.1f}"
              f"  ({r['r']:.2f})  {tag}")
    unpinned = set(off["unit"]) - set(AREA_MOVED)
    stale = set(AREA_MOVED) - set(off["unit"])
    if unpinned or stale:
        raise SystemExit(f"area witness: unpinned {sorted(unpinned)}, pinned but fine "
                         f"{sorted(stale)}")
    # a shuffled control: the same comparison with the areas dealt out at random
    import numpy as np
    rng = np.random.default_rng(0)
    ext = units["extter"].to_numpy()
    shuffled = max(float(np.mean(np.abs(units["km2"].to_numpy() / rng.permutation(ext) - 1)
                                 <= AREA_TOL)) for _ in range(200))
    print(f"  share within {AREA_TOL:.0%}: {1 - len(off) / N:.3f} on the code join, at most "
          f"{shuffled:.3f} over 200 shuffles")
    return units.to_crs(4326)


def outside_hexes(units, layer_path):
    """hex_layer drops every hex whose centroid is in no unit. Here: a hex inside Natural Earth's
    Nicaragua or Panama is the neighbour's and stays dropped; any other (the sea off Puntarenas's
    sandspit, the Gulf of Nicoya, river mouths) is snapped to the nearest distrito within SNAP_M."""
    import geopandas as gpd
    from _grid import kontur_path
    hexes = gpd.read_file(kontur_path("cr"))
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(CRS_M)
    um = units[["unit", "geometry"]].to_crs(CRS_M)
    inside = gpd.sjoin(pts, um, how="left", predicate="within")
    inside = inside[~inside.index.duplicated(keep="first")].reindex(pts.index)
    out = pts[inside["unit"].isna()].copy()
    ne = gpd.read_file(RD / "data" / "geo" / "ne_10m_admin_0_countries.geojson")
    ne = ne[ne["ISO_A2_EH"].isin(["NI", "PA"])][["ISO_A2_EH", "geometry"]].to_crs(CRS_M)
    nb = gpd.sjoin(out, ne, how="left", predicate="within")
    nb = nb[~nb.index.duplicated(keep="first")].reindex(out.index)
    abroad = nb["ISO_A2_EH"].notna()
    print(f"  {len(out):,} hexes ({out['pop'].sum():,.0f} people) outside every distrito: "
          f"{int(abroad.sum()):,} ({out.loc[abroad, 'pop'].sum():,.0f}) in Nicaragua or Panama, "
          f"dropped; the rest coast or sea")
    sea = out[~abroad]
    snap = gpd.sjoin_nearest(sea, um, how="left", max_distance=SNAP_M, distance_col="d")
    snap = snap[~snap.index.duplicated(keep="first")]
    ok = snap["unit"].notna()
    print(f"     snapped {int(ok.sum()):,} hexes ({snap.loc[ok, 'pop'].sum():,.0f} people) within "
          f"{SNAP_M:,} m; left {int((~ok).sum()):,} ({snap.loc[~ok, 'pop'].sum():,.0f}) further out")
    for u, p in snap[ok].groupby("unit")["pop"].sum().sort_values().tail(5).items():
        print(f"       {u}: {p:,.0f}")
    add = gpd.GeoDataFrame({"unit": snap.loc[ok, "unit"].astype(str).to_numpy(),
                            "pop": snap.loc[ok, "pop"].to_numpy()},
                           geometry=hexes.geometry.loc[snap.index[ok]].to_numpy(),
                           crs=hexes.crs).to_crs(4326)
    return add


def main():
    import geopandas as gpd
    names, areas, pop = census_tables()
    units = build_units(names, areas)
    OUT_UNITS.parent.mkdir(parents=True, exist_ok=True)
    units[["unit", "km2", "extter", "geometry"]].to_file(OUT_UNITS, layer="distritos",
                                                       driver="GPKG")
    from _grid import hex_layer
    layer = hex_layer("cr", units[["unit", "geometry"]], census=pop)
    out = HERE / "data" / "geo" / "cr" / "cr_hexes.gpkg"
    add = outside_hexes(units, out)
    layer = pd.concat([layer, add], ignore_index=True)
    # a distrito no hex centroid falls in (San Francisco de Goicoechea, 0.6 km2) is placed on its
    # own polygon, at its census population times Kontur's national ratio
    ratio = layer["pop"].sum() / sum(pop.values())
    have = set(layer.loc[layer["pop"] > 0, "unit"])
    empty = units[~units["unit"].isin(have)]
    if set(empty["unit"]) != EXPECT_NO_HEX:
        raise SystemExit(f"distritos with no populated hex: {sorted(empty['unit'])}, expected "
                         f"{sorted(EXPECT_NO_HEX)}")
    own = gpd.GeoDataFrame({"unit": empty["unit"].to_numpy(),
                            "pop": [pop[u] * ratio for u in empty["unit"]]},
                           geometry=empty.geometry.to_numpy(), crs=4326)
    layer = gpd.GeoDataFrame(pd.concat([layer, own], ignore_index=True), crs=4326)
    if set(layer["unit"]) != set(pop):
        raise SystemExit("placement layer does not hold every distrito")
    layer.to_file(out, layer="hexes", driver="GPKG")
    print(f"  rewrote {out}: {len(layer):,} features ({len(add):,} snapped hexes, "
          f"{len(own)} own polygon)")
    per = layer.groupby("unit")["pop"].sum()
    r = sorted((per.get(u, 0) / ratio / pop[u], u) for u in pop)
    print("  Kontur / census per distrito after the fixes, normalised: lowest "
          + ", ".join(f"{u} {x:.2f}" for x, u in r[:4]) + "; highest "
          + ", ".join(f"{u} {x:.2f}" for x, u in r[-4:]))
    print("  Puriscal witness (COD gives these two codes each other's names and polygons; "
          "POLYGON_FOR swaps them back):")
    for c in ("10405", "10408"):
        print(f"     {c} {names[c]:<12} census {pop[c]:>6,}  Kontur/ratio "
              f"{per.get(c, 0) / ratio:>8,.0f}")
    bad = [c for c in ("10405", "10408") if not 0.67 < per.get(c, 0) / ratio / pop[c] < 1.5]
    if bad:
        raise SystemExit(f"Puriscal witness fails on {bad}: recheck POLYGON_FOR")


if __name__ == "__main__":
    main()
