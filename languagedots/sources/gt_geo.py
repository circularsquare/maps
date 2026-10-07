"""Guatemala placement layer: Kontur 2023-11 400 m hexes keyed to the 340 municipios of the 2018
census -> data/geo/gt/gt_hexes.gpkg.

    python sources/gt_geo.py          (needs data/normalized/gt_units.csv from sources/gt_censo.py)

BOUNDARIES. OCHA COD-AB Guatemala (valid_on 2019-02-07), admin2, read from religiondots' download
(religiondots/data/raw/gt/shp/, read-only; religiondots draws Guatemala by department and has no
municipio layer). 342 features: the 340 municipios plus Lago de Amatitlan (GT0100) and Lago de
Atitlan (GT0700), which are water and are left out of the units.

THE JOIN is on two keys, both asserted. COD's pcode is GT + INE's 4-digit municipio code, and the
census (REDATAM's AREABREAK) prints each code with its name; the codes must match 340 for 340 in
both directions, and on every code the two names must agree once accents, case and punctuation are
folded, except for the spellings pinned in NAME_PINNED (each read and checked by hand). Departments
are a third witness: a municipio code's first two digits are its department, and COD's adm1_pcode
must say the same.

THE NAME WITNESS CAUGHT A SWAP. COD-AB gives GT0206 to Sanarate and GT0207 to Sansare; INE's census
has 0206 Sansare and 0207 Sanarate. The codes join 340 for 340 and every total passes either way.
The polygons follow their names, not their codes: COD's "Sanarate" is 274 km2 with 40,553 Kontur
people, its "Sansare" 144 km2 with 13,574, and the census counts 39,444 in its Sanarate (0207) and
13,154 in its Sansare (0206). So those two census codes take the polygon of the same NAME
(POLYGON_FOR), and the Kontur witness is asserted. INE's code is the unit id throughout.

PLACEMENT is religiondots' recipe (sources/_grid.py's hex_layer): each Kontur hex goes to the
municipio its centroid falls in. Hexes whose centroid falls in one of the two lakes go to the
nearest municipio instead (shore hexes at Panajachel, Santiago Atitlan, Villa Canales); hexes
outside every municipio and outside the lakes (Kontur's GT extract runs into Mexico, Belize,
Honduras, El Salvador and the sea) are dropped, and their people printed.

CHECKS: 340 units, the code and name join, the department witness, every municipio has a
populated hex, and Kontur per municipio against the census's all-ages population (band and the
shuffled-join control, printed).
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

SHP = RD / "data" / "raw" / "gt" / "shp" / "gtm_admin2.shp"
UNITS = HERE / "data" / "normalized" / "gt_units.csv"
OUT = HERE / "data" / "geo" / "gt" / "gt_hexes.gpkg"
LAKES = {"GT0100", "GT0700"}
N = 340

# census code -> COD pcode, where COD numbers a polygon differently from INE (see the docstring)
POLYGON_FOR = {"GT0206": "GT0207", "GT0207": "GT0206"}

# code: (census name, COD name), each looked at: the same municipio, the census's full name against
# COD's short one (saint's name dropped), or a spelling
NAME_PINNED = {
    "GT0111": ("San Raymundo", "San Raimundo"),
    "GT0117": ("San Miguel Petapa", "Petapa"),
    "GT0314": ("San Juan Alotenango", "Alotenango"),
    "GT0404": ("San Juan Comalapa", "Comalapa"),
    "GT0408": ("San Miguel Pochuta", "Pochuta"),
    "GT0412": ("San Pedro Yepocapa", "Yepocapa"),
    "GT0808": ("San Bartolo Aguas Calientes", "San Bartolo"),
    "GT0903": ("San Juan Olintepeque", "Olintepeque"),
    "GT0909": ("San Juan Ostuncalco", "Ostuncalco"),
    "GT0917": ("Colomba Costa Cuca", "Colomba"),
    "GT1214": ("San José el Rodeo", "El Rodeo"),
    "GT1308": ("San Pedro Soloma", "Soloma"),
    "GT1309": ("San Ildefonso Ixtahuacán", "Ixtahuacán"),
    "GT1326": ("Santa Cruz Barillas", "Barillas"),
    "GT1406": ("Santo Tomás Chichicastenango", "Chichicastenango"),
    "GT1413": ("Santa María Nebaj", "Nebaj"),
    "GT1415": ("San Miguel Uspantán", "Uspantán"),
    "GT1420": ("Playa Grande Ixcán", "Ixcán"),
    "GT1506": ("Santa Cruz El Chol", "El Chol"),
    "GT1606": ("San Miguel Tucurú", "Tucurú"),
    "GT1611": ("San Agustín Lanquín", "Lanquín"),
    "GT1612": ("Santa María Cahabón", "Cahabón"),
}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    import geopandas as gpd
    from _grid import hex_layer

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    g = gpd.read_file(SHP, engine="fiona", encoding="utf-8")
    check(len(g) == N + len(LAKES), f"COD-AB admin2 has {len(g)} features (340 municipios + 2 lakes)")
    lakes = g[g["adm2_pcode"].isin(LAKES)]
    check(len(lakes) == 2 and all("lago" in fold(n) for n in lakes["adm2_name"]),
          f"the two lakes are {list(lakes['adm2_name'])}")
    g = g[~g["adm2_pcode"].isin(LAKES)].copy()
    back = {v: k for k, v in POLYGON_FOR.items()}
    g["unit"] = g["adm2_pcode"].astype(str).map(lambda c: back.get(c, c))
    check(g["unit"].is_unique and len(g) == N, f"{len(g)} municipio polygons, pcodes unique")

    u = pd.read_csv(UNITS, dtype={"geo_id": str})
    cen, cod = set(u["geo_id"]), set(g["unit"])
    check(cen == cod, f"codes join both ways ({len(cen - cod)} census-only {sorted(cen - cod)[:4]}, "
                      f"{len(cod - cen)} COD-only {sorted(cod - cen)[:4]})")
    m = u.merge(g[["unit", "adm2_name", "adm1_pcode"]], left_on="geo_id", right_on="unit")
    differ = [(r.geo_id, r.geo_name, r.adm2_name) for r in m.itertuples()
              if fold(r.geo_name) != fold(r.adm2_name)]
    unpinned = [d for d in differ if NAME_PINNED.get(d[0]) != (d[1], d[2])]
    for d in differ:
        print(f"     name differs on {d[0]}: census {d[1]!r}, COD {d[2]!r}"
              + ("" if d in unpinned else "  (pinned)"))
    check(not unpinned, f"names agree on every code once folded, but for {len(NAME_PINNED)} pinned "
                        f"spellings ({len(unpinned)} unpinned differences)")
    dept = [r.geo_id for r in m.itertuples() if r.adm1_pcode != r.geo_id[:4]]
    check(not dept, f"every municipio's code prefix is its COD department ({dept[:4]})")
    if not ok:
        raise SystemExit("gt_geo: join checks FAILED, nothing written")

    # lake hexes: give them to the nearest municipio before hex_layer sees the units, by making
    # each lake part of its nearest shore unit for the centroid join only (a projected CRS, metres)
    units = g[["unit", "geometry"]].reset_index(drop=True)
    utm = units.to_crs(32615)
    lk = lakes.to_crs(32615)
    from _grid import kontur_path
    hx = gpd.read_file(kontur_path("gt"))
    pts = gpd.GeoDataFrame(geometry=hx.geometry.centroid, crs=hx.crs).to_crs(32615)
    in_lake = gpd.sjoin(pts, lk[["geometry"]], how="inner", predicate="within")
    pop_col = next(c for c in hx.columns if c.lower() == "population")
    lake_pts = pts.loc[in_lake.index.unique()]
    near = gpd.sjoin_nearest(lake_pts, utm, how="left", distance_col="d")
    near = near[~near.index.duplicated(keep="first")]
    print(f"     {len(lake_pts)} hexes centred in the lakes ({hx.loc[lake_pts.index, pop_col].sum():,.0f} "
          f"Kontur people), each given to its nearest municipio (at most {near['d'].max():,.0f} m)")
    # enlarge each receiving unit by small discs round those centroids, so the centroid join
    # inside hex_layer finds them; the discs lie inside the lake, which no unit owns
    add = gpd.GeoDataFrame({"unit": near["unit"].to_numpy()},
                           geometry=lake_pts.loc[near.index].buffer(5).to_numpy(), crs=32615)
    grown = pd.concat([utm, add], ignore_index=True).dissolve(by="unit", as_index=False)
    grown = grown.to_crs(4326)
    check(len(grown) == N, f"{len(grown)} units after the lake hexes are added")

    census = dict(zip(u["geo_id"], u["population"]))
    layer = hex_layer("gt", grown, census=census)
    check(set(layer["unit"]) == cod, "every municipio has hexes in the layer")
    per = layer.groupby("unit")["pop"].sum()
    check((per > 0).all(), f"every municipio has a populated hex ({int((per <= 0).sum())} do not)")
    # the swap witness: on the name join each of the pair sits near the national Kontur/census
    # ratio; on COD's codes each would be off by a factor of about three
    nat = per.sum() / sum(census.values())
    for c in POLYGON_FOR:
        r = per[c] / census[c] / nat
        check(0.67 <= r <= 1.5, f"{c} {u.set_index('geo_id').at[c, 'geo_name']}: Kontur/census "
                                f"{r:.2f} of the national ratio on the name join")
    if not ok:
        raise SystemExit("gt_geo: FAILED (the layer was written; do not use it)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
