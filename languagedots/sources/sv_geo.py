"""El Salvador placement layer: Kontur 2023-11 400 m hexes keyed to the 262 distritos of the 2024
census -> data/geo/sv/sv_hexes.gpkg (`unit` = the census's six-digit distrito id, `pop`).

    python sources/sv_geo.py          (needs data/raw/sv/ and data/normalized/sv.csv from
                                       sources/sv_censo.py)

BOUNDARIES. The BCR's own distrito polygons, from the same feature service as the language
figures (`Limites_idiomas` layer 2, the 262 `Todos` features; sources/sv_censo.py fetches them).
So the census id IS the polygon id and there is no name join to get wrong. The distritos are the
262 pre-2024 municipalities; COD-AB El Salvador's admin2 is the same 262 but on alphabetical
pcodes, which religiondots found mispair the official order (religiondots/sources/sv.md), so it
is not used here.

THE WITNESSES, since a join on one shared id proves nothing by itself:
  1. Department. A distrito id's first two digits are its department in the OFFICIAL west-to-east
     order. religiondots' department hexes are keyed to COD's alphabetical pcodes, whose official
     number religiondots' lookup gives as LAPOP prov - 300 (sv_lookup.csv, name-joined there).
     Each hex's distrito must sit in the hex's religiondots department, border hexes aside; and
     the department names in the census table must equal the lookup's names.
  2. Area: the 262 polygons dissolved per department against religiondots' COD departments, IoU.
     The two are different tracings: IoU 0.86 (Cuscatlan) to 0.95, and 1.8% of Kontur people
     sit in a hex whose department differs between them. The bars (IoU 0.85, 3% of hexes and
     people) ask only that the departments are the same ones; the census counted on the BCR's
     lines, so those are the ones used.
  3. Kontur per distrito against the census population (all ages): band printed, and the log
     correlation asserted above its best over 500 shuffles of the same 262 values.

PLACEMENT. religiondots' sv_hexes.gpkg is plain Kontur SV keyed to department by hex centroid,
with the hexes outside every department already dropped (religiondots/sources/sv_grid.py); the
Kontur extract itself is no longer on disk. Its hexes are re-keyed here to the distrito holding
their centroid; a hex whose centroid lies in no distrito (the BCR's coast and lake lines differ
from COD's) goes to the nearest distrito of its own department, and the count and the largest
distance are printed.
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "sv" / "distritos.geojson"
NORM = HERE / "data" / "normalized" / "sv.csv"
RD_HEX = RD_GEO / "sv" / "sv_hexes.gpkg"
RD_DEP = RD_GEO / "sv" / "sv_departamentos.gpkg"
RD_LUT = RD_GEO / "sv" / "sv_lookup.csv"
OUT = HERE / "data" / "geo" / "sv" / "sv_hexes.gpkg"
N = 262
UTM = 32616


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    import geopandas as gpd
    from shapely.validation import make_valid

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    g = gpd.read_file(RAW)
    g["unit"] = g["id_distrito"].astype(str)
    check(len(g) == N and g["unit"].is_unique, f"{len(g)} distrito polygons, ids unique")
    g["geometry"] = g.geometry.map(make_valid)
    g = g.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    area = g.to_crs(UTM).area / 1e6
    check((area > 0).all(), f"every polygon has area (smallest {area.min():.1f} km2)")
    print(f"     total {area.sum():,.0f} km2 (El Salvador is 21,041 km2)")

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    pop = df[df["source_category"] == "Población"].set_index("geo_id")["count"]
    deptname = df.drop_duplicates("geo_id").set_index("geo_id")["departamento"]
    check(set(pop.index) == set(g["unit"]), "census and polygon ids join both ways")

    # witness 1a: department names, official number -> COD pcode through religiondots' lookup
    lut = pd.read_csv(RD_LUT)
    lut["official"] = (lut["lapop_prov"] - 300).map(lambda x: f"{x:02d}")
    off2cod = dict(zip(lut["official"], lut["unit"]))
    names = dict(zip(lut["official"], lut["name"]))
    bad = [(d, n, names.get(d)) for d, n in deptname.groupby(deptname.index.str[:2]).first().items()
           if fold(n) != fold(names.get(d))]
    check(not bad, f"the census's department names match religiondots' lookup on all 14 ({bad})")

    # witness 2: area, dissolved departments against COD's
    dep = gpd.read_file(RD_DEP).to_crs(UTM)
    mine = g.assign(cod=g["unit"].str[:2].map(off2cod)).to_crs(UTM).dissolve(by="cod")
    ious = {}
    for cod, geom in zip(dep["unit"], dep.geometry):
        a = mine.geometry.get(cod)
        ious[cod] = a.intersection(geom).area / a.union(geom).area
    worst = min(ious, key=ious.get)
    print("     IoU per department: " + ", ".join(f"{names[o]} {ious[c]:.3f}"
                                                for o, c in sorted(off2cod.items())))
    # COD's and the BCR's department lines are two tracings and disagree by several percent of
    # area (Chalatenango 1,950 km2 in COD against 2,021 here, Cuscatlan 678 against 748); the
    # census counted on the BCR's, so the bar only asks that they are the same departments
    print("     area per department, BCR against COD (km2): " + ", ".join(
        f"{names[o]} {mine.geometry[c].area / 1e6:,.0f}/{dep.set_index('unit').geometry[c].area / 1e6:,.0f}"
        for o, c in sorted(off2cod.items())))
    check(min(ious.values()) > 0.85, f"every department IoU over 0.85 (lowest {worst} {ious[worst]:.3f})")

    # placement: re-key religiondots' department hexes to distritos
    hx = gpd.read_file(RD_HEX)
    check(len(hx) > 0, f"religiondots' sv_hexes.gpkg: {len(hx):,} hexes")
    pts = gpd.GeoDataFrame({"pop": hx["pop"].to_numpy(dtype=float), "dep": hx["unit"].to_numpy()},
                           geometry=hx.geometry.to_crs(UTM).centroid, crs=UTM)
    gu = g.to_crs(UTM)
    j = gpd.sjoin(pts, gu[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["unit"].isna()
    print(f"     {len(hx):,} hexes, {pts['pop'].sum():,.0f} Kontur people; {int(out.sum())} hexes "
          f"({pts.loc[out, 'pop'].sum():,.0f} people) centred in no distrito")
    cod2off = {v: k for k, v in off2cod.items()}
    far = 0.0
    for i in np.flatnonzero(out.to_numpy()):
        cand = gu[gu["unit"].str[:2] == cod2off[pts["dep"].iat[i]]]
        d = cand.distance(pts.geometry.iat[i])
        j.iloc[i, j.columns.get_loc("unit")] = cand.loc[d.idxmin(), "unit"]
        far = max(far, d.min())
    print(f"     each given to the nearest distrito of its own department (farthest {far:,.0f} m)")
    check(far < 3000, "no snap over 3 km")

    # witness 1b: the hex's distrito is in the hex's religiondots department
    wrong = j["unit"].str[:2].map(off2cod) != pts["dep"]
    wp = pts.loc[wrong, "pop"].sum()
    print(f"     hexes whose distrito is in another department than religiondots': {int(wrong.sum())} "
          f"({wp:,.0f} people, {100 * wp / pts['pop'].sum():.2f}%)")
    check(wrong.mean() < 0.03 and wp < 0.03 * pts["pop"].sum(),
          "under 3% of hexes and of people (the two department lines differ; see witness 2)")

    layer = gpd.GeoDataFrame({"unit": j["unit"].to_numpy(), "pop": pts["pop"].to_numpy()},
                             geometry=hx.geometry.to_numpy(), crs=hx.crs)
    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(set(g["unit"]) - set(per.index[per > 0]))
    check(not empty, f"every distrito has a populated hex ({empty})")

    # witness 3: Kontur against the census per distrito
    c = pop.reindex(per.index).astype(float)
    ratio = per.sum() / c.sum()
    norm = (per / c / ratio).sort_values()
    nm = df.drop_duplicates("geo_id").set_index("geo_id")["geo_name"]
    print(f"     Kontur / census nationally {ratio:.3f}; per distrito normalised p10 "
          f"{norm.quantile(.1):.2f} median {norm.median():.2f} p90 {norm.quantile(.9):.2f}")
    print("     lowest: " + ", ".join(f"{nm[u]} {v:.2f}" for u, v in norm.head(5).items()))
    print("     highest: " + ", ".join(f"{nm[u]} {v:.2f}" for u, v in norm.tail(5).items()))
    print(f"     {int(((norm < 1 / 3) | (norm > 3)).sum())} distritos outside a factor of 3")
    lc, lk = np.log(c.to_numpy()), np.log(per.to_numpy())
    r = np.corrcoef(lc, lk)[0, 1]
    rng = np.random.default_rng(0)
    best = max(abs(np.corrcoef(lc, rng.permutation(lk))[0, 1]) for _ in range(500))
    check(r > best, f"log correlation r = {r:.3f} above the best of 500 shuffles, {best:.3f}")
    med = np.median(area.to_numpy()) / 0.74
    print(f"     median distrito {np.median(area):.1f} km2, about {med:.0f} Kontur hexes")

    if not ok:
        raise SystemExit("sv_geo: checks FAILED, nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name("sv_hexes.part.gpkg")
    layer.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(layer):,} hexes)")


if __name__ == "__main__":
    main()
