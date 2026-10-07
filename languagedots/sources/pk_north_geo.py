"""Pakistan's placement layer with Gilgit-Baltistan added -> data/geo/pk/pk_hexes.gpkg.

    python sources/pk_north_geo.py

religiondots' pk2023 hex layer (read-only) has the 146 districts it draws, Azad Kashmir's ten
among them, but not Gilgit-Baltistan, which it leaves blank. This writes a copy of that layer
with GB's ten census districts appended, so the four provinces, Islamabad and AJK are placed
exactly as religiondots places them.

GB'S UNITS. COD-AB v01 (religiondots/data/raw/pk2023/cod/pak_admin2.shp, valid_on 2022-09-09)
has GB as 14 districts; the 2023 census (GB at a Glance 2025 p.4) prints the pre-2019 ten. Darel
and Tangir were cut from Diamer, Gupis-Yasin from Ghizer and Rondu from Skardu in 2019, so those
four fold back into their parents (COD spells Diamer "Diamir"). Asserted both ways: every COD GB
district has a census district and every census district has a polygon.

GB'S HEXES. Kontur 2023-11-01 Pakistan, the file religiondots' layer was cut from, each hex given
to the GB district its centroid falls in; hexes religiondots' layer already holds (AJK's edge)
are not added twice.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
sys.path.insert(0, str(HERE))
from _grid import kontur_path  # noqa: E402
from rdlink import RD_GEO, RD  # noqa: E402
import pk_north  # noqa: E402

RD_LAYER = RD_GEO / "pk2023" / "pk_hexes.gpkg"
COD = RD / "data" / "raw" / "pk2023" / "cod" / "pak_admin2.shp"
OUT = HERE / "data" / "geo" / "pk" / "pk_hexes.gpkg"
OUT_UNITS = HERE / "data" / "geo" / "pk" / "pk_gb_districts.gpkg"

FOLD = {"Diamir": "Diamer", "Darel": "Diamer", "Tangir": "Diamer", "Gupis-Yasin": "Ghizer",
        "Rondu": "Skardu"}


def unit_id(d):
    return f"PK23-gilgit-baltistan/{d.lower()}-district"


def main():
    a2 = gpd.read_file(COD)
    gb = a2[a2["adm1_name"] == "Gilgit Baltistan"].copy()
    if len(gb) != 14:
        raise SystemExit(f"COD GB has {len(gb)} districts, expected 14")
    gb["district"] = gb["adm2_name"].map(lambda n: FOLD.get(n, n))
    census = set(pk_north.GB_POP)
    if set(gb["district"]) != census:
        raise SystemExit(f"COD -> census: {sorted(set(gb['district']) ^ census)}")
    units = gb.dissolve(by="district").reset_index()[["district", "geometry"]]
    units["unit"] = units["district"].map(unit_id)
    print(f"  COD GB 14 districts -> {len(units)} census districts: "
          + ", ".join(f"{k}->{v}" for k, v in FOLD.items()))

    rd = gpd.read_file(RD_LAYER)
    print(f"  religiondots pk2023 layer: {len(rd):,} hexes, {rd['unit'].nunique()} units, "
          f"{rd['pop'].sum():,.0f} people, crs {rd.crs}")

    hexes = gpd.read_file(kontur_path("pk"))
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    cen = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(float)}, geometry=cen,
                           crs=hexes.crs).to_crs(units.crs)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    # hexes religiondots already has: match on the hex centroid in Kontur's metres, to 1 m
    rd_c = rd.to_crs(hexes.crs).geometry.centroid
    have = set(zip(np.round(rd_c.x).astype(np.int64), np.round(rd_c.y).astype(np.int64)))
    c = cen.loc[j.index]
    dup = np.array([(int(round(x)), int(round(y))) in have for x, y in zip(c.x, c.y)])
    print(f"  Kontur hexes in GB districts: {len(j):,} ({j['pop'].sum():,.0f} people); "
          f"{dup.sum():,} already in religiondots' layer ({j.loc[dup, 'pop'].sum():,.0f} people), "
          f"not added again")
    j = j[~dup]
    new = gpd.GeoDataFrame({"unit": j["unit"].to_numpy(), "pop": j["pop"].to_numpy()},
                           geometry=hexes.geometry.loc[j.index].to_numpy(),
                           crs=hexes.crs).to_crs(rd.crs)

    per = new.groupby("unit")["pop"].sum()
    pop = pd.Series({unit_id(d): m + f for d, (m, f) in pk_north.GB_POP.items()})
    t = pd.DataFrame({"census": pop, "kontur": per}).fillna(0)
    t["ratio"] = t["kontur"] / t["census"]
    print("  Kontur / census 2023 per GB district:")
    print(t.round(3).to_string())
    ratio = t["kontur"].sum() / t["census"].sum()
    print(f"  GB overall {ratio:.3f}")
    if (t["kontur"] <= 0).any():
        raise SystemExit("a GB district has no populated hex")
    if not 0.5 <= ratio <= 1.5:
        raise SystemExit("GB Kontur/census out of band")
    lc, lk = np.log(t["census"]), np.log(t["kontur"])
    r = np.corrcoef(lc, lk)[0, 1]
    rng = np.random.default_rng(0)
    best = max(abs(np.corrcoef(lc, rng.permutation(lk.to_numpy()))[0, 1]) for _ in range(500))
    print(f"  log correlation r = {r:.3f}; best of 500 shuffles {best:.3f}")

    out = pd.concat([rd, new], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=rd.crs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.gpkg")
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    units.to_crs(4326).to_file(OUT_UNITS, layer="districts", driver="GPKG")
    print(f"  wrote {OUT} ({len(out):,} hexes, {out['unit'].nunique()} units) and {OUT_UNITS.name}")


if __name__ == "__main__":
    main()
