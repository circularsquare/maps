"""Mexico placement layer: Kontur 400 m hexes keyed to the 2,469 municipios of the 2020 census,
each carrying an estimate of the indigenous-language speakers living in it (`spk`).

    python sources/mx_geo.py

Writes data/geo/mx/mx_hexes.gpkg (unit, pop, spk).

UNITS. INEGI's Marco Geoestadistico 2020 (the census's own geography), layer 00mun, 2,469
polygons, CVEGEO = entidad + municipio, read in place from religiondots' data/geo/mx/mg2020/
(read-only; religiondots' sources/mx_geo.md has the fetch). The join to the census is asserted
both ways.

KONTUR. kontur_population_MX_20231101 through sources/_grid.py's hex_layer and its checks.

SPEAKERS PER HEX. The language table is per municipio, but ITER 2020 (religiondots' copy, read-only)
gives every locality's speakers aged 3+ (P3YM_HLI) and its coordinates, so inside a municipio the
indigenous-language dots can go to the villages where the speakers are and the Spanish dots to the
rest. Every locality's speakers are put on hexes:
  * a locality with a polygon in the Marco's 00l layer (every urban locality and the rural ones
    with blocks, 50,308) spreads its speakers over the hexes whose centroid lies inside, by Kontur
    population; a polygon holding no hex centroid uses the nearest hex of its municipio;
  * any other locality puts them on the nearest hex of its municipio to its ITER point;
  * a masked locality ("*", INEGI's confidentiality rule for the smallest) gets the municipio's
    speakers not otherwise accounted for, shared by its POBTOT; and ITER's two roll-up rows per
    municipio (localities of one and of two dwellings, no coordinates) are spread over the
    municipio's hexes by Kontur population.
`spk` sums to each municipio's P3YM_HLI (asserted), so to the census's speakers. It is a placement
weight only: countries/mx.py takes the counts per language from the cube either way.
"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import _grid  # noqa: E402

RD_MX = HERE.parent / "religiondots" / "data"
MG = RD_MX / "geo" / "mx" / "mg2020" / "conjunto_de_datos"
ITER = RD_MX / "raw" / "mx" / "iter_00_cpv2020_csv.zip"
OUT = HERE / "data" / "geo" / "mx" / "mx_hexes.gpkg"


def _dms(s):
    m = re.match(r"\s*(\d+)\D+(\d+)\D+([\d.]+)\D*([NSEW])", str(s))
    if not m:
        return np.nan
    v = int(m.group(1)) + int(m.group(2)) / 60 + float(m.group(3)) / 3600
    return -v if m.group(4) in "SW" else v


def read_iter():
    z = zipfile.ZipFile(ITER)
    name = next(n for n in z.namelist() if n.endswith("conjunto_de_datos_iter_00CSV20.csv"))
    df = pd.read_csv(z.open(name), dtype=str, encoding="utf-8",
                     usecols=["ENTIDAD", "MUN", "LOC", "LONGITUD", "LATITUD", "POBTOT", "P3YM_HLI"])
    df = df[df["MUN"] != "000"].copy()
    df["unit"] = df["ENTIDAD"] + df["MUN"]
    df["POBTOT"] = df["POBTOT"].astype(int)
    return df


def main():
    import geopandas as gpd
    from scipy.spatial import cKDTree

    units = gpd.read_file(MG / "00mun.shp")[["CVEGEO", "geometry"]].rename(columns={"CVEGEO": "unit"})
    it = read_iter()
    mun = it[it["LOC"] == "0000"].set_index("unit")
    mun_hli = mun["P3YM_HLI"].astype(int)
    assert len(units) == 2469 and units["unit"].is_unique
    a, b = set(units["unit"]), set(mun.index)
    assert a == b, f"00mun against ITER: {sorted(a - b)[:5]} {sorted(b - a)[:5]}"
    print(f"  units: 2,469 municipios, joined to ITER both ways")

    if OUT.exists():
        hexes = gpd.read_file(OUT)
        print(f"  {OUT.name} exists ({len(hexes):,} hexes); recomputing spk only")
    else:
        hexes = _grid.hex_layer("mx", units, census=mun["POBTOT"].to_dict(), out=OUT)
    hexes = hexes.to_crs(units.crs)
    hexes["unit"] = hexes["unit"].astype(str)
    cen = hexes.geometry.centroid
    hx, hy = cen.x.to_numpy(), cen.y.to_numpy()
    hpop = hexes["pop"].to_numpy(dtype=float)
    hunit = hexes["unit"].to_numpy()
    spk = np.zeros(len(hexes))
    by_unit = {u: np.flatnonzero(hunit == u) for u in np.unique(hunit)}
    trees = {u: cKDTree(np.c_[hx[i], hy[i]]) for u, i in by_unit.items()}

    def nearest(u, x, y):
        d, j = trees[u].query([x, y])
        return by_unit[u][j]

    def spread(idx, n):
        p = hpop[idx]
        spk[idx] += n * (p / p.sum() if p.sum() > 0 else np.full(len(idx), 1 / len(idx)))

    loc = it[~it["LOC"].isin(["0000", "9998", "9999"])].copy()
    loc["masked"] = ~loc["P3YM_HLI"].str.fullmatch(r"\d+")
    loc["hli"] = pd.to_numeric(loc["P3YM_HLI"], errors="coerce").fillna(0)
    roll = it[it["LOC"].isin(["9998", "9999"])].copy()
    roll["hli"] = pd.to_numeric(roll["P3YM_HLI"], errors="coerce").fillna(0)
    # masked localities share what their municipio's known localities and roll-ups leave over
    known = loc.groupby("unit")["hli"].sum().add(roll.groupby("unit")["hli"].sum(), fill_value=0)
    resid = (mun_hli - known.reindex(mun_hli.index).fillna(0))
    assert (resid >= -0.5).all(), f"known speakers exceed the municipio in {list(resid[resid < -0.5].index[:5])}"
    mpop = loc[loc["masked"]].groupby("unit")["POBTOT"].sum()
    rolled_spare = 0.0
    for u, r in resid[resid > 0].items():
        if mpop.get(u, 0) > 0:
            continue
        rolled_spare += r                       # no masked locality to carry it: municipio-wide
    m = loc["masked"]
    loc.loc[m, "hli"] = loc.loc[m, "POBTOT"] * loc.loc[m, "unit"].map(
        (resid.clip(lower=0) / mpop).replace([np.inf], 0)).fillna(0)
    print(f"  ITER: {len(loc):,} localities, {int(m.sum()):,} masked carrying "
          f"{loc.loc[m, 'hli'].sum():,.0f} estimated speakers; roll-ups {roll['hli'].sum():,.0f}; "
          f"{rolled_spare:,.0f} with no masked locality to carry them, spread municipio-wide")

    # localities with a polygon
    lpol = gpd.read_file(MG / "00l.shp")[["CVEGEO", "geometry"]]
    loc["cvegeo"] = loc["ENTIDAD"] + loc["MUN"] + loc["LOC"]
    haspol = loc["cvegeo"].isin(set(lpol["CVEGEO"]))
    pts = gpd.GeoDataFrame({"hi": np.arange(len(hexes))},
                           geometry=gpd.points_from_xy(hx, hy), crs=units.crs)
    j = gpd.sjoin(pts, lpol, how="inner", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    hex_of = j.groupby("CVEGEO")["hi"].apply(np.array).to_dict()
    lp = lpol.set_index("CVEGEO").geometry.representative_point()
    n_poly = n_poly_empty = n_pt = 0
    for r in loc[haspol & (loc["hli"] > 0)].itertuples():
        idx = hex_of.get(r.cvegeo)
        if idx is not None:
            idx = idx[hunit[idx] == r.unit]
        if idx is None or len(idx) == 0:
            p = lp[r.cvegeo]
            spk[nearest(r.unit, p.x, p.y)] += r.hli
            n_poly_empty += 1
        else:
            spread(idx, r.hli)
            n_poly += 1
    # point localities: ITER lon/lat into the units' CRS
    rest = loc[~haspol & (loc["hli"] > 0)].copy()
    rest["lon"] = rest["LONGITUD"].map(_dms)
    rest["lat"] = rest["LATITUD"].map(_dms)
    nocoord = rest["lon"].isna() | rest["lat"].isna()
    p = gpd.GeoSeries(gpd.points_from_xy(rest["lon"].fillna(0), rest["lat"].fillna(0)),
                      crs=4326).to_crs(units.crs)
    for (u, h, bad), x, y in zip(rest[["unit", "hli"]].assign(b=nocoord).itertuples(index=False),
                                 p.x, p.y):
        if bad:
            spread(by_unit[u], h)
        else:
            spk[nearest(u, x, y)] += h
        n_pt += 1
    # roll-ups and the uncarried remainder: municipio-wide by Kontur population
    extra = roll.groupby("unit")["hli"].sum()
    for u, r in resid[resid > 0].items():
        if mpop.get(u, 0) <= 0:
            extra[u] = extra.get(u, 0) + r
    for u, n in extra[extra > 0].items():
        spread(by_unit[u], n)
    print(f"  placed: {n_poly:,} polygon localities, {n_poly_empty:,} on the nearest hex "
          f"(no hex centroid inside), {n_pt:,} point localities ({int(nocoord.sum())} without "
          f"coordinates, municipio-wide)")

    got = pd.Series(spk).groupby(hunit).sum()
    diff = (got - mun_hli.reindex(got.index)).abs()
    assert diff.max() < 0.5, f"spk does not sum to P3YM_HLI: {diff.sort_values().tail(5).to_dict()}"
    print(f"  spk sums to every municipio's P3YM_HLI (total {spk.sum():,.0f})")
    hexes["spk"] = spk
    hexes.to_crs(4326).to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(hexes):,} hexes, unit, pop, spk)")


if __name__ == "__main__":
    main()
