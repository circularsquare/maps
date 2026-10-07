"""Venezuela: placement units for the 2011 census parroquias, and Kontur hexes keyed to them.

    python sources/ve_geo.py

Writes
  data/geo/ve/ve_units.csv     census parroquia (geo_id, VE + 6 digits) -> unit
  data/geo/ve/ve_hexes.gpkg    Kontur 400 m hexes with `unit` and `pop`

READ-ONLY INPUTS, both religiondots':
  ../religiondots/data/raw/ve/ven_admin_boundaries.shp.zip   COD-AB Venezuela (INE, valid
      2021-02-23), ven_admin3 = 1,135 parroquias, read inside the zip
  ../religiondots/data/raw/ve/kontur_population_VE_20231101.gpkg   Kontur, 2023-11
religiondots' own ve_hexes.gpkg is keyed to state and holds no hexes in Amazonas or Delta Amacuro
(it draws neither: the survey it uses never sampled them), so it cannot be re-keyed; the Kontur
extract is.

THE JOIN. COD's adm3_pcode is "VE" + the census's six-digit entidad-municipio-parroquia code,
for 1,121 of the census's 1,128 parroquias. The rest are municipios whose parroquias changed
between the 2011 census and COD's 2021 file (Amazonas's autonomous municipios gained parroquias;
Simón Rodríguez in Anzoátegui split; Rojas and Sosa in Barinas gained one each; Anaco, Julio César
Salas, Francisco Javier Pulgar and the Dependencias Federales lost or renumbered some). In those
municipios codes no longer mean the same place, so the census parroquias and COD polygons are
both merged to the municipio, which is one unit `VE` + 4 digits. MERGED pins the list: the script
stops if a municipio joins differently.

THE WITNESS no key decides: Kontur's people per unit against the census's, as a log correlation
against 500 shuffles of the census figures among units OF THE SAME STATE (a right state with a
wrong parroquia would pass a national shuffle). Kontur is 2023 and the census 2011, with a large
emigration between, so the band is printed, not asserted.

Hexes whose centroid is in no unit are the extract's overrun into Colombia, Brazil, Guyana and
the sea, and are dropped (religiondots' ve_grid.py does the same and found the same). A unit
with no populated hex gets its own polygon at its census population.
"""
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RD_RAW = HERE.parent / "religiondots" / "data" / "raw" / "ve"
COD = "zip://" + (RD_RAW / "ven_admin_boundaries.shp.zip").as_posix() + "!ven_admin3.shp"
KONTUR = RD_RAW / "kontur_population_VE_20231101.gpkg"
NORM = HERE / "data" / "normalized" / "ve.csv"
OUT = HERE / "data" / "geo" / "ve"

COD_UNITS = 1_135
CENSUS_UNITS = 1_128
MERGED = {"VE0201", "VE0202", "VE0204", "VE0205", "VE0206", "VE0207", "VE0301", "VE0319",
          "VE0610", "VE0611", "VE1410", "VE2306", "VE2501"}


def main():
    cod = gpd.read_file(COD)
    if len(cod) != COD_UNITS:
        raise SystemExit(f"COD adm3 has {len(cod)} features, expected {COD_UNITS}")
    if cod["adm3_pcode"].duplicated().any() or cod.geometry.is_empty.any():
        raise SystemExit("COD adm3: duplicated pcode or empty geometry")
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    pop = df.groupby("geo_id")["count"].sum()
    if len(pop) != CENSUS_UNITS:
        raise SystemExit(f"ve.csv has {len(pop)} parroquias, expected {CENSUS_UNITS}")

    # municipios where the two code lists differ
    cen_by_m, cod_by_m = {}, {}
    for g in pop.index:
        cen_by_m.setdefault(g[:6], set()).add(g)
    for g in cod["adm3_pcode"]:
        cod_by_m.setdefault(g[:6], set()).add(g)
    if set(cen_by_m) != set(cod_by_m):
        raise SystemExit(f"municipios differ: census only {sorted(set(cen_by_m) - set(cod_by_m))}, "
                         f"COD only {sorted(set(cod_by_m) - set(cen_by_m))}")
    differ = {m for m in cen_by_m if cen_by_m[m] != cod_by_m[m]}
    if differ != MERGED:
        raise SystemExit(f"municipios whose parroquia codes differ: {sorted(differ)}; MERGED pins "
                         f"{sorted(MERGED)}")
    print(f"  COD {len(cod):,} parroquias, census {len(pop):,}; {len(MERGED)} municipios merged "
          f"({sum(len(cen_by_m[m]) for m in MERGED)} census parroquias, "
          f"{sum(len(cod_by_m[m]) for m in MERGED)} COD polygons)")

    unit_of = {g: (g[:6] if g[:6] in MERGED else g) for g in pop.index}
    cod["unit"] = cod["adm3_pcode"].map(lambda g: g[:6] if g[:6] in MERGED else g)
    units = cod.dissolve("unit", as_index=False)[["unit", "geometry"]]
    upop = pop.groupby(pop.index.map(unit_of)).sum()
    if set(units["unit"]) != set(upop.index):
        raise SystemExit("units and census differ after the merge")
    print(f"  {len(units):,} units, each one census row set and one polygon (bijection asserted)")

    # Kontur, centroids in its own CRS
    hexes = gpd.read_file(KONTUR)
    if len(hexes) == 0:
        raise SystemExit("Kontur VE extract has ZERO features")
    pts = gpd.GeoDataFrame({"pop": hexes["population"].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["unit"].isna()
    print(f"  Kontur VE: {len(hexes):,} hexes, {pts['pop'].sum():,.0f} people; {int(out.sum()):,} "
          f"hexes ({pts.loc[out, 'pop'].sum():,.0f} people) outside every unit, dropped")
    keep = ~out.to_numpy()
    layer = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].astype(str).to_numpy(),
                              "pop": pts.loc[keep, "pop"].to_numpy()},
                             geometry=hexes.geometry[keep].to_crs(units.crs).to_numpy(),
                             crs=units.crs)
    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(u for u in units["unit"] if per.get(u, 0) <= 0)
    if empty:
        add = units[units["unit"].isin(empty)].copy()
        add["pop"] = add["unit"].map(upop).astype(float)
        layer = pd.concat([layer, add[["unit", "pop", "geometry"]]], ignore_index=True)
        print(f"  {len(empty)} units with no populated hex get their own polygon at census "
              f"population: " + ", ".join(f"{u} ({int(upop[u]):,})" for u in empty))
    per = layer.groupby("unit")["pop"].sum()
    assert set(per.index[per > 0]) == set(units["unit"]), "a unit has no weight"

    # the witness: Kontur against the census per unit, shuffled within state
    rows = [(u, float(upop[u]), float(per[u])) for u in units["unit"] if upop[u] > 0]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    norm = sorted((k / c / ratio, u) for u, c, k in rows)
    n = len(norm)
    print(f"  Kontur / census nationally {ratio:.3f}; per unit, normalised: p10 "
          f"{norm[n // 10][0]:.2f}  median {norm[n // 2][0]:.2f}  p90 {norm[9 * n // 10][0]:.2f}; "
          f"{sum(1 for r, _ in norm if not 1 / 3 <= r <= 3)} of {n} outside a factor of 3")
    print("  lowest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[:5]))
    print("  highest: " + ", ".join(f"{u} {r:.2f}" for r, u in norm[-5:]))

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    lc = [math.log(c) for _, c, _ in rows]
    lk = [math.log(k) for _, _, k in rows]
    r = pear(lc, lk)
    state = [u[2:4] for u, _, _ in rows]
    rng = random.Random(0)
    best = 0.0
    for _ in range(500):
        sh = lc[:]
        for s in set(state):
            idx = [i for i, x in enumerate(state) if x == s]
            vals = [lc[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                sh[i] = v
        best = max(best, abs(pear(sh, lk)))
    print(f"  log correlation r = {r:.3f}; best of 500 within-state shuffles {best:.3f}")
    if r <= best:
        raise SystemExit("the parroquia join is not carrying information beyond the state")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"geo_id": list(unit_of), "unit": list(unit_of.values())}).to_csv(
        OUT / "ve_units.csv", index=False)
    layer.to_crs(4326).to_file(OUT / "ve_hexes.gpkg", layer="hexes", driver="GPKG")
    print(f"  wrote {OUT / 've_units.csv'} and {OUT / 've_hexes.gpkg'} ({len(layer):,} features)")


if __name__ == "__main__":
    main()
