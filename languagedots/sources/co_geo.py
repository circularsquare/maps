"""Colombia placement layer: Kontur 400 m hexes keyed to the 1,122 municipios of the 2018 census,
each municipio split into the three zones sources/co_cnpv.py counts: resguardo (`<pcode>-r`),
cabecera (`-u`) and the rest (`-x`). See zones() for how hexes are assigned and folded.

    python sources/co_geo.py [--fetch]

Writes data/geo/co/co_hexes.gpkg (unit, mpio, zone, in_resg, pop) and
data/geo/co/co_zones.csv (census geo_id x zone -> unit, which countries/co.py reads).

UNITS. COD-AB Colombia adm2 (OCHA, from DANE's MGN, `date` 2018-01-01: the census's own
vintage), 1,122 polygons, ADM2_PCODE = "CO" + the DIVIPOLA code that REDATAM's MUPIO prints.
Read in place from religiondots/data/raw/co/ (read-only). The join is asserted both ways.

WHY SPLIT AT RESGUARDOS. Two thirds of the people who speak their people's language live in a
dwelling the census places in a resguardo indigena (VIVIENDA.UVA_ESTATER, sources/co_cnpv.py),
and the census counts every person on each side of that line per municipio. Without the split,
Arhuaco dots would sit in Valledupar's streets and Wayuu dots in Riohacha's. The resguardo
polygons are the Agencia Nacional de Tierras' "Resguardo Indígena Formalizado" (984 features,
updated 2026-09-07, CC BY-SA 4.0), fetched from its ArcGIS FeatureServer into data/raw/co/.
A hex is in the resguardo part when its centroid is inside any of them.

KONTUR is religiondots' CO extract (kontur_population_CO_20231101.gpkg, its data/raw/co/,
read-only), through sources/_grid.py's hex_layer and its checks against the census per municipio.

CHECKS: unit count and the code join both ways; _grid's outside-every-unit count, empty units,
Kontur-over-census band and shuffle null per municipio; and, per department, Kontur people in
the resguardo parts against the census's people in resguardos (printed, a witness that the ANT
polygons are the census's territories; ANT's 2026 layer holds resguardos formalised after 2018).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import _grid  # noqa: E402

RD_RAW = HERE.parent / "religiondots" / "data" / "raw" / "co"
COD_ZIP = RD_RAW / "col-administrative-divisions-shapefiles.zip"
KONTUR = RD_RAW / "kontur_population_CO_20231101.gpkg"
RAW = HERE / "data" / "raw" / "co"
RESG = RAW / "ant_resguardo_indigena_formalizado.geojson"
NORM = HERE / "data" / "normalized"
OUT = HERE / "data" / "geo" / "co" / "co_hexes.gpkg"

FS = ("https://utility.arcgis.com/usrsvcs/servers/8944116ccfd34a7189c4bc44b8e19186/rest/"
      "services/DatosAbiertos/Resguardo_Indigena_Formalizado/FeatureServer/0/query")
LOOKUP = HERE / "data" / "geo" / "co" / "co_zones.csv"
EXPECTED_UNITS = 1122
EXPECTED_RESGUARDOS = 984
HEX_KM2 = 0.74          # a Kontur r8 hex (religiondots playbook)
# A resguardo part is used only where the census's people in resguardos, spread evenly over
# its hexes, come to no more than this: above it, ANT's polygons in that municipio are too
# small for the census's territory (the colonial resguardos of Cauca and Narino, mostly), and
# using them would pile a whole resguardo population onto a few hexes.
R_MAX_DENSITY = 400


def fetch():
    import requests
    if RESG.exists():
        print("already have", RESG.name)
        return
    RAW.mkdir(parents=True, exist_ok=True)
    feats, off = [], 0
    while True:
        r = requests.get(FS, params={"where": "1=1", "outFields": "*", "outSR": 4326,
                                     "f": "geojson", "resultOffset": off,
                                     "resultRecordCount": 100},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
        r.raise_for_status()
        got = r.json().get("features", [])
        feats += got
        print(f"  {len(feats)} resguardos")
        if len(got) < 100:
            break
        off += 100
        time.sleep(0.5)
    if len(feats) != EXPECTED_RESGUARDOS:
        raise SystemExit(f"{len(feats)} resguardos, expected {EXPECTED_RESGUARDOS}")
    RESG.write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                    encoding="utf-8")


def main():
    if "--fetch" in sys.argv:
        fetch()
    import geopandas as gpd

    units = gpd.read_file(f"zip://{COD_ZIP}!col_admbnda_adm2_mgn_20200416.shp")
    units["unit"] = units["ADM2_PCODE"]
    if len(units) != EXPECTED_UNITS or units["unit"].duplicated().any():
        raise SystemExit(f"COD adm2: {len(units)} polygons, expected {EXPECTED_UNITS} unique")
    co = pd.read_csv(NORM / "co.csv", dtype={"geo_id": str})
    census = co.groupby("geo_id")["count"].sum()
    have, want = set(units["unit"]), set(census.index)
    if have != want:
        raise SystemExit(f"join: {len(want - have)} census municipios with no polygon "
                         f"{sorted(want - have)[:5]}, {len(have - want)} polygons with no row "
                         f"{sorted(have - want)[:5]}")
    print(f"  {len(units):,} COD-AB municipios join the census's {len(census):,} one to one")

    _grid.kontur_path = lambda cc: KONTUR      # religiondots' extract, read in place
    layer = _grid.hex_layer("co", units[["unit", "geometry"]], census=census.to_dict(),
                            out=OUT)

    resg = gpd.read_file(RESG)
    if len(resg) != EXPECTED_RESGUARDOS:
        raise SystemExit(f"{len(resg)} resguardos, expected {EXPECTED_RESGUARDOS}")
    resg = resg[resg.geometry.notna() & ~resg.geometry.is_empty].to_crs(3857)
    resg["geometry"] = resg.geometry.buffer(0)
    pts = gpd.GeoDataFrame(geometry=layer.to_crs(3857).geometry.centroid, crs=3857)
    j = gpd.sjoin(pts, resg[["geometry"]], how="left", predicate="within")
    in_resg = ~j["index_right"].isna().groupby(level=0).all()
    layer["mpio"] = layer["unit"]
    layer["in_resg"] = in_resg.reindex(layer.index).fillna(False).astype(bool).to_numpy()
    resguardo_witness(co, layer, units)

    layer, lookup = zones(co, layer)
    missing = sorted(set(census.index) - set(layer["mpio"]))
    if missing:   # no Kontur hex at all: the municipio's own polygon, at its census count
        extra = units[units["unit"].isin(missing)][["unit", "geometry"]].to_crs(4326)
        extra = extra.assign(mpio=extra["unit"], zone="all", in_resg=False,
                             pop=extra["unit"].map(census).astype(float))
        print(f"  {len(missing)} municipios with no Kontur hex drawn on their own polygon: "
              + ", ".join(f"{m} ({int(census[m]):,})" for m in missing))
        layer = pd.concat([layer, extra], ignore_index=True)
        lookup = pd.concat([lookup, pd.DataFrame([(m, z, m) for m in missing for z in "rux"],
                                                 columns=["geo_id", "zone", "unit"])])
    layer = gpd.GeoDataFrame(layer, geometry="geometry", crs=4326)
    layer[["unit", "mpio", "zone", "in_resg", "pop", "geometry"]].to_file(
        OUT, layer="hexes", driver="GPKG")
    lookup.to_csv(LOOKUP, index=False)
    print(f"  {layer['unit'].nunique():,} units; wrote {OUT.name} and {LOOKUP.name}")


def resguardo_witness(co, layer, units):
    """Per department: the census's share of people living in a resguardo against Kontur's
    share inside ANT's polygons. Printed, not asserted."""
    co = co.assign(dep=co["geo_id"].str[:4], r=co["zone"] == "r")
    cen = co.groupby(["dep", "r"])["count"].sum().unstack(fill_value=0)
    kon = layer.assign(dep=layer["mpio"].str[:4]).groupby(["dep", "in_resg"])["pop"].sum().unstack(fill_value=0)
    names = units.drop_duplicates("ADM1_PCODE").set_index("ADM1_PCODE")["ADM1_ES"]
    print("  share of people in resguardos, census 2018 against Kontur inside ANT's polygons:")
    for dep in cen.index:
        c = cen.loc[dep, True] / cen.loc[dep].sum() if True in cen.columns else 0
        k = kon.loc[dep, True] / kon.loc[dep].sum() if dep in kon.index and True in kon.columns else 0
        if c > 0.01 or k > 0.01:
            print(f"    {dep} {names.get(dep, '')[:22]:<22} census {100 * c:5.1f}%  Kontur {100 * k:5.1f}%")


def zones(co, layer):
    """Split each municipio's hexes into the three zones the census counts separately:
    r  hexes inside an ANT resguardo polygon, for the census's people in a resguardo, where the
       polygons can hold them (at most R_MAX_DENSITY people per km2 over their hexes); otherwise
       those people join x and the hexes join the rest;
    u  the cabecera: the municipio's densest remaining Kontur hexes, taken in order of density
       until they hold the census's cabecera share of the people outside resguardos;
    x  the rest of the remaining hexes, for centros poblados and rural disperso.
    A zone the census has people in but that gets no hex is folded into a neighbouring zone of
    the same municipio. -> (layer with `zone` and `unit`, lookup geo_id x zone -> unit)."""
    c = co.groupby(["geo_id", "zone"])["count"].sum().unstack(fill_value=0)
    for z in "rux":
        if z not in c.columns:
            c[z] = 0
    layer = layer.copy()
    layer["zone"] = ""
    rows, folds = [], {"r_to_x": 0, "r_people": 0, "u_to_x": 0, "x_to_u": 0, "to_r": 0}
    for m, idx in layer.groupby("mpio").groups.items():
        cr, cu, cx = (int(c.at[m, z]) if m in c.index else 0 for z in "rux")
        sub = layer.loc[idx]
        rmask = sub["in_resg"].to_numpy()
        target = {"r": "r", "u": "u", "x": "x"}
        nr = int(rmask.sum())
        if cr > 0 and nr > 0 and cr / (nr * HEX_KM2) <= R_MAX_DENSITY:
            r_idx = sub.index[rmask]
            rest = sub.index[~rmask]
        else:
            if cr > 0:
                folds["r_to_x"] += 1
                folds["r_people"] += cr
                target["r"] = "x"
            r_idx, rest = sub.index[:0], sub.index
        layer.loc[r_idx, "zone"] = "r"
        if len(rest) == 0:              # the whole municipio is resguardo
            folds["to_r"] += 1
            target.update(u="r", x="r")
        else:
            want_u = cu / (cu + cx + (cr if target["r"] == "x" else 0) or 1)
            order = layer.loc[rest, "pop"].sort_values(ascending=False)
            cum = order.cumsum() / max(order.sum(), 1e-9)
            n_u = int((cum < want_u).sum()) + 1 if cu > 0 else 0
            if len(order) > 1 and (cx > 0 or target["r"] == "x"):
                n_u = min(n_u, len(order) - 1)
            n_u = min(n_u, len(order))
            layer.loc[order.index[:n_u], "zone"] = "u"
            layer.loc[order.index[n_u:], "zone"] = "x"
            if cu > 0 and n_u == 0:
                target["u"] = "x"
                folds["u_to_x"] += 1
            if n_u == len(order) and (cx > 0 or target["r"] == "x"):
                target["x"] = "u"
                if target["r"] == "x":
                    target["r"] = "u"
                folds["x_to_u"] += 1
        for z in "rux":
            rows.append((m, z, f"{m}-{target[z]}"))
    layer["unit"] = layer["mpio"] + "-" + layer["zone"]
    lookup = pd.DataFrame(rows, columns=["geo_id", "zone", "unit"])
    print(f"  resguardo polygons too small for the census's people in {folds['r_to_x']} municipios "
          f"({folds['r_people']:,} people placed with the rural part instead); "
          f"{folds['to_r']} municipios all resguardo; {folds['u_to_x']} cabeceras and "
          f"{folds['x_to_u']} rural parts with no hex of their own, folded")
    return layer, lookup


if __name__ == "__main__":
    main()
