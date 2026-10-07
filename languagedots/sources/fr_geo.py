"""France: the placement layer, data/geo/fr/fr_place.gpkg.

    python sources/fr_geo.py

religiondots' data/geo/fr/fr_lau.gpkg (read-only) is GISCO's 2021 communes with each commune's
French and foreign nationals from INSEE's RP 2021 TD_NAT1, plus Kontur 400 m hexes for the five
overseas départements. It leaves Corsica out (religiondots' survey does not reach it), so its
360 communes are added here from the same GISCO LAU 2021 shapefile and the same TD_NAT1 file.

Columns: lau, unit (the département: "01".."95", "2A", "2B", "971".."976"), nuts3, pop, french,
foreign, zone (the Basque survey's zone, or ""), w_basque (pop x the zone's first-language
share; 0 outside the Pays Basque). The counting unit is the département; this layer only
places dots inside it.
"""
import io
import os
import sys
import zipfile

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fr_regional as R  # noqa: E402

RD = os.path.join(os.path.dirname(ROOT), "religiondots")
RD_LAU = os.path.join(RD, "data", "geo", "fr", "fr_lau.gpkg")
LAU_SHP = os.path.join(RD, "data", "geo", "lau2021", "shp4326", "LAU_RG_01M_2021_4326.shp")
TD_NAT1 = os.path.join(RD, "data", "raw", "fr", "TD_NAT1_2021.csv")
EPCI_ZIP = os.path.join(ROOT, "data", "raw", "fr", "epci_2023.zip")
OUT = os.path.join(ROOT, "data", "geo", "fr", "fr_place.gpkg")

DOM_NUTS3 = {"FRY10": "971", "FRY20": "972", "FRY30": "973", "FRY40": "974", "FRY50": "976"}


def read_xlsx_no_styles(zpath):
    """INSEE's EPCI workbook has a stylesheet openpyxl refuses; drop that member and read."""
    with zipfile.ZipFile(zpath) as z:
        inner = z.read(z.namelist()[0])
    src = zipfile.ZipFile(io.BytesIO(inner))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as dst:
        for item in src.infolist():
            if item.filename != "xl/styles.xml":
                dst.writestr(item, src.read(item.filename))
    buf.seek(0)
    return pd.read_excel(buf, sheet_name="Composition_communale", skiprows=5, dtype=str)


def nationals():
    t = pd.read_csv(TD_NAT1, sep=";", dtype={"CODGEO": str}, encoding="latin-1")
    t = t[t["NIVGEO"] == "COM"]
    g = t.groupby(["CODGEO", "INATC"])["NB"].sum().unstack(fill_value=0.0)
    return g[1], g[2]


def main():
    base = gpd.read_file(RD_LAU)
    print(f"religiondots fr_lau.gpkg: {len(base):,} rows, {base['nuts3'].nunique()} NUTS 3")
    dom = base["nuts3"].isin(DOM_NUTS3)
    base["unit"] = base["lau"].str[:2]
    base.loc[dom, "unit"] = base.loc[dom, "nuts3"].map(DOM_NUTS3)
    if base.loc[~dom, "lau"].str.len().ne(5).any():
        sys.exit("!! a metropolitan LAU code is not five characters")

    # Corsica, from the same GISCO file religiondots read
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='FR'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip()
    co = g[g["lau"].str[:2].isin(["2A", "2B"])].copy()
    fr_, fo = nationals()
    co["pop"] = co["POP_2021"].astype(float)
    co["french"] = co["lau"].map(fr_)
    co["foreign"] = co["lau"].map(fo)
    miss = co["french"].isna().sum()
    print(f"Corsica: {len(co)} communes from GISCO, {miss} without a TD_NAT1 row")
    if len(co) != 360 or miss:
        sys.exit("!! expected Corsica's 360 communes, every one in TD_NAT1")
    co["unit"] = co["lau"].str[:2]
    co["nuts3"] = co["unit"].map({"2A": "FRM01", "2B": "FRM02"})
    co["nuts2"] = "FRM0"
    co["name"] = co["LAU_NAME"]
    keep = ["lau", "nuts3", "nuts2", "unit", "pop", "french", "foreign", "name", "geometry"]
    out = pd.concat([base[keep], co[keep].to_crs(base.crs)], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=base.crs)

    # unit <-> NUTS 3 is one to one
    pairs = out[["unit", "nuts3"]].drop_duplicates()
    if pairs["unit"].duplicated().any() or pairs["nuts3"].duplicated().any():
        sys.exit(f"!! unit and NUTS 3 are not 1:1: {pairs[pairs['unit'].duplicated(keep=False)]}")
    print(f"units: {out['unit'].nunique()} départements")
    if out["unit"].nunique() != 101:
        sys.exit("!! expected 101 départements (96 metropolitan, 5 overseas)")

    # Basque zones
    epci = read_xlsx_no_styles(EPCI_ZIP)
    capb = epci[epci["EPCI"] == R.CAPB_EPCI]
    print(f"Communaute d'agglomeration du Pays Basque: {len(capb)} communes (2023)")
    names = dict(zip(capb["LIBGEO"], capb["CODGEO"]))
    lost = [n for n in R.LAPURDI if n not in names]
    if lost:
        sys.exit(f"!! Labourd communes not in the CAPB list: {lost}")
    lap = {names[n] for n in R.LAPURDI}
    zone = {c: ("bab" if c in R.BAB else "lapurdi" if c in lap else "bn_zuberoa")
            for c in capb["CODGEO"]}
    out["zone"] = out["lau"].map(zone).fillna("")
    found = set(out.loc[out["zone"] != "", "lau"])
    if found != set(zone):
        sys.exit(f"!! CAPB communes not in the layer: {sorted(set(zone) - found)[:10]}")
    out["w_basque"] = out["pop"] * out["zone"].map(R.BASQUE_ZONES).fillna(0.0)
    zp = out[out["zone"] != ""].groupby("zone")["pop"].agg(["size", "sum"])
    print(f"Basque zones (communes, GISCO population):\n{zp}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp.gpkg"
    out.to_file(tmp, driver="GPKG", layer="place")
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(out):,} rows, population {out['pop'].sum():,.0f}")


if __name__ == "__main__":
    main()
