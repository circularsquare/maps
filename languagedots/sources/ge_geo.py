"""Georgia: the 64 self-governed units of the 2024 census, and a Kontur placement layer on them.

    python sources/ge_geo.py      -> data/geo/ge/ge_units.gpkg, data/geo/ge/ge_hexes.gpkg

religiondots draws Georgia on 11 regions (its religion table goes no finer), so its placement
layer is no use here; this builds units for the 64 self-governed cities and municipalities of
the 2024 native-language table and re-keys the same Kontur GE extract to them.

BOUNDARIES: COD-AB Georgia on HDX (cod-ab-geo, CC BY-IGO, valid_on 2019-10-18;
data/raw/ge/geo_admin_boundaries.geojson.zip, fetched 2026-10-04). Chosen over geoBoundaries, the
other candidate, because:
  * geoBoundaries GEO ADM2 (68 features) has no Batumi, Kutaisi, Poti or Rustavi of its own (they
    sit inside the municipalities around them), and draws Tbilisi at 329 km2; COD-AB draws all four
    cities and Tbilisi (its ADM1 GE11) at 502 km2, the city's own figure.
  * COD-AB's ADM2 covers only territory under the government's control: Abkhazia (GE12) and the
    "Provisional Administration" of the Tskhinvali region (GE48) are ADM1 with no ADM2, which is
    exactly the ground the census did not enumerate ("Does not include occupied territories").

COD-AB's 70 ADM2 are the 2014 map: seven towns were self-governing cities then and were merged
back into their municipalities in 2017 (Ozurgeti, Telavi, Mtskheta, Ambrolauri, Zugdidi,
Akhaltsikhe, Gori). The 2024 table has the merged municipalities, so each `<X> City` polygon is
dissolved into `<X>` (pinned in MERGED_2017 by pcode, asserted). 70 - 7 = 63, plus Tbilisi = 64.

THE JOIN is by name inside region: the table's `Keda Municipality` / `C. Batumi` against COD's
`Keda` / `Batumi`, two spellings pinned (ALIAS), and each pair must also agree on its region (the
table's own region row above it against COD's adm1_name), which is the witness the name does not
decide. Asserted both ways: 64 matched, nothing left on either side.

PLACEMENT: `_grid.hex_layer` (Kontur 2023 r8, each hex to the unit its centroid falls in), which
prints people outside every unit, Kontur against the census per unit and the shuffled-join
control.
"""
import io
import os
import re
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "ge", "geo_admin_boundaries.geojson.zip")
NORM = os.path.join(ROOT, "data", "normalized", "ge.csv")
GEO = os.path.join(ROOT, "data", "geo", "ge")
UNITS_OUT = os.path.join(GEO, "ge_units.gpkg")
COD_URL = ("https://data.humdata.org/dataset/3ee95199-2dfe-40fc-b9bd-b44cc8c91024/resource/"
           "e5f2125c-d2aa-49ee-a7d1-1fa27fb030df/download/geo_admin_boundaries.geojson.zip")

EXPECTED_ADM2 = 70
EXPECTED_UNITS = 64
# 2014's self-governing cities merged back into their municipalities in 2017: city -> municipality
MERGED_2017 = {"GE2311": "GE2325",   # Ozurgeti
               "GE2911": "GE2930",   # Telavi
               "GE3211": "GE3229",   # Mtskheta
               "GE3511": "GE3523",   # Ambrolauri
               "GE3811": "GE3825",   # Zugdidi
               "GE4111": "GE4129",   # Akhaltsikhe
               "GE4711": "GE4724"}   # Gori
TBILISI_ADM1 = "GE11"
NOT_ENUMERATED_ADM1 = {"GE12", "GE48"}      # Abkhazia, Provisional Administration (Tskhinvali)
ALIAS = {"sighnaghi": "sighnagi"}           # COD spelling -> the table's
REGION_ALIAS = {"autonomousrepublicofadjara": "adjaraar"}   # the table writes "Adjara A.R."


def fold(s):
    s = re.sub(r"\s+Municipality$", "", str(s).strip())
    s = re.sub(r"^C\.\s*", "", s)
    s = re.sub(r"[^A-Za-z]+", "", s).lower()
    return ALIAS.get(s, s)


def rfold(s):
    s = re.sub(r"^C\.\s*", "", str(s).strip())
    s = re.sub(r"[^A-Za-z]+", "", s).lower()
    return REGION_ALIAS.get(s, s)


def fetch():
    import requests
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    if os.path.exists(RAW) and os.path.getsize(RAW) > 5_000_000:
        print("already have", RAW)
        return
    r = requests.get(COD_URL, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if r.content[:2] != b"PK":
        raise SystemExit("COD-AB download is not a zip")
    open(RAW, "wb").write(r.content)


def build_units():
    import geopandas as gpd
    import pandas as pd

    z = zipfile.ZipFile(RAW)
    a2 = gpd.read_file(io.BytesIO(z.read("geo_admin2.geojson"))).to_crs(4326)
    a1 = gpd.read_file(io.BytesIO(z.read("geo_admin1.geojson"))).to_crs(4326)
    if len(a2) != EXPECTED_ADM2:
        raise SystemExit(f"COD-AB ADM2 has {len(a2)} features, expected {EXPECTED_ADM2}")
    if set(a2["adm1_pcode"]) & NOT_ENUMERATED_ADM1:
        raise SystemExit("COD-AB ADM2 now has units in Abkhazia or the Tskhinvali region")
    for city, muni in MERGED_2017.items():
        c = a2.loc[a2["adm2_pcode"] == city, "adm2_name"]
        m = a2.loc[a2["adm2_pcode"] == muni, "adm2_name"]
        if len(c) != 1 or len(m) != 1 or c.iloc[0] != f"{m.iloc[0]} City":
            raise SystemExit(f"MERGED_2017 {city}->{muni}: {list(c)} / {list(m)}")
    a2["key_pcode"] = a2["adm2_pcode"].map(lambda p: MERGED_2017.get(p, p))
    d = a2.dissolve("key_pcode", aggfunc="first").reset_index()
    d["name"] = d["adm2_name"]
    d["region"] = d["adm1_name"]
    tb = a1[a1["adm1_pcode"] == TBILISI_ADM1].copy()
    tb["key_pcode"], tb["name"], tb["region"] = TBILISI_ADM1, "Tbilisi", "Tbilisi"
    cod = pd.concat([d[["key_pcode", "name", "region", "geometry"]],
                     tb[["key_pcode", "name", "region", "geometry"]]], ignore_index=True)
    cod = gpd.GeoDataFrame(cod, crs=4326)
    print(f"  COD-AB: {len(a2)} ADM2, {len(MERGED_2017)} 2014 cities merged back, plus Tbilisi "
          f"-> {len(cod)} units")
    if len(cod) != EXPECTED_UNITS:
        raise SystemExit(f"{len(cod)} COD units, expected {EXPECTED_UNITS}")

    df = pd.read_csv(NORM)
    u = df[(df["geo_level"] == "unit") & (df["source_category"] == "Total")].copy()
    u["region"] = u["note"].str.extract(r"region=([^;]+)")[0]
    u["key"] = u["geo_name"].map(fold)
    cod["key"] = cod["name"].map(fold)
    for label, s in (("census", u["key"]), ("COD", cod["key"])):
        if s.duplicated().any():
            raise SystemExit(f"{label} keys repeat: {sorted(s[s.duplicated()])}")
    only_c = sorted(set(u["key"]) - set(cod["key"]))
    only_g = sorted(set(cod["key"]) - set(u["key"]))
    print(f"  name join: {len(set(u['key']) & set(cod['key']))} matched; census only {only_c}; "
          f"COD only {only_g}")
    if only_c or only_g:
        raise SystemExit("the name join is not a bijection")
    m = cod.merge(u[["key", "geo_id", "geo_name", "region", "count"]], on="key",
                  suffixes=("_cod", "_census"))
    wrong = m[m["region_cod"].map(rfold) != m["region_census"].map(rfold)]
    if len(wrong):
        raise SystemExit("matched names in different regions (a wrong twin?):\n"
                         + wrong[["name", "region_cod", "region_census"]].to_string())
    print(f"  OK every pair also agrees on its region ({len(m)} of {len(m)})")
    m = m.rename(columns={"geo_id": "unit"})
    m["area_km2"] = m.to_crs(32638).area.to_numpy() / 1e6
    print(f"  median unit {m['area_km2'].median():,.0f} km2 "
          f"(~{m['area_km2'].median() / 0.74:,.0f} Kontur hexes); Tbilisi "
          f"{float(m.loc[m['unit'] == 'C. Tbilisi', 'area_km2'].iloc[0]):,.0f} km2")
    out = m[["unit", "key_pcode", "name", "region_census", "count", "area_km2", "geometry"]]
    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="units", driver="GPKG")
    print(f"  wrote {UNITS_OUT}")
    return gpd.GeoDataFrame(out, crs=4326), a1


SNAP_M = 1000
NE_COUNTRIES = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "geo",
                            "ne_10m_admin_0_countries.geojson")


def snap_coast(units, a1, layer):
    """Hexes hex_layer left outside every unit: class them, and snap the coastal ones back.

    Outside a unit is one of four things here: Abkhazia or the Tskhinvali region (COD-AB's own
    ADM1 for the ground the census did not count: dropped); a neighbour's town across the border
    (centroid inside Armenia, Azerbaijan, Russia or Turkey in Natural Earth: dropped, the
    playbook's "across a land border, drop it, not snap it"); Abkhazia's coast, whose hexes
    fall in the sea beside COD's Abkhazia polygon (over 5 km from any unit: dropped); and hexes
    of Batumi, Kobuleti, Poti and the other coastal units whose centroids sit just offshore or in
    a sliver between COD polygons (Georgia or sea in Natural Earth, within SNAP_M of a unit):
    snapped to the nearest unit, since dropping them moves the coast's dots inland.
    """
    import geopandas as gpd
    import pandas as pd
    from _grid import kontur_path

    k = gpd.read_file(kontur_path("ge"))
    pts = gpd.GeoDataFrame({"pop": k["population"].to_numpy(dtype=float)},
                           geometry=k.geometry.centroid, crs=k.crs).to_crs(4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(pts.index)
    out = pts[j["unit"].isna().to_numpy()].copy()
    o1 = gpd.sjoin(out, a1[["adm1_pcode", "geometry"]], how="left", predicate="within")
    o1 = o1[~o1.index.duplicated()].reindex(out.index)
    occ = o1["adm1_pcode"].isin(NOT_ENUMERATED_ADM1).to_numpy()
    ne = gpd.read_file(NE_COUNTRIES)[["ADM0_A3", "geometry"]].to_crs(4326)
    on = gpd.sjoin(out, ne, how="left", predicate="within")
    on = on[~on.index.duplicated()].reindex(out.index)
    out["ne"] = on["ADM0_A3"].fillna("sea").to_numpy()
    nn = gpd.sjoin_nearest(out.to_crs(32638), units[["unit", "geometry"]].to_crs(32638),
                           how="left", distance_col="d")
    nn = nn[~nn.index.duplicated()].reindex(out.index)
    out["near"], out["d"] = nn["unit"].to_numpy(), nn["d"].to_numpy()
    abroad = ~out["ne"].isin(["GEO", "sea"]).to_numpy()
    snap = (~occ) & (~abroad) & (out["d"] <= SNAP_M).to_numpy()
    far = (~occ) & (~abroad) & ~snap
    for label, m in (("Abkhazia and the Tskhinvali region (not enumerated), dropped", occ),
                     ("across the border in Natural Earth, dropped", abroad & ~occ),
                     (f"Georgia or sea, over {SNAP_M} m from a unit (Abkhazia's coast), dropped",
                      far),
                     (f"Georgia or sea, within {SNAP_M} m of a unit, snapped", snap)):
        print(f"    {int(m.sum()):>5,} hexes {out.loc[m, 'pop'].sum():>9,.0f} people  {label}")
    fb = out.loc[far].total_bounds
    print(f"    (the over-{SNAP_M} m hexes lie in lon {fb[0]:.2f}-{fb[2]:.2f}, lat {fb[1]:.2f}-"
          f"{fb[3]:.2f}; Abkhazia's coast runs from the Psou at 40.0 to the Enguri at 41.6)")
    if fb[2] > 41.7:
        raise SystemExit("a dropped Georgian/sea hex lies east of Abkhazia; look before dropping")
    if out.loc[far, "d"].min() < 5000:
        raise SystemExit("a dropped Georgian/sea hex is under 5 km from a unit; look before "
                         "dropping it")
    if out.loc[snap, "pop"].sum() > 20_000:
        raise SystemExit("snapping more than 20,000 people; the boundaries may have moved")
    by = out[snap].groupby("near")["pop"].sum().sort_values(ascending=False)
    print("    snapped, by unit: " + ", ".join(f"{u} {p:,.0f}" for u, p in by.head(6).items()))
    idx = out.index[snap]
    add = gpd.GeoDataFrame({"unit": out.loc[idx, "near"].astype(str).to_numpy(),
                            "pop": out.loc[idx, "pop"].to_numpy()},
                           geometry=k.geometry.loc[idx].to_crs(4326).to_numpy(), crs=4326)
    return gpd.GeoDataFrame(pd.concat([layer, add], ignore_index=True), crs=4326)


def main():
    if "--fetch" in sys.argv:
        fetch()
    from _grid import hex_layer
    units, a1 = build_units()
    census = dict(zip(units["unit"], units["count"].astype(int)))
    layer = hex_layer("ge", units[["unit", "geometry"]], census=census)
    print("\n  the hexes outside every unit:")
    layer = snap_coast(units, a1, layer)
    per = layer.groupby("unit")["pop"].sum()
    ratio = per.sum() / sum(census.values())
    coast = ["C. Batumi", "Kobuleti Municipality", "C. Poti", "Khelvachauri Municipality",
             "Ozurgeti Municipality", "Zugdidi Municipality", "Khobi Municipality"]
    print("    coastal units after the snap, Kontur/census normalised: "
          + ", ".join(f"{u} {per[u] / census[u] / ratio:.2f}" for u in coast))
    missing = sorted(set(census) - set(per.index[per > 0]))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    out = os.path.join(GEO, "ge_hexes.gpkg")
    layer.to_file(out, layer="hexes", driver="GPKG")
    print(f"  rewrote {out} ({len(layer):,} hexes, {layer['pop'].sum():,.0f} Kontur people)")


if __name__ == "__main__":
    main()
