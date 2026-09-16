"""Uganda 2024: the census's own subcounty polygons, from UBOS's NPHC 2024 portal, and the Kontur layer.

Writes data/geo/ug/ug2024_subcounties.gpkg (drawn unit, names, polygon) and
data/geo/ug/ug2024_hexes.gpkg (drawn unit, pop, hex polygon), the place layer countries/ug.py reads.

WHY THE PORTAL AND NOT COD-AB. HDX's `cod-ab-uga` is still the 2020 edition (135 districts,
1,520 subcounties at admin 4, no parishes; its caveat reads "This COD-AB reflects the 135
district administrative system"). geoBoundaries has 137 districts (2020) and 1,521 subcounties
(UNHCR 2019). HDX's other Uganda layers are the 2006 gazetteer (a table, no polygons) and 2010
subcounties. The 2024 census counts 146 districts, 312 counties, 2,207 subcounties and 10,854
parishes. `statistics.ubos.org/nphc/map` draws the census's own units over
`api/get_geospatial_data.php`, checked 2026-09-15:
  level=district                               147 polygons, 146 population rows
  level=subcounty (no filter)                  313 COUNTY features on 312 codes (Kampala's 1025 is
                                               two), with 2,208 subcounty population rows
  level=subcounty&district_code&county_code    that county's subcounty polygons
  level=parish                                 `{"error":"Could not read GeoJSON file: "}` with or
                                               without a subcounty_code; the map's own code has
                                               no parish level
So subcounty polygons come one county at a time, 312 requests at half a second, and no 2024
parish polygon is published: the parish tier the stability test supports cannot be drawn.
The portal's home page states no terms of use.

THE JOIN IS ON CODES, WITNESSED BY NAMES. Each polygon carries DCode, CCode and SCode, the same
codes as the sample file's HH_DISTRICT, HH_COUNTY and HH_SUBCOUNTY, so the unit id is built from
them on both sides. The witness the codes do not determine is the name: the polygon's
`Sub_County` against the sample's HH_SUBCOUNTYN for the same code (one differs by a hyphen).

TWO REFUGEE SETTLEMENTS DO NOT FIT THE POLYGONS. 2,204 of the 2,207 census subcounties have a
polygon. The three without are Bidi Bidi Refugee Camp in Yumbe, one census subcounty in each of
counties 1, 2 and 4, 121,919 people in households; the settlement's zones lie inside host
subcounties' polygons. Each is drawn merged with the hosts in its own county, and the hosts are
the subcounties there whose Kontur people exceed the census household count by more than 1.2
times (the national Kontur ratio is 1.10); `CAMP_HOSTS` in sources/ug_2024.py pins them and this
module recomputes and asserts the set. The witness that the set is right is that the Kontur
surplus in each host set is about the same fraction of its camp in all three counties. The
portal's one polygon with no census row, `LOBULE REFUGEE CAMP` (2.2 km2) in Koboko, joins
Lobule subcounty, whose people it holds.

PLACEMENT is Kontur's 2023-11 population hexes, joined on centroids, as a within-unit weight
only: every unit's people come from the census workbook, so a subcounty Kontur models badly
gets its dots on a worse surface, never the wrong number of them.

Usage:
    python sources/ug_2024_geo.py --fetch    download what is missing into data/raw/ug/nphc2024_portal/
    python sources/ug_2024_geo.py            build the two layers
"""
import json
import os
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "ug", "nphc2024_portal")
GEO = os.path.join(ROOT, "data", "geo", "ug")
UNITS_OUT = os.path.join(GEO, "ug2024_subcounties.gpkg")
HEXES_OUT = os.path.join(GEO, "ug2024_hexes.gpkg")
KONTUR = os.path.join(ROOT, "data", "raw", "ug", "kontur_population_UG_20231101.gpkg")
CODAB = os.path.join(ROOT, "data", "raw", "ug", "uga_admin_boundaries.shp.zip")
PORTAL = "https://statistics.ubos.org/nphc/"
API = PORTAL + "api/get_geospatial_data.php"
XLSX_URL = PORTAL + "resources/NPHC-2024-Subcounty-Profiles-Excel-Tables.xlsx"
XLSX = os.path.join(RAW, "NPHC-2024-Subcounty-Profiles-Excel-Tables.xlsx")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
PAUSE_S = 0.5
EXPECT_COUNTIES = 312
BBOX = (29.4, -1.6, 35.2, 4.4)
SNAP_M = 1_000
KONTUR_BAND = (0.65, 1.35)
# The portal polygon with no census row, and the census subcounty it belongs to.
POLYGON_INTO = {"UG3190107": ("LOBULE REFUGEE CAMP", "UG3190104")}
NAME_HYPHEN = {"UG3050108"}  # OMIYA-ANYIMA in the sample, OMIYA ANYIMA on the polygon
CAMP_SURPLUS_BAND = (0.5, 0.9)


def _session():
    import requests
    import urllib3

    urllib3.disable_warnings()
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _json_body(raw):
    """The PHP prints deprecation notices before the JSON on some paths; start at the first brace."""
    i = raw.find(b"{")
    if i < 0:
        raise ValueError(f"no JSON in reply: {raw[:200]!r}")
    d = json.loads(raw[i:])
    if "error" in d:
        raise ValueError(f"portal error: {d['error']}")
    return d


def _get_json(s, params, path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                _json_body(fh.read())
            return False
        except ValueError:
            pass
    r = s.get(API, params=params, timeout=180, verify=False)
    r.raise_for_status()
    _json_body(r.content)
    with open(path + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(path + ".part", path)
    return True


def _counties(d):
    """(district code, county code) from the county features: 313 features on 312 codes."""
    feats = d["geojson"]["features"]
    keys = [(f["properties"]["DCode"], f["properties"]["DCode"] + f["properties"]["CCode"])
            for f in feats]
    counties = sorted(set(keys))
    if len(feats) != EXPECT_COUNTIES + 1 or len(counties) != EXPECT_COUNTIES:
        raise SystemExit(f"{len(feats)} county features on {len(counties)} codes, expected "
                         f"{EXPECT_COUNTIES + 1} on {EXPECT_COUNTIES}")
    return counties, sorted({k for k in keys if keys.count(k) > 1})


def fetch():
    os.makedirs(RAW, exist_ok=True)
    s = _session()
    if not os.path.exists(XLSX):
        r = s.get(XLSX_URL, timeout=900, verify=False)
        r.raise_for_status()
        if r.content[:4] != b"PK\x03\x04":
            raise SystemExit(f"{XLSX_URL} is not an xlsx: {r.content[:16]!r}")
        with open(XLSX + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(XLSX + ".part", XLSX)
        print(f"  xlsx {len(r.content):,} bytes")
    _get_json(s, {"level": "district"}, os.path.join(RAW, "district.json"))
    allsub = os.path.join(RAW, "counties_with_subcounty_populations.json")
    _get_json(s, {"level": "subcounty"}, allsub)
    with open(allsub, "rb") as fh:
        d = _json_body(fh.read())
    counties, dup = _counties(d)
    print(f"  county codes drawn as more than one feature: {dup}")
    got = 0
    for i, (dc, cc) in enumerate(counties):
        path = os.path.join(RAW, f"subcounties_{cc}.json")
        if _get_json(s, {"level": "subcounty", "district_code": dc, "county_code": cc}, path):
            got += 1
            time.sleep(PAUSE_S)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(counties)} counties ({got} fetched this run)", flush=True)
    print(f"  {len(counties)} county files present, {got} fetched this run")


def _polygonal(geom):
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    g = make_valid(geom)
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    parts = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    return unary_union(parts) if parts else None


def build():
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import shape

    import ug_2024

    failures = []

    def say(ok, msg):
        ug_2024.say(ok, msg, failures)

    with open(os.path.join(RAW, "counties_with_subcounty_populations.json"), "rb") as fh:
        counties, _ = _counties(_json_body(fh.read()))
    rows = []
    for dc, cc in counties:
        with open(os.path.join(RAW, f"subcounties_{cc}.json"), "rb") as fh:
            d = _json_body(fh.read())
        for f in d["geojson"]["features"]:
            p = f["properties"]
            if p["DCode"] != dc or p["DCode"] + p["CCode"] != cc:
                raise SystemExit(f"subcounties_{cc}.json holds a feature of {p}")
            rows.append(dict(unit=ug_2024.unit_id(p["DCode"], p["CCode"], p["SCode"]),
                             polygon_name=ug_2024.norm(p["Sub_County"]),
                             geometry=_polygonal(shape(f["geometry"]))))
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    print(f"1. {len(g):,} subcounty features from {len(counties)} county files, "
          f"{g['unit'].nunique():,} codes")
    say(g.geometry.notna().all(), "every feature has a polygonal geometry after make_valid")
    b = g.total_bounds
    say(BBOX[0] <= b[0] and b[2] <= BBOX[2] and BBOX[1] <= b[1] and b[3] <= BBOX[3],
        f"bounds {np.round(b, 3).tolist()} inside Uganda's box (lon/lat order, not swapped)")
    g = g.dissolve(by="unit", aggfunc="first").reset_index()

    P, lv, national, apaa = ug_2024.profiles()
    sm, _ = ug_2024.load_sample(failures)
    J = ug_2024.join(sm, P, lv, ug_2024.district_names(), failures)
    S = ug_2024.subcounty_table(J, P)
    m = S.merge(g, on="unit", how="outer", indicator=True)
    no_poly = sorted(m.loc[m["_merge"] == "left_only", "unit"])
    extra = {r["unit"]: r["polygon_name"] for _, r in m[m["_merge"] == "right_only"].iterrows()}
    print(f"\n2. join on codes: {int((m['_merge'] == 'both').sum()):,} both")
    say(no_poly == sorted(ug_2024.CAMP_HOSTS),
        f"the sample subcounties with no polygon are Bidi Bidi's three: {no_poly}")
    say(extra == {k: v[0] for k, v in POLYGON_INTO.items()},
        f"the polygons with no census subcounty are the pinned one: {extra}")
    both = m[m["_merge"] == "both"]
    fold = lambda s: s.str.replace("-", " ", regex=False).str.split().str.join(" ")  # noqa: E731
    diff = set(both.loc[both["subcounty_name"] != both["polygon_name"], "unit"])
    folded = both[fold(both["subcounty_name"]) != fold(both["polygon_name"])]
    say(folded.empty and diff == NAME_HYPHEN,
        f"the polygon's name is the sample's for all {len(both):,} codes, once a hyphen is a space "
        f"({sorted(diff)} differ by a hyphen)")
    for src, (_, into) in POLYGON_INTO.items():
        g.loc[g["unit"] == src, "unit"] = into
    g = g.dissolve(by="unit", aggfunc="first").reset_index()

    ea = g.to_crs("ESRI:54034")
    areas = ea.area / 1e6
    union = ea.union_all() if hasattr(ea, "union_all") else ea.unary_union
    ua = union.area / 1e6
    say(areas.sum() / ua < 1.01, f"subcounty areas sum to {areas.sum():,.0f} km2 against a union of "
                                 f"{ua:,.0f} km2 (overlap {areas.sum() / ua - 1:+.3%})")
    member = next(n for n in zipfile.ZipFile(CODAB).namelist()
                  if n.lower().endswith(".shp") and "admin0" in n.lower())
    a0 = gpd.read_file(f"zip://{CODAB}!{member}").to_crs("ESRI:54034")
    a0a = a0.area.sum() / 1e6
    say(0.9 < ua / a0a < 1.1, f"the union is {ua / a0a:.3f} of COD-AB's admin0 area ({a0a:,.0f} km2)")
    print(f"     subcounty area: median {areas.median():,.1f} km2, smallest {areas.min():.2f}, "
          f"{int((areas < 0.74 * 5).sum())} under five Kontur hexes of area")

    print("\n3. Kontur")
    hexes = gpd.read_file(KONTUR)
    say(len(hexes) > 0, f"{len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")
    cent = gpd.GeoDataFrame({"population": hexes["population"].to_numpy()},
                            geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs("EPSG:4326")
    j = gpd.sjoin(cent, g[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(cent.index)
    out = j["unit"].isna()
    if out.any():
        near = gpd.sjoin_nearest(cent[out].to_crs("EPSG:32636"),
                                 g.to_crs("EPSG:32636")[["unit", "geometry"]],
                                 how="left", max_distance=SNAP_M)
        near = near[~near.index.duplicated(keep="first")]
        j.loc[near.index, "unit"] = near["unit"]
    lost = j["unit"].isna()
    print(f"     hex centroids outside every subcounty: {int(out.sum()):,}; "
          f"{int((out & ~lost).sum()):,} snapped within {SNAP_M} m; "
          f"{int(lost.sum()):,} dropped ({cent.loc[lost, 'population'].sum():,.0f} people, "
          f"{cent.loc[lost, 'population'].sum() / cent['population'].sum():.3%})")
    kp = cent.loc[~lost, "population"].groupby(j.loc[~lost, "unit"]).sum()
    S["kontur"] = S["unit"].map(kp).fillna(0.0)
    S["ratio"] = S["kontur"] / S["hhpop"]

    # Bidi Bidi's hosts, recomputed from Kontur and held against CAMP_HOSTS
    for camp, hosts in ug_2024.CAMP_HOSTS.items():
        county = camp[:7]
        found = sorted(S.loc[(S["unit"].str[:7] == county) & (S["unit"] != camp)
                             & (S["ratio"] > ug_2024.HOST_RATIO), "unit"])
        hs = S[S["unit"].isin(hosts)]
        surplus = float((hs["kontur"] - hs["hhpop"]).sum())
        camp_pop = int(S.loc[S["unit"] == camp, "hhpop"].iloc[0])
        say(found == sorted(hosts) and CAMP_SURPLUS_BAND[0] < surplus / camp_pop < CAMP_SURPLUS_BAND[1],
            f"{camp} ({camp_pop:,} in households): hosts over {ug_2024.HOST_RATIO} are "
            + ", ".join(f"{h} {S.loc[S['unit'] == h, 'subcounty_name'].iloc[0].title()} "
                        f"{S.loc[S['unit'] == h, 'ratio'].iloc[0]:.2f}" for h in found)
            + f"; their Kontur surplus is {surplus / camp_pop:.2f} of the camp")

    national_ratio = S["kontur"].sum() / S["hhpop"].sum()
    say(KONTUR_BAND[0] < national_ratio < KONTUR_BAND[1],
        f"Kontur 2023 over the census household population nationally: {national_ratio:.3f}")
    real = S[~S["unit"].isin(list(ug_2024.CAMP_HOSTS))]
    q = np.quantile(real["ratio"], [0.01, 0.1, 0.5, 0.9, 0.99])
    print(f"     per subcounty: p1 {q[0]:.2f}, p10 {q[1]:.2f}, median {q[2]:.2f}, p90 {q[3]:.2f}, "
          f"p99 {q[4]:.2f} (a weight inside each unit, not a count)")
    for _, r in pd.concat([real.nsmallest(4, "ratio"), real.nlargest(4, "ratio")]).iterrows():
        print(f"       {r['unit']} {r['subcounty_name'].title()}, {r['district_name'].title()}: "
              f"{r['ratio']:.2f}")

    # drawn units
    g["drawn"] = g["unit"].map(ug_2024.drawn_unit)
    units = g.dissolve(by="drawn", aggfunc="first").reset_index()[["drawn", "geometry"]]
    units = units.rename(columns={"drawn": "unit"})
    D = S.groupby("drawn").agg(hhpop=("hhpop", "sum"), district_name=("district_name", "first"),
                               county_name=("county_name", "first"),
                               subcounty_name=("subcounty_name", " and ".join)).reset_index()
    units = units.merge(D.rename(columns={"drawn": "unit"}), on="unit", how="outer", indicator=True)
    say(bool((units["_merge"] == "both").all()),
        f"{len(units):,} drawn units, every one with a polygon and a population")
    units = gpd.GeoDataFrame(units.drop(columns="_merge"), geometry="geometry", crs="EPSG:4326")
    place = gpd.GeoDataFrame({"unit": j.loc[~lost, "unit"].map(ug_2024.drawn_unit).to_numpy(),
                              "pop": cent.loc[~lost, "population"].to_numpy(dtype=float)},
                             geometry=hexes.geometry[(~lost).to_numpy()].to_crs("EPSG:4326").to_numpy(),
                             crs="EPSG:4326")
    per = place.groupby("unit")["pop"].sum()
    hexless = sorted(set(units["unit"]) - set(per.index[per > 0]))
    say(len(hexless) <= 0.01 * len(units),
        f"{len(hexless)} drawn units with no populated hex centroid take their own polygon: {hexless}")
    if hexless:
        add = units[units["unit"].isin(hexless)][["unit", "geometry"]].copy()
        add["pop"] = 1.0
        place = gpd.GeoDataFrame(pd.concat([place, add[["unit", "pop", "geometry"]]],
                                           ignore_index=True), geometry="geometry", crs="EPSG:4326")
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed; nothing written")

    os.makedirs(GEO, exist_ok=True)
    for path, frame, layer in ((UNITS_OUT, units, "subcounties"), (HEXES_OUT, place, "hexes")):
        tmp = path[:-5] + ".part.gpkg"
        if os.path.exists(tmp):
            os.remove(tmp)
        frame.to_file(tmp, layer=layer, driver="GPKG")
        os.replace(tmp, path)
    print(f"\nwrote {UNITS_OUT} ({len(units):,}) and {HEXES_OUT} ({len(place):,})")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        build()
