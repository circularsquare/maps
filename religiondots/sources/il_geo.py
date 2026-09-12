"""Israel — the 2022 census statistical areas, cut on the Green Line, and a placement grid.

Writes:
    data/geo/il/il_units.gpkg        the drawn units, keyed as il.py keys them
                                     -- and this is ALSO the `place` layer; see below

Usage:
    python sources/il_geo.py --fetch   # CBS statistical areas, OCHA oPt, Kontur extract
    python sources/il_geo.py           # build the units and run §8.2e's grid test

**THE BOUNDARY FILE IS CBS'S OWN AND IT IS THE TABULATION GEOGRAPHY** — §12's first rule
about boundaries, satisfied for once without a hunt. CBS runs an ArcGIS Online organisation
(`ISRAEL_CBS_GIS`) whose `Statistical__Areas_2022` layer is **3,857 polygons keyed by
`SEMEL_YISHUV` + `STAT_2022`**, which are exactly the two codes the census tables carry. No
name join, no vintage gap, no correspondence workbook. It is in **EPSG:2039** (Israeli TM
Grid), not 4326, and a read that forgets to reproject puts Israel near the Gulf of Guinea.

**THE PUBLISHED UNIT IS A GROUP OF POLYGONS, AND THE TABLE SAYS WHICH.** CBS merges small
statistical areas before publishing, so the census file's `StatAreaCmb` is `"1022+1023+1024"`
where `StatArea` is `1022`. The boundary layer has one polygon per *raw* area, so every `+`
in a Cmb string is a dissolve instruction — §12's "treat a duplicated key as a grouping
instruction until proved otherwise", arriving for once as an explicit list. Ignoring it would
leave 193 units drawn at a fraction of their real extent while every count stayed correct.

---------------------------------------------------------------------------------------
THE TERRITORIAL CUT — Anita's decision, 2026-09-07, and the reasoning, because it is the
kind of thing that gets quietly reversed by someone who does not know it was decided.

**The map stops at the 1949 Green Line, and the Golan is the one deliberate exception.**

  * **The West Bank and Gaza are not drawn.** CBS counts 503,732 Israelis in its "Judea and
    Samaria Area" district; none of them are here. Gaza has no Israeli data at all.
  * **East Jerusalem is not drawn** — and that means *both* halves of it. Cutting only the
    Palestinian neighbourhoods while keeping Gilo, Pisgat Ze'ev, Ramot and Neve Ya'akov
    would draw the settlements and erase the people they were built among, which is §14.2's
    second risk exactly. So the line is the line, and Jerusalem draws as a western fragment:
    **the 15 Muslim-majority statistical areas (363,600 people) go, and so do the Jewish
    settlement neighbourhoods beyond the line.**
  * **The Golan IS drawn**, against the same rule, on Anita's reasoning that Israel counts
    those people and nobody else does — 56,600 of them, including **24,900 Druze** in Majdal
    Shams, Buq'ata, Mas'ade and Ein Qiniyye, about a sixth of the Druze on this map. It is an
    exception and is written down as one rather than being folded into the rule.

**The cut is a published geometry, not a line drawn here.** OCHA's Common Operational
Dataset for the State of Palestine (`cod-ab-pse`, `pse_admin0.geojson`) draws the West Bank
*including East Jerusalem* and Gaza. A CBS unit is dropped when the majority of its area
falls inside that polygon. Two things this buys: the call is citable to a body whose mandate
is the territory rather than to us, and the Golan falls out correctly for free, because OCHA's
oPt is not Syria.

**geoBoundaries PSE is the wrong file for this and looks right.** Its ADM0 excludes the
annexed Jerusalem municipality, so Shu'afat, Silwan and the Old City all test as *outside*
Palestine and East Jerusalem would have stayed on the map with nothing to show it had been
considered. Ten hand-checked points separate the two files; the check is kept in `check()`
so a future release of either cannot silently move the line.
---------------------------------------------------------------------------------------

**THERE IS NO POPULATION GRID, AND THAT WAS MEASURED RATHER THAN SKIPPED** — §8.2e, whose
test Israel fails harder than the country it was written for. A Kontur grid was built here
first, on the reflex that every country since Kenya has had one; `measure_grid_floor()` is
what is left of it and it runs on every build so the decision cannot rot:

| | Israel | Saint Vincent, where §8.2e was found |
|---|---|---|
| median unit area ÷ hex area | **0.6** | single digits |
| units getting no hex at all | **42.1%** | 36% — enough to abandon it |
| units smaller than ONE hex | **68.9%** | 43 of 221 |
| per-unit Kontur/census ratio | p10 0.27, median 1.01, p90 **3.49** | p10 0.00, p90 2.68 |

**A ratio below 1 means the grid is COARSER than the thing it is supposed to be refining.**
Israel's median statistical area is 0.69 km² against a 1.17 km² hex, so for two units in
three the "weight" is one cell covering the whole unit and several of its neighbours — which
is uniform placement with extra steps, applied to whichever unit the hex centre happened to
land in. §8.2e's conclusion applies unchanged: *"a unit a few hundred metres across does not
need its interior modelled, because at any zoom this map reaches, uniform inside it is
indistinguishable from correct."*

So `place` is the units layer itself and there is no `place_weight`. **What that costs is
worth naming**: the handful of genuinely large units — Negev Bedouin localities and regional
councils, up to 195 km² — get an even wash where the population really is clustered. That is
§8.2a's India problem in miniature, and it is the price of not applying a noisy weight to the
2,900 units that do not need one.
"""

import gzip
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "il")
RAW = os.path.join(ROOT, "data", "raw", "il")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

ARCGIS = ("https://services2.arcgis.com/xMRYm7cNgdR5RN6F/arcgis/rest/services"
          "/Statistical__Areas_2022/FeatureServer/0")
SA_GEOJSON = os.path.join(RAW, "statareas2022.geojson")

OCHA_ZIP_URL = ("https://data.humdata.org/dataset/2caf8373-816f-458c-9913-71bddb9cab7c/"
                "resource/ca372385-4c79-4378-abf1-cb506fb98023/download/"
                "pse_admin_boundaries.geojson.zip")
PSE_ADM0 = os.path.join(RAW, "pse_admin0.geojson")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_IL_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_IL_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_IL_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "il_units.gpkg")
GRID_OUT = os.path.join(GEO, "il_grid.gpkg")

SA_CSV = os.path.join(RAW, "sa2022.csv")

EXPECTED_POLYGONS = 3857
SRC_CRS = 2039
DROP_IF_INSIDE = 0.5        # share of a unit's area inside oPt before it is dropped

# Ten points whose side of the Green Line is not in dispute, used to prove the OCHA polygon
# is the file we think it is. geoBoundaries PSE fails five of them. (lon, lat, inside oPt)
GREEN_LINE_PROBES = [
    ("West Jerusalem centre", 35.2137, 31.7833, False),
    ("Shu'afat", 35.2340, 31.8080, True),
    ("Silwan", 35.2370, 31.7710, True),
    ("Old City", 35.2340, 31.7767, True),
    ("Gilo", 35.1930, 31.7300, True),
    ("Pisgat Ze'ev", 35.2450, 31.8250, True),
    ("Tel Aviv", 34.7818, 32.0853, False),
    ("Ma'ale Adumim", 35.2980, 31.7730, True),
    ("Majdal Shams (Golan, kept)", 35.7660, 33.2680, False),
    ("Gaza City", 34.4667, 31.5000, True),
]


# =====================================================================================
# fetch
# =====================================================================================

def _get(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_statareas():
    if os.path.exists(SA_GEOJSON) and os.path.getsize(SA_GEOJSON) > 1_000_000:
        print(f"  have {os.path.basename(SA_GEOJSON)} "
              f"({os.path.getsize(SA_GEOJSON):,} b)")
        return
    fields = "SEMEL_YISHUV,SHEM_YISHUV,SHEM_YISHUV_ENGLISH,STAT_2022,YISHUV_STAT_2022"
    feats, offset = [], 0
    while True:
        url = (f"{ARCGIS}/query?where=1%3D1&outFields={fields}&returnGeometry=true"
               f"&outSR=4326&resultOffset={offset}&resultRecordCount=1000&f=geojson")
        doc = json.loads(_get(url).decode("utf-8"))
        batch = doc.get("features", [])
        feats.extend(batch)
        print(f"    +{len(batch)} (total {len(feats)})")
        if len(batch) < 1000:
            break
        offset += len(batch)
    if len(feats) != EXPECTED_POLYGONS:
        raise SystemExit(f"got {len(feats)} polygons, expected {EXPECTED_POLYGONS} -- "
                         "CBS has revised the layer; check before trusting the join")
    os.makedirs(RAW, exist_ok=True)
    with open(SA_GEOJSON, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    print(f"  wrote {SA_GEOJSON} ({os.path.getsize(SA_GEOJSON):,} b)")


def fetch_ocha():
    if os.path.exists(PSE_ADM0) and os.path.getsize(PSE_ADM0) > 100_000:
        print(f"  have {os.path.basename(PSE_ADM0)}")
        return
    blob = _get(OCHA_ZIP_URL)
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        raise SystemExit(f"OCHA download is not a zip ({len(blob)} bytes)")
    z = zipfile.ZipFile(io.BytesIO(blob))
    name = "pse_admin0.geojson"
    if name not in z.namelist():
        raise SystemExit(f"{name} missing from OCHA zip: {z.namelist()}")
    os.makedirs(RAW, exist_ok=True)
    with open(PSE_ADM0, "wb") as fh:
        fh.write(z.read(name))
    print(f"  wrote {PSE_ADM0} ({os.path.getsize(PSE_ADM0):,} b)")


def fetch_kontur():
    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print(f"  have {os.path.basename(KONTUR)}")
        return
    if not os.path.exists(KONTUR_GZ):
        blob = _get(KONTUR_URL, timeout=600)
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(blob)
        print(f"  {len(blob):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  wrote {KONTUR} ({os.path.getsize(KONTUR):,} b)")


def fetch():
    print("CBS statistical areas 2022:")
    fetch_statareas()
    print("OCHA oPt admin0 (the Green Line, East Jerusalem included):")
    fetch_ocha()
    print("Kontur Israel population grid:")
    fetch_kontur()


# =====================================================================================
# build
# =====================================================================================

def unit_map():
    """{(locality_code, raw_stat_area): drawn unit key} from the census file's Cmb column.

    A `+` in StatAreaCmb is a dissolve instruction (see docstring).
    """
    import csv
    with open(SA_CSV, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    split_parents = {r["LocalityCode"].strip() for r in rows if r["StatArea"].strip()}

    def canon(cmb):
        """`2312+2311+2312` -> `2312+2311`. CBS repeats a code in one Jerusalem Cmb string;
        `sources/il.py`'s canon_cmb() is the same function on the same column and the two
        must agree, or the counts and these polygons stop sharing a key."""
        seen, out = set(), []
        for part in str(cmb).split("+"):
            part = part.strip()
            if part and part not in seen:
                seen.add(part)
                out.append(part)
        return "+".join(out)

    m, whole = {}, set()
    for r in rows:
        loc, cmb = r["LocalityCode"].strip(), canon(r["StatAreaCmb"])
        if not loc:
            continue
        if cmb:
            key = f"{loc}_{cmb}"
            for part in cmb.split("+"):
                part = part.strip()
                if part:
                    m[(loc, part)] = key
        elif loc not in split_parents:
            whole.add(loc)
    return m, whole, split_parents


def build_units():
    import geopandas as gpd

    print("units:")
    g = gpd.read_file(SA_GEOJSON)
    # §12 failure 4 / the zero-feature read: assert the count, not the absence of an error.
    if len(g) != EXPECTED_POLYGONS:
        raise SystemExit(f"{len(g)} polygons read, expected {EXPECTED_POLYGONS}")
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    print(f"  {len(g):,} raw statistical-area polygons")

    m, whole, split_parents = unit_map()
    print(f"  census file: {len(split_parents):,} split localities, "
          f"{len(whole):,} drawn whole, {len(set(m.values())):,} statistical-area units")

    # Some polygons carry a null SEMEL_YISHUV or STAT_2022 -- CBS ships a few areas with no
    # locality or no statistical-area code. They belong to no census row, so they map to no
    # drawn unit; they are counted rather than allowed to raise or to pass silently.
    def key_for(row):
        loc_v, stat_v = row["SEMEL_YISHUV"], row["STAT_2022"]
        if loc_v is None or stat_v is None:
            return None
        try:
            loc, stat = str(int(loc_v)), str(int(stat_v))
        except (TypeError, ValueError):
            return None
        if loc in split_parents:
            return m.get((loc, stat))
        return loc if loc in whole else None

    g["unit"] = g.apply(key_for, axis=1)
    nullcode = int(g["SEMEL_YISHUV"].isna().sum() + g["STAT_2022"].isna().sum())
    unmapped = int(g["unit"].isna().sum())
    print(f"  {nullcode:,} polygons carry a null locality or statistical-area code")
    print(f"  {unmapped:,} polygons map to no drawn unit "
          f"({100 * unmapped / len(g):.1f}%) -- no census row for that locality")
    g = g[g["unit"].notna()].copy()

    names = {}
    for _, r in g.iterrows():
        names.setdefault(r["unit"], r.get("SHEM_YISHUV") or "")
    units = g.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    units["name"] = units["unit"].map(names)
    print(f"  dissolved to {len(units):,} units")
    return units


def apply_green_line(units):
    """Drop every unit that is mostly inside OCHA's oPt polygon. See the docstring."""
    import geopandas as gpd
    from shapely.geometry import Point

    print("the Green Line:")
    pse = gpd.read_file(PSE_ADM0)
    if pse.empty:
        raise SystemExit("OCHA pse_admin0 read as ZERO features")
    pse = pse.to_crs(4326)
    opt = pse.union_all() if hasattr(pse, "union_all") else pse.unary_union

    bad = []
    for label, lon, lat, expect in GREEN_LINE_PROBES:
        got = opt.contains(Point(lon, lat))
        if got != expect:
            bad.append(f"{label}: inside={got}, expected {expect}")
    if bad:
        raise SystemExit("the oPt polygon is not the file this was written against:\n  "
                         + "\n  ".join(bad))
    print(f"  OK all {len(GREEN_LINE_PROBES)} Green Line probe points as expected")

    eq = units.to_crs(3857)
    opt_eq = gpd.GeoSeries([opt], crs=4326).to_crs(3857).iloc[0]
    area = eq.geometry.area
    inter = eq.geometry.intersection(opt_eq).area
    frac = (inter / area.replace(0, 1)).fillna(0.0)
    units = units.assign(_opt=frac.values)

    drop = units[units["_opt"] > DROP_IF_INSIDE]
    straddle = units[(units["_opt"] > 0.01) & (units["_opt"] <= DROP_IF_INSIDE)]
    print(f"  {len(drop):,} units dropped as beyond the Green Line")
    print(f"  {len(straddle):,} units straddle it and are KEPT (majority west):")
    for _, r in straddle.head(12).iterrows():
        print(f"      {r['unit']:<22} {str(r['name'])[:22]:<22} {100 * r['_opt']:.0f}% east")
    kept = units[units["_opt"] <= DROP_IF_INSIDE].drop(columns=["_opt"])
    print(f"  {len(kept):,} units drawn")
    return kept, drop


def measure_grid_floor(units):
    """Apply spec 8.2e's test and REFUSE to build a grid Israel is too fine for.

    This runs on every build rather than being a note, because the decision is a property
    of CBS's tabulation tier against Kontur's resolution and either could change. If the
    numbers ever come out the other way, the grid is worth building and this says so.
    """
    import geopandas as gpd
    import numpy as np

    print("placement -- spec 8.2e's resolution-floor test:")
    layers = gpd.list_layers(KONTUR) if hasattr(gpd, "list_layers") else None
    layer = layers["name"].iloc[0] if layers is not None else None
    hexes = gpd.read_file(KONTUR, layer=layer)
    if hexes.empty:
        raise SystemExit("Kontur extract read as ZERO features (try engine='fiona')")
    hexes = hexes.to_crs(3857)
    eq = units.to_crs(3857)

    ua = eq.geometry.area / 1e6
    ha = hexes.geometry.area / 1e6
    ratio = float(np.median(ua) / np.median(ha))

    cent = hexes.copy()
    cent["geometry"] = cent.geometry.representative_point()
    j = gpd.sjoin(cent[["geometry"]], eq[["unit", "geometry"]],
                  how="inner", predicate="within")
    nohex = len(eq) - len(set(j["unit"]))
    smaller = int((ua < float(np.median(ha))).sum())

    print(f"  {len(hexes):,} hexes, median {float(np.median(ha)):.3f} km2")
    print(f"  {len(eq):,} units,  median {float(np.median(ua)):.3f} km2")
    print(f"  median unit area / hex area = {ratio:.2f}")
    print(f"  units with no hex centre    = {nohex:,} ({100 * nohex / len(eq):.1f}%)")
    print(f"  units smaller than one hex  = {smaller:,} "
          f"({100 * smaller / len(eq):.1f}%)")

    if ratio < 5 or nohex > 0.3 * len(eq):
        print("  -> BELOW THE FLOOR. No grid is written; placement is the unit polygon,")
        print("     i.e. spec 8.2's uniform share, which 8.2e says is the better answer")
        print("     and not a fallback. Saint Vincent abandoned its grid at 36% and a")
        print("     ratio in single digits; Israel is worse on both.")
        return False
    print("  -> above the floor: a Kontur grid WOULD pay here. Nothing builds it yet --")
    print("     see this module's docstring, and read 8.2e before adding it.")
    return True


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def main():
    if "--fetch" in sys.argv:
        fetch()
    os.makedirs(GEO, exist_ok=True)
    units = build_units()
    units, dropped = apply_green_line(units)
    units.to_file(UNITS_OUT, driver="GPKG", layer="units")
    print(f"\nwrote {UNITS_OUT} ({len(units):,} units)")

    measure_grid_floor(units)

    with open(os.path.join(GEO, "dropped_units.json"), "w", encoding="utf-8") as fh:
        json.dump(sorted(dropped["unit"].tolist()), fh, ensure_ascii=False, indent=1)
    print(f"wrote dropped_units.json ({len(dropped):,} beyond the Green Line)")


if __name__ == "__main__":
    main()
