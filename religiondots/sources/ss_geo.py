"""South Sudan: the ten former states from COD-AB v03, and a Kontur grid calibrated to the 2025 county estimates.

Writes:
    data/geo/ss/ss_units.gpkg          the 10 states (`units`), all ten, sampled or not
    data/geo/ss/ss_hexes.gpkg          Kontur 400 m hexes with `unit`, `county`, `kontur_pop` and
                                       `pop` (`place`); `pop` is calibrated to the county
    data/geo/ss/ss_lookup.csv          state -> name, 2025 estimate, COD-PS 2022, raw Kontur
    data/geo/ss/ss_counties.csv        county -> state, 2025 estimate, raw Kontur, scale factor

Usage:
    python sources/ss_geo.py --fetch    COD-AB shapefile zip (~2.6 MB), Kontur (~5 MB), COD-PS
    python sources/ss_geo.py            rebuild from data/raw/ss/

`sources/ss.md` is the record; `sources/ss.py` lays the survey's shares on these states.

## BOUNDARIES: OCHA COD-AB SOUTH SUDAN (`cod-ab-ssd`), VERSION 03

10 states (SS01-SS10), 78 counties, 512 payams; valid on 2022-12-19, reviewed 2024-10-09, CC BY-IGO.
Its admin2 layer has a 79th feature, SS0001 Abyei under a parent `SS00 Abyei Region` that admin1
does not have; it is dropped by pcode.
These are the ten states of 2005-2015 and of 2020 onward (the 28- and 32-state maps of 2015-2020
were reversed by the February 2020 peace agreement), which are also the states the High Frequency
Survey sampled in 2015-16. Abyei is not in this file; Sudan's COD-AB carries it and `countries/sd.py`
leaves it undrawn.

## POPULATION: THE 2025 COUNTY ESTIMATES, NOT THE 2022 COD-PS

`cod-ps-ssd` holds two editions. `ssd_admpop_*_2022_v2` is the NBS and UNFPA projection the
government endorsed in November 2020 (12,394,970, no Abyei). `SSD_2024_population_estimates_data.xlsx`
is the 2025 figure per county "based on the 2008 census and annual natural growth and attrition rates
with displacement adjusted estimates", cleared with NBS and adopted by the humanitarian information
management group: 13,442,554 with Abyei's 145,358 (its own row, dropped here). The 2025 edition is
used because it is the newer count of where people are after the returns from Sudan since 2023, and
because it is by county, which the grid calibration below needs. Every county is 1.03x to 1.59x its
2022 figure (Nagero 1.59, Baliet 1.32, Raja 1.29; most 1.03-1.10), printed per state.

The workbook files Uror county (SS0311) under `Admin1_Pcode` SS04 (Lakes) while naming its state
Jonglei; COD-AB and the 2022 table both put SS0311 in Jonglei. The join is on the county pcode and the
parent comes from COD-AB; the one misfiled row is pinned in `MISFILED_PARENT`.

## WHY THE GRID IS CALIBRATED

Raw Kontur (November 2023) is laid on each county's 2025 total, keeping Kontur's shape inside the
county (DR Congo's construction, `sources/cd_geo.py`). A county Kontur has largely lost (factor over
`HOLE_FACTOR` times the national one) gets its total spread evenly over Kontur's hexes instead of
scaled into a false city.
"""

import gzip
import math
import os
import re
import shutil
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)      # kontur_cap
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "ss")
GEO = os.path.join(ROOT, "data", "geo", "ss")
OUT_UNITS = os.path.join(GEO, "ss_units.gpkg")
OUT_HEXES = os.path.join(GEO, "ss_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "ss_lookup.csv")
OUT_COUNTIES = os.path.join(GEO, "ss_counties.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
SHP_ZIP = "ssd_admin_boundaries.shp.zip"
KONTUR_GPKG = "kontur_population_SS_20231101.gpkg"
POP_2025 = "ssd_2024_population_estimates_data.xlsx"
POP_2022 = "ssd_admpop_adm1_2022_v2.csv"
DOWNLOADS = {
    SHP_ZIP: (
        "https://data.humdata.org/dataset/cdd62bd9-e442-4eac-9b44-cfee8bf79153/resource/"
        "47b1ce82-198e-4707-a70d-63c47021d669/download/ssd_admin_boundaries.shp.zip",
        b"PK", 1_000_000),
    KONTUR_GPKG + ".gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_SS_20231101.gpkg.gz", b"\x1f\x8b", 1_000_000),
    POP_2025: (
        "https://data.humdata.org/dataset/5116cb9f-5db8-4f56-9766-735d88dcbbfa/resource/"
        "dfad888d-c594-4795-9b5b-3cbe958a3c18/download/ssd_2024_population_estimates_data.xlsx",
        b"PK", 50_000),
    POP_2022: (
        "https://data.humdata.org/dataset/5116cb9f-5db8-4f56-9766-735d88dcbbfa/resource/"
        "7a5243b5-9b54-4453-a30e-044e79417e1f/download/ssd_admpop_adm1_2022_v2.csv",
        b"\xef\xbb\xbf", 1_000),
}

N_STATES = 10
N_COUNTIES = 78
CODPS_2022 = 12_394_970
# The workbook's own total row, Abyei included, and Abyei's row, which has no polygon here.
POP_2025_WITH_ABYEI = 13_442_554
ABYEI = "SS0001"
MISFILED_PARENT = {"SS0311": "SS04"}   # Uror: Jonglei by name and by COD-AB, SS04 in the workbook
AREA_TOL = 0.02

HOLE_FACTOR = 10.0
EXPECT_HOLES = set()

# Raw Kontur blocks at the 46,200/km2 cap, by (lon, lat) of the block's peak, matched within 1 km.
BLOCKS = {}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (url, magic, least) in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > least:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=1800)
        r.raise_for_status()
        if not r.content.startswith(magic) or len(r.content) < least:
            raise SystemExit(f"{name} starts {r.content[:16]!r}, {len(r.content):,} bytes")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def unpack():
    for name in (SHP_ZIP, POP_2025, POP_2022):
        if not os.path.exists(os.path.join(RAW, name)):
            raise SystemExit(f"missing {name}; run with --fetch first")
    gz = os.path.join(RAW, KONTUR_GPKG + ".gz")
    gpkg = os.path.join(RAW, KONTUR_GPKG)
    if not os.path.exists(gpkg):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    return gpkg


def read_pop_2025(a2):
    """The 2025 county estimates, joined to COD-AB's 79 counties on the county pcode."""
    import pandas as pd

    ps = pd.read_excel(os.path.join(RAW, POP_2025), sheet_name="Pop_Stats_Summary")
    ps = ps[ps["Admin2_Pcode"].notna()].copy()
    total = float(ps.loc[ps["Admin2_Pcode"].notna(), "Population - 2025"].sum())
    if round(total) != POP_2025_WITH_ABYEI:
        raise SystemExit(f"the 2025 county rows sum to {total:,.0f}, expected {POP_2025_WITH_ABYEI:,}")
    if ps["Admin2_Pcode"].duplicated().any():
        raise SystemExit("a county pcode repeats in the 2025 workbook")
    ab = ps[ps["Admin2_Pcode"] == ABYEI]
    if len(ab) != 1 or not str(ab["Admin2"].iloc[0]).startswith("Abyei"):
        raise SystemExit(f"{ABYEI} is not Abyei in the 2025 workbook")
    abyei = float(ab["Population - 2025"].iloc[0])
    ps = ps[ps["Admin2_Pcode"] != ABYEI]
    cod = dict(zip(a2["adm2_pcode"], a2["adm1_pcode"]))
    if set(ps["Admin2_Pcode"]) != set(cod):
        raise SystemExit(f"county pcodes in one file only: {sorted(set(ps['Admin2_Pcode']) ^ set(cod))}")
    names = dict(zip(a2["adm2_pcode"], a2["adm2_name"]))
    off = {c: (n, names[c]) for c, n in zip(ps["Admin2_Pcode"], ps["Admin2"])
           if fold(n) != fold(names[c])}
    if set(off) - {"SS0304"}:     # Canal/Pigi against COD-AB's Canal-Pigi folds the same
        raise SystemExit(f"county names differ between COD-AB and the 2025 workbook: {off}")
    misfiled = {c: p for c, p in zip(ps["Admin2_Pcode"], ps["Admin1_Pcode"]) if cod[c] != p}
    if misfiled != MISFILED_PARENT:
        raise SystemExit(f"counties the workbook files under another state: {misfiled}, "
                         f"pinned {MISFILED_PARENT}")
    print(f"  2025 county estimates: {N_COUNTIES} counties joined on pcode, names agree; Abyei's "
          f"{abyei:,.0f} dropped; {len(misfiled)} county filed under the wrong state in the workbook "
          f"({', '.join(names[c] for c in misfiled)}), parent taken from COD-AB")
    s = ps.set_index("Admin2_Pcode")["Population - 2025"].astype(float)
    return s, abyei


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    import geo_checks
    import kontur_cap

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()

    z = "zip://" + os.path.join(RAW, SHP_ZIP)
    a1 = gpd.read_file(z + "!ssd_admin1.shp")
    a2 = gpd.read_file(z + "!ssd_admin2.shp")
    if (len(a1), len(a2)) != (N_STATES, N_COUNTIES + 1):
        raise SystemExit(f"COD-AB has {len(a1)}/{len(a2)} features, expected {N_STATES}/{N_COUNTIES + 1}")
    ab = a2[a2["adm1_pcode"] == "SS00"]
    if list(ab["adm2_pcode"]) != [ABYEI] or ab["adm2_name"].iloc[0] != "Abyei":
        raise SystemExit(f"COD-AB admin2's one unit outside the states is not {ABYEI} Abyei: "
                         f"{ab[['adm2_pcode', 'adm2_name']].to_dict('records')}")
    a2 = a2[a2["adm2_pcode"] != ABYEI].copy()
    print("  dropped admin2 SS0001 Abyei (parent SS00 `Abyei Region`, in no state of admin1)")
    if set(a1["version"]) != {"v03"} or set(a2["version"]) != {"v03"}:
        raise SystemExit(f"COD-AB version {set(a1['version'])}, expected v03")
    if set(a1["adm1_pcode"]) != {f"SS{i:02d}" for i in range(1, 11)}:
        raise SystemExit(f"COD-AB states are not SS01-SS10: {sorted(a1['adm1_pcode'])}")
    if not set(a2["adm1_pcode"]) <= set(a1["adm1_pcode"]):
        raise SystemExit("a COD-AB county has a parent that is not a state")
    print(f"COD-AB South Sudan: {len(a1)} states, {len(a2)} counties, version {a1['version'].iloc[0]}, "
          f"valid_on {a1['valid_on'].iloc[0]}, crs={a1.crs}")

    county_pop, abyei = read_pop_2025(a2)
    target = POP_2025_WITH_ABYEI - abyei

    eq = "EPSG:6933"
    units = a1[["adm1_pcode", "adm1_name", "area_sqkm", "geometry"]].rename(
        columns={"adm1_pcode": "unit", "adm1_name": "name", "area_sqkm": "cod_area"})
    units["area_sqkm"] = units.to_crs(eq).area / 1e6
    off = units[(units["area_sqkm"] / units["cod_area"] - 1).abs() > AREA_TOL]
    if len(off):
        raise SystemExit(f"polygon area disagrees with COD-AB's own area_sqkm: "
                         f"{off[['name', 'area_sqkm', 'cod_area']].to_dict('records')}")
    parent = dict(zip(a2["adm2_pcode"], a2["adm1_pcode"]))
    state_pop = county_pop.groupby(county_pop.index.map(parent)).sum()
    units["geo_id"] = units["unit"]
    units["pop"] = units["unit"].map(state_pop).round().astype("int64")
    drift = int(round(target)) - int(units["pop"].sum())
    if abs(drift) > N_STATES:
        raise SystemExit(f"states sum {units['pop'].sum():,} against {target:,.0f}")

    # ---- the 2022 COD-PS as a per-state witness on the 2025 edition ----
    p22 = pd.read_csv(os.path.join(RAW, POP_2022), encoding="utf-8-sig")
    if int(p22["T_TL"].sum()) != CODPS_2022:
        raise SystemExit(f"COD-PS 2022 sums to {p22['T_TL'].sum():,}, expected {CODPS_2022:,}")
    n22 = dict(zip(p22["ADM1_PCODE"], p22["ADM1_EN"]))
    bad = {c: (n22.get(c), n) for c, n in zip(units["unit"], units["name"]) if fold(n22.get(c)) != fold(n)}
    if bad:
        raise SystemExit(f"COD-PS 2022 names a state pcode differently: {bad}")
    units["codps_2022"] = units["unit"].map(dict(zip(p22["ADM1_PCODE"], p22["T_TL"]))).astype("int64")
    print(f"\n  states: 2025 estimate (Abyei dropped) {units['pop'].sum():,} against COD-PS 2022 "
          f"{CODPS_2022:,}, {units['pop'].sum() / CODPS_2022:.3f}x")
    for r in units.sort_values("unit").itertuples():
        print(f"    {r.unit}  {r.name:<24}{r.pop:>11,}  2022 {r.codps_2022:>11,}  "
              f"{r.pop / r.codps_2022:5.3f}x  {r.area_sqkm:>9,.0f} km2")
    ratio = units["pop"] / units["codps_2022"]
    if ratio.min() < 1.0 or ratio.max() > 1.25:
        raise SystemExit(f"a state's 2025 estimate is outside 1.00-1.25x its 2022 figure: "
                         f"{ratio.min():.3f}-{ratio.max():.3f}")

    # ---- Kontur, joined at county ----
    hexes = geo_checks.read_layer(gpkg, "Kontur SS")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    pts = gpd.GeoDataFrame({"kontur": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(a2.crs)
    joined = gpd.sjoin(pts, a2[["adm2_pcode", "adm1_pcode", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["adm2_pcode"].isna()
    lost = float(pts.loc[outside, "kontur"].sum())
    print(f"\nKontur hexes: {len(hexes):,}, population {pts['kontur'].sum():,.0f}; "
          f"{int(outside.sum()):,} centroids outside every county ({lost:,.0f} people, "
          f"{100 * lost / pts['kontur'].sum():.3f}%) dropped")
    keep = (~outside).to_numpy()
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "adm1_pcode"].to_numpy(),
                            "county": joined.loc[keep, "adm2_pcode"].to_numpy(),
                            "kontur_pop": pts.loc[keep, "kontur"].to_numpy()},
                           geometry=hexes.geometry[keep].to_crs(a2.crs).to_numpy(),
                           crs=a2.crs).reset_index(drop=True)

    per = out.groupby("unit")["kontur_pop"].agg(["size", "sum"])
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    national = float(out["kontur_pop"].sum()) / target
    print(f"  raw Kontur {out['kontur_pop'].sum():,.0f} against the 2025 estimate {target:,.0f}: "
          f"{national:.3f}. Per state (printed; the calibration below replaces it):")
    for r in units.sort_values("pop", ascending=False).itertuples():
        print(f"    {r.unit}  {r.name:<24}{r.pop:>11,}  Kontur {r.kontur_pop:>11,}  "
              f"{r.kontur_pop / r.pop:5.2f}x  {r.hexes:>7,} hexes")

    # ---- Kontur's density cap, read on the RAW layer ----
    raw = out[["unit", "geometry"]].copy()
    raw["pop"] = out["kontur_pop"].to_numpy()
    blk = kontur_cap.find_blocks(raw)
    at_cap = [idx for idx in blk["groups"] if (blk["dens"][idx] >= kontur_cap.AT_CAP).any()]
    print(f"\n  raw Kontur: densest hex {blk['dens'].max():,.0f}/km2; {len(blk['groups'])} blocks over "
          f"{kontur_cap.HI:,.0f}/km2, {len(at_cap)} reaching the cap of {kontur_cap.CAP:,.0f}")
    cname = dict(zip(a2["adm2_pcode"], a2["adm2_name"] + ", " + a2["adm1_name"]))
    ck = out.groupby("county")["kontur_pop"].sum()
    weight = out["kontur_pop"].to_numpy(dtype=float).copy()
    found, unnamed = set(), []
    for idx in at_cap:
        pk = idx[np.argmax(blk["dens"][idx])]
        lon, lat = float(blk["lon"][pk]), float(blk["lat"][pk])
        cty = out["county"].to_numpy()[idx]
        shares = "; ".join(f"{cname[t]} {100 * weight[idx][cty == t].sum() / ck[t]:.0f}%"
                           for t in sorted(set(cty)))
        n_cap = int((blk["dens"][idx] >= kontur_cap.AT_CAP).sum())
        inblock = np.zeros(len(weight), dtype=bool)
        inblock[idx] = True
        near = blk["tree"].query_ball_point(blk["xy"][idx], kontur_cap.RING_KM * 1000.0)
        ring = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
        ring = ring[~inblock[ring] & (weight[ring] > 0)]
        ring_med = float(np.median(blk["dens"][ring])) if len(ring) else float("nan")
        print(f"    block of {len(idx)} hexes ({n_cap} at the cap) peaking at ({lon:.4f}, {lat:.4f}), "
              f"{blk['pop'][idx].sum():,.0f} people, ring of {len(ring)} populated hexes at a median "
              f"{ring_med:,.0f}/km2; share of each county's Kontur: {shares}")
        key = next((k for k in BLOCKS if km(k[1], k[0], lat, lon) <= 1.0), None)
        if key is None:
            unnamed.append((round(lon, 4), round(lat, 4)))
            continue
        found.add(key)
        action, name = BLOCKS[key]
        if action == "left":
            print(f"      {name}: left as Kontur has it (BLOCKS)")
            continue
        if action != "capped":
            raise SystemExit(f"BLOCKS action {action!r} is neither `capped` nor `left`")
        if len(ring) == 0:
            raise SystemExit(f"{name} has no populated hex in its ring")
        weight[idx] = np.minimum(weight[idx], ring_med * blk["area"][idx])
        print(f"      {name}: lowered to the {kontur_cap.RING_KM:g} km ring's median of "
              f"{ring_med:,.0f}/km2, now {weight[idx].sum():,.0f} people")
    if unnamed:
        raise SystemExit(f"raw Kontur blocks at the cap not in BLOCKS, review and name them: {unnamed}")
    if found != set(BLOCKS):
        raise SystemExit(f"BLOCKS rows matching no block at the cap: {sorted(set(BLOCKS) - found)}")

    # ---- calibrate every hex to its county's 2025 estimate ----
    out["capped"] = weight
    wt = out.groupby("county")["capped"].sum()
    if (wt.reindex(county_pop.index).fillna(0) <= 0).any():
        raise SystemExit("a county has no Kontur weight to calibrate")
    factor = county_pop / wt.reindex(county_pop.index)
    rel = factor / (1 / national)
    holes = set(rel[rel > HOLE_FACTOR].index)
    if holes != EXPECT_HOLES:
        raise SystemExit(f"counties over {HOLE_FACTOR:g}x the national factor are {sorted(holes)}, "
                         f"pinned as {sorted(EXPECT_HOLES)}; update EXPECT_HOLES after reading them")
    out["pop"] = out["capped"] * out["county"].map(factor)
    n_hex = out.groupby("county").size()
    in_hole = out["county"].isin(holes).to_numpy()
    out.loc[in_hole, "pop"] = out.loc[in_hole, "county"].map(county_pop / n_hex).to_numpy()

    chk = out.groupby("unit")["pop"].sum()
    want = state_pop
    worst = float((chk.reindex(want.index) - want).abs().max())
    if worst > 1.0:
        raise SystemExit(f"calibrated states miss the 2025 estimate by up to {worst:,.2f} people")
    q = rel.quantile([0.1, 0.5, 0.9])
    print(f"\n  calibrated to the {N_COUNTIES} counties: every state's weight equals its 2025 estimate "
          f"(worst {worst:.4f}). Scale factor over the national one: p10 {q[0.1]:.2f}, median "
          f"{q[0.5]:.2f}, p90 {q[0.9]:.2f}; over 3x or under 0.33x:")
    for c, v in pd.concat([rel[rel > 3], rel[rel < 1 / 3]]).sort_values(ascending=False).items():
        how = "even over Kontur's hexes" if c in holes else "scaled"
        print(f"    {c:<8}{cname[c]:<36}2025 {county_pop[c]:>10,.0f}  raw Kontur {wt[c]:>10,.0f}  "
              f"{v:6.2f}  {how} ({n_hex[c]:,} hexes)")
    dens = out["pop"].to_numpy() / (out.to_crs(eq).geometry.area.to_numpy() / 1e6)
    top = int(np.argmax(dens))
    print(f"  calibrated densest hex {dens.max():,.0f}/km2 in {cname[out['county'].iloc[top]]} "
          f"({'above' if dens.max() > kontur_cap.OVER_CAP else 'not above'} Kontur's limit, so "
          f"kontur_cap.py {'skips' if dens.max() > kontur_cap.OVER_CAP else 'checks'} this layer)")

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    cols = ["geo_id", "unit", "name", "pop", "codps_2022", "kontur_pop", "hexes", "area_sqkm", "geometry"]
    units = units.sort_values("unit")
    units[cols].to_file(OUT_UNITS, layer="units", driver="GPKG")
    pd.DataFrame(units[cols].drop(columns="geometry")).to_csv(OUT_LOOKUP, index=False, encoding="utf-8")
    t = pd.DataFrame({"county": county_pop.index, "name": [cname[c] for c in county_pop.index],
                      "unit": [parent[c] for c in county_pop.index],
                      "pop_2025": county_pop.round().astype("int64").to_numpy(),
                      "kontur_raw": ck.reindex(county_pop.index).round().to_numpy(),
                      "factor_over_national": rel.reindex(county_pop.index).round(3).to_numpy(),
                      "placement": ["even" if c in holes else "scaled" for c in county_pop.index]})
    t.sort_values("county").to_csv(OUT_COUNTIES, index=False, encoding="utf-8")
    out[["unit", "county", "kontur_pop", "pop", "geometry"]].to_file(OUT_HEXES, layer="hexes",
                                                                     driver="GPKG")
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_LOOKUP}\nwrote {OUT_COUNTIES}\nwrote {OUT_HEXES} "
          f"({len(out):,} hexes)")


if __name__ == "__main__":
    main()
