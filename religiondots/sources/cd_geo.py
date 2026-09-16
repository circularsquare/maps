"""DR Congo: 26 provinces from COD-AB v01, and a Kontur grid calibrated to COD-PS 2024 by territoire.

Writes:
    data/geo/cd/cd_units.gpkg          the 26 provinces (`units`)
    data/geo/cd/cd_hexes.gpkg          Kontur 400 m hexes with `unit`, `territoire`, `kontur_pop`
                                       and `pop` (`place`); `pop` is calibrated to the territoire
    data/geo/cd/cd_lookup.csv          province -> name, COD-PS 2024, raw Kontur
    data/geo/cd/cd_territoires.csv     territoire -> province, COD-PS 2024, raw Kontur, scale factor

Usage:
    python sources/cd_geo.py --fetch    COD-AB shapefile zip (~22 MB), Kontur (~25 MB), COD-PS 2024
    python sources/cd_geo.py            rebuild from data/raw/cd/

## BOUNDARIES: OCHA COD-AB DR CONGO (`cod-ab-cod`), VERSION 01

26 provinces, 164 admin2 units (145 territoires plus 19 cities split out to match the health
zones), 519 health zones; valid on 2019-09-11, reviewed 2024-12-20, republished 2026-04-16. The
province and admin2 pcodes (CD10, CD1000) are the codes COD-PS 2024 carries, so both joins are on
the code and the names are the witness. The health-zone pcodes are not (COD-AB `CD100001`, COD-PS
`CD1000ZS01`), so nothing here reads the health zones.

## POPULATION: COD-PS 2024, BECAUSE THERE IS NO CENSUS

The last census was 1984. `drc-hpc-projection-population-2024.xlsx` is OCHA's humanitarian
projection by health zone, 117,808,872 people, and it is the base `sources/cd.py` lays the shares on.

## WHY THE GRID IS CALIBRATED: KONTUR HAS LOST SANKURU

Raw Kontur (November 2023) holds 102.1 million people inside the provinces, 0.867 of COD-PS, and
per province it runs from 0.09x (Sankuru) to 1.61x (Haut-Lomami); Kinshasa 0.47x, Kongo-Central
0.35x. One level down it is worse and it is local: Sankuru's territoires read 0.02 to 0.08 of the
national ratio (Lubefu 9,776 people against 665,858; Lodja 32,134 against 733,632), with hundreds
of hexes each, while Likasi reads 4.25 and Kipushi 3.38. The counts are at province and exact
either way; what is at stake is which territoire inside a province the dots fall in. So every hex
keeps Kontur's shape inside its territoire and takes the territoire's COD-PS 2024 total (Iran's
construction, `sources/ir_geo.py`; spec §12's Haiti rule, a false unit is scaled, not capped).

## WHERE KONTUR HAS LOST A TERRITOIRE, ITS SHAPE IS NOT SCALED UP

The first calibrated build scaled Lubefu's hexes 59 times the national factor and drew its densest
hex at 214,456/km2: whatever faint spot Kontur kept became a false city of the whole territoire's
people (Mauritania's trap, `playbooks/geography.md`). A territoire whose factor is over
`HOLE_FACTOR` times the national one keeps only Kontur's footprint, the hexes it marks as settled,
and its COD-PS total is spread evenly over them. Five territoires, all in Sankuru, pinned in
`EXPECT_HOLES`; the next largest factor is Moanda's 4.9.

## THE CAP BLOCKS ARE READ ON RAW KONTUR

A calibrated layer goes above Kontur's 46,200/km2 limit, and `kontur_cap.apply` then skips it, so
the scan for blocks at the cap runs here on Kontur's own figures before calibration (Mauritania's
order, `sources/mr_grid.py`). Every block at the cap must be named in `BLOCKS`.
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

RAW = os.path.join(ROOT, "data", "raw", "cd")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "cd")
OUT_UNITS = os.path.join(GEO, "cd_units.gpkg")
OUT_HEXES = os.path.join(GEO, "cd_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "cd_lookup.csv")
OUT_TERR = os.path.join(GEO, "cd_territoires.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
KONTUR_GPKG = "kontur_population_CD_20231101.gpkg"
POP_NAME = "drc-hpc-projection-population-2024.xlsx"
DOWNLOADS = {
    "cod_admin_boundaries.shp.zip": (
        "https://data.humdata.org/dataset/f42132b9-8cc6-4201-b020-9259c56e8868/resource/"
        "7514482b-f7af-4654-8ea2-e8a34a6acb6a/download/cod_admin_boundaries.shp.zip",
        b"PK", 10_000_000),
    KONTUR_GPKG + ".gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_CD_20231101.gpkg.gz", b"\x1f\x8b", 10_000_000),
    POP_NAME: (
        "https://data.humdata.org/dataset/d1160fa9-1d58-4f96-9df5-edbff2e80895/resource/"
        "9509b41f-efb2-4305-ac43-1e735e3efc60/download/drc-hpc-projection-population-2024.xlsx",
        b"PK", 500_000),
}

N_PROVINCES = 26
N_ADMIN2 = 164
COD_PS_2024 = 117_808_872
AREA_TOL = 0.02

# A territoire whose calibration factor is over this many times the national one is a Kontur hole:
# its COD-PS total goes evenly over Kontur's hexes there. Measured 2026-09-15: Lubefu 59.1, Lomela
# 50.7, Lodja 19.8, Kole 15.3, Katako-Kombe 11.9; then Moanda 4.9, Katanda 4.5, Matadi 4.5.
HOLE_FACTOR = 10.0
EXPECT_HOLES = {"CD8303", "CD8306", "CD8307", "CD8308", "CD8309"}

# Raw Kontur blocks at the 46,200/km2 cap, by (lon, lat) of the block's peak, matched within 1 km.
# "capped" lowers the block to its 3 km ring's median before calibration (kontur_cap.py's method);
# "left" keeps Kontur's shape. Reviewed 2026-09-15 (cb8b206e-cd) against GeoNames `CD.zip`:
BLOCKS = {
    # 165 hexes, 48 at the cap, 4,287,016 people, 62% of Kinshasa's raw Kontur, peaking 0.8 km from
    # GeoNames' Kinshasa (PPLC). The city's dense communes; a real core.
    (15.3152, -4.3345): ("left", "Kinshasa, the city's central communes"),
    # 101 hexes, 51 at the cap, 3,434,223 people, 96% of Mbuji-Mayi's raw Kontur, 4.5 km from
    # GeoNames' Mbuji-Mayi (PPLA, 2,101,332). The admin2 unit is the city itself; a real core.
    (23.6221, -6.1111): ("left", "Mbuji-Mayi, the city"),
    # 2 hexes, 1 at the cap, 73,449 people, 1.2 km from GeoNames' Kilwa, a Lake Mweru town with no
    # population recorded. Two hexes at the limit with no ramp is the false-block shape, and 73,449
    # on 1.6 km2 is a city's density for a fishing town. Capped; it is 6% of Pweto territoire's
    # Kontur, so the call moves little either way.
    (28.3332, -9.2749): ("capped", "Kilwa, Pweto territoire"),
}


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
    z = os.path.join(RAW, "cod_admin_boundaries.shp.zip")
    if not os.path.exists(z):
        raise SystemExit(f"missing {z}; run with --fetch first")
    if not os.path.exists(os.path.join(SHP_DIR, "cod_admin2.shp")):
        with zipfile.ZipFile(z) as zz:
            for m in zz.namelist():
                if re.match(r"cod_admin[12]\.", m):
                    zz.extract(m, SHP_DIR)
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


def read_pop():
    import pandas as pd

    ps = pd.read_excel(os.path.join(RAW, POP_NAME), header=0)
    ps.columns = [str(c).strip() for c in ps.columns]
    ps["Population 2024"] = pd.to_numeric(ps["Population 2024"], errors="coerce")
    if int(ps["Population 2024"].sum()) != COD_PS_2024:
        raise SystemExit(f"COD-PS 2024 sums to {ps['Population 2024'].sum():,.0f}, expected {COD_PS_2024:,}")
    parent = ps.groupby("Code Terrtoire")["Code Province"].nunique()
    if (parent != 1).any():
        raise SystemExit("a COD-PS territoire sits in two provinces")
    prov = ps.groupby(["Code Province", "Province"])["Population 2024"].sum().reset_index()
    terr = ps.groupby(["Code Terrtoire", "Code Province"])["Population 2024"].sum().reset_index()
    return prov, terr


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    import geo_checks
    import kontur_cap

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    prov_pop, terr_pop = read_pop()

    a1 = gpd.read_file(os.path.join(SHP_DIR, "cod_admin1.shp"))
    a2 = gpd.read_file(os.path.join(SHP_DIR, "cod_admin2.shp"))
    if (len(a1), len(a2)) != (N_PROVINCES, N_ADMIN2):
        raise SystemExit(f"COD-AB has {len(a1)}/{len(a2)} features, expected {N_PROVINCES}/{N_ADMIN2}")
    if set(a1["version"]) != {"v01"} or set(a2["version"]) != {"v01"}:
        raise SystemExit(f"COD-AB version {set(a1['version'])}, expected v01")
    print(f"COD-AB DR Congo: {len(a1)} provinces, {len(a2)} admin2 units, version "
          f"{a1['version'].iloc[0]}, valid_on {a1['valid_on'].iloc[0]}, crs={a1.crs}")

    ab = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    ps = dict(zip(prov_pop["Code Province"], prov_pop["Province"]))
    if set(ab) != set(ps):
        raise SystemExit(f"province pcodes in one file only: {sorted(set(ab) ^ set(ps))}")
    named = {c: (ab[c], ps[c]) for c in ab if fold(ab[c]) != fold(ps[c])}
    if named:
        raise SystemExit(f"COD-AB and COD-PS name a province pcode differently: {named}")
    tp = dict(zip(terr_pop["Code Terrtoire"], terr_pop["Code Province"]))
    if set(tp) != set(a2["adm2_pcode"]):
        raise SystemExit(f"admin2 pcodes in one file only: {sorted(set(tp) ^ set(a2['adm2_pcode']))}")
    wrong = {c: (p, tp[c]) for c, p in zip(a2["adm2_pcode"], a2["adm1_pcode"]) if tp[c] != p}
    if wrong:
        raise SystemExit(f"COD-AB and COD-PS put an admin2 unit in different provinces: {wrong}")
    print("  26 province and 164 admin2 pcodes agree with COD-PS 2024, names and parents included")

    eq = "EPSG:6933"
    units = a1[["adm1_pcode", "adm1_name", "area_sqkm", "geometry"]].rename(
        columns={"adm1_pcode": "unit", "adm1_name": "name", "area_sqkm": "cod_area"})
    units["area_sqkm"] = units.to_crs(eq).area / 1e6
    off = units[(units["area_sqkm"] / units["cod_area"] - 1).abs() > AREA_TOL]
    if len(off):
        raise SystemExit(f"polygon area disagrees with COD-AB's own area_sqkm: "
                         f"{off[['name', 'area_sqkm', 'cod_area']].to_dict('records')}")
    units["geo_id"] = units["unit"]
    units["pop"] = units["unit"].map(dict(zip(prov_pop["Code Province"],
                                              prov_pop["Population 2024"]))).astype("int64")
    if int(units["pop"].sum()) != COD_PS_2024:
        raise SystemExit("the provinces do not sum to COD-PS 2024")

    # ---- Kontur, joined at admin2 ----
    hexes = geo_checks.read_layer(gpkg, "Kontur CD")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    pts = gpd.GeoDataFrame({"kontur": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(a2.crs)
    joined = gpd.sjoin(pts, a2[["adm2_pcode", "adm1_pcode", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["adm2_pcode"].isna()
    lost = float(pts.loc[outside, "kontur"].sum())
    print(f"\nKontur hexes: {len(hexes):,}, population {pts['kontur'].sum():,.0f}; "
          f"{int(outside.sum()):,} centroids outside every admin2 unit ({lost:,.0f} people, "
          f"{100 * lost / pts['kontur'].sum():.3f}%) dropped")
    keep = (~outside).to_numpy()
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "adm1_pcode"].to_numpy(),
                            "territoire": joined.loc[keep, "adm2_pcode"].to_numpy(),
                            "kontur_pop": pts.loc[keep, "kontur"].to_numpy()},
                           geometry=hexes.geometry[keep].to_crs(a2.crs).to_numpy(),
                           crs=a2.crs).reset_index(drop=True)

    per = out.groupby("unit")["kontur_pop"].agg(["size", "sum"])
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    national = float(out["kontur_pop"].sum()) / COD_PS_2024
    print(f"  raw Kontur {out['kontur_pop'].sum():,.0f} against COD-PS 2024 {COD_PS_2024:,}: "
          f"{national:.3f}. Per province (printed, not banded: the calibration below replaces it):")
    for r in units.sort_values("pop", ascending=False).itertuples():
        print(f"    {r.unit}  {r.name:<16}{r.pop:>12,}  Kontur {r.kontur_pop:>12,}  "
              f"{r.kontur_pop / r.pop:5.2f}x  {r.hexes:>7,} hexes  {r.area_sqkm:>9,.0f} km2")

    # ---- Kontur's density cap, read on the RAW layer ----
    raw = out[["unit", "geometry"]].copy()
    raw["pop"] = out["kontur_pop"].to_numpy()
    blk = kontur_cap.find_blocks(raw)
    at_cap = [idx for idx in blk["groups"] if (blk["dens"][idx] >= kontur_cap.AT_CAP).any()]
    print(f"\n  raw Kontur: densest hex {blk['dens'].max():,.0f}/km2; {len(blk['groups'])} blocks over "
          f"{kontur_cap.HI:,.0f}/km2, {len(at_cap)} reaching the cap of {kontur_cap.CAP:,.0f}")
    tname = dict(zip(a2["adm2_pcode"], a2["adm2_name"] + ", " + a2["adm1_name"]))
    tk = out.groupby("territoire")["kontur_pop"].sum()
    weight = out["kontur_pop"].to_numpy(dtype=float).copy()
    found, unnamed = set(), []
    for idx in at_cap:
        pk = idx[np.argmax(blk["dens"][idx])]
        lon, lat = float(blk["lon"][pk]), float(blk["lat"][pk])
        terr = out["territoire"].to_numpy()[idx]
        shares = "; ".join(f"{tname[t]} {100 * weight[idx][terr == t].sum() / tk[t]:.0f}%"
                           for t in sorted(set(terr)))
        n_cap = int((blk["dens"][idx] >= kontur_cap.AT_CAP).sum())
        print(f"    block of {len(idx)} hexes ({n_cap} at the cap) peaking at ({lon:.4f}, {lat:.4f}), "
              f"{blk['pop'][idx].sum():,.0f} people; share of each territoire's Kontur: {shares}")
        key = next((k for k in BLOCKS if km(k[1], k[0], lat, lon) <= 1.0), None)
        if key is None:
            unnamed.append((lon, lat))
            continue
        found.add(key)
        action, name = BLOCKS[key]
        if action == "left":
            print(f"      {name}: left as Kontur has it (BLOCKS)")
            continue
        if action != "capped":
            raise SystemExit(f"BLOCKS action {action!r} is neither `capped` nor `left`")
        inblock = np.zeros(len(weight), dtype=bool)
        inblock[idx] = True
        near = blk["tree"].query_ball_point(blk["xy"][idx], kontur_cap.RING_KM * 1000.0)
        ring = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
        ring = ring[~inblock[ring] & (weight[ring] > 0)]
        if len(ring) == 0:
            raise SystemExit(f"{name} has no populated hex in its ring")
        ceiling = float(np.median(blk["dens"][ring]))
        weight[idx] = np.minimum(weight[idx], ceiling * blk["area"][idx])
        print(f"      {name}: lowered to the {kontur_cap.RING_KM:g} km ring's median of "
              f"{ceiling:,.0f}/km2, now {weight[idx].sum():,.0f} people")
    if unnamed:
        raise SystemExit(f"raw Kontur blocks at the cap not in BLOCKS, review and name them: {unnamed}")
    if found != set(BLOCKS):
        raise SystemExit(f"BLOCKS rows matching no block at the cap: {sorted(set(BLOCKS) - found)}")

    # ---- calibrate every hex to its territoire's COD-PS 2024 total ----
    out["capped"] = weight
    wt = out.groupby("territoire")["capped"].sum()
    target = terr_pop.set_index("Code Terrtoire")["Population 2024"]
    if (wt.reindex(target.index).fillna(0) <= 0).any():
        raise SystemExit("an admin2 unit has no Kontur weight to calibrate")
    factor = target / wt.reindex(target.index)
    rel = factor / (1 / national)
    holes = set(rel[rel > HOLE_FACTOR].index)
    if holes != EXPECT_HOLES:
        raise SystemExit(f"territoires over {HOLE_FACTOR:g}x the national factor are {sorted(holes)}, "
                         f"pinned as {sorted(EXPECT_HOLES)}; re-read the docstring and update EXPECT_HOLES")
    out["pop"] = out["capped"] * out["territoire"].map(factor)
    n_hex = out.groupby("territoire").size()
    in_hole = out["territoire"].isin(holes).to_numpy()
    out.loc[in_hole, "pop"] = out.loc[in_hole, "territoire"].map(target / n_hex).to_numpy()

    chk = out.groupby("unit")["pop"].sum()
    want = units.set_index("unit")["pop"]
    worst = float((chk.reindex(want.index) - want).abs().max())
    if worst > 1.0:
        raise SystemExit(f"calibrated provinces miss COD-PS 2024 by up to {worst:,.2f} people")
    q = rel.quantile([0.1, 0.5, 0.9])
    print(f"\n  calibrated to the 164 territoires: every province's weight equals COD-PS 2024 "
          f"(worst {worst:.4f}). Scale factor over the national one: p10 {q[0.1]:.2f}, median "
          f"{q[0.5]:.2f}, p90 {q[0.9]:.2f}; over 4x in {int((rel > 4).sum())}, under 0.25x in "
          f"{int((rel < 0.25).sum())}:")
    for c, v in pd.concat([rel[rel > 4], rel[rel < 0.25]]).sort_values(ascending=False).items():
        how = "even over Kontur's hexes" if c in holes else "scaled"
        print(f"    {c:<8}{tname[c]:<34}COD-PS {target[c]:>11,.0f}  raw Kontur {wt[c]:>11,.0f}  "
              f"{v:6.2f}  {how} ({n_hex[c]:,} hexes)")
    dens = out["pop"].to_numpy() / (out.to_crs(eq).geometry.area.to_numpy() / 1e6)
    top = int(np.argmax(dens))
    hole_d = dens[in_hole]
    print(f"  calibrated densest hex {dens.max():,.0f}/km2 in {tname[out['territoire'].iloc[top]]} "
          f"({'above' if dens.max() > kontur_cap.OVER_CAP else 'not above'} Kontur's limit, so "
          f"kontur_cap.py {'skips' if dens.max() > kontur_cap.OVER_CAP else 'checks'} this layer); "
          f"in the five holes {hole_d.min():,.0f} to {hole_d.max():,.0f}/km2")

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    cols = ["geo_id", "unit", "name", "pop", "kontur_pop", "hexes", "area_sqkm", "geometry"]
    units[cols].to_file(OUT_UNITS, layer="units", driver="GPKG")
    pd.DataFrame(units[cols].drop(columns="geometry")).to_csv(OUT_LOOKUP, index=False, encoding="utf-8")
    t = pd.DataFrame({"territoire": target.index, "name": [tname[c] for c in target.index],
                      "unit": [tp[c] for c in target.index], "codps_2024": target.to_numpy(),
                      "kontur_raw": tk.reindex(target.index).round().to_numpy(),
                      "factor_over_national": rel.reindex(target.index).round(3).to_numpy(),
                      "placement": ["even" if c in holes else "scaled" for c in target.index]})
    t.to_csv(OUT_TERR, index=False, encoding="utf-8")
    out[["unit", "territoire", "kontur_pop", "pop", "geometry"]].to_file(OUT_HEXES, layer="hexes",
                                                                         driver="GPKG")
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_LOOKUP}\nwrote {OUT_TERR}\nwrote {OUT_HEXES} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
