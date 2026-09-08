"""Indonesia — boundaries for the SP2010 kabupaten/kota.

Writes data/geo/id/id_regencies.gpkg and data/geo/id/id_lookup.csv.

OCHA COD-AB Indonesia, from HDX. **The join is by CODE and it is exact**: COD's
`adm2_pcode` is the string `ID` followed by BPS's own 4-digit kode wilayah, which is the
same identifier `sources/id.py` reads out of the census payload. Not one name has to be
matched, which after Ghana's `TMA` (§9n) and Sri Lanka's shared keys is worth saying
plainly.

**READ IT WITH `engine="fiona"`.** `geopandas.read_file` on this geodatabase returns
**522 features under fiona and ZERO under pyogrio** — same file, same `OpenFileGDB` driver,
one call apart, no exception either way; pyogrio reports `EPSG:4326` and a `geometry`-only
column list, which is exactly Chile's symptom in §12. pyogrio is geopandas' default
whenever it is installed, so the failing path is the one you get for free. The feature count
is asserted after the read regardless — that assertion is the only reason the engine was
ever suspected.

**AND EXTRACT THE ZIP FIRST.** Reading the same geodatabase through `zip://` returns zero
features under *both* engines. Two independent ways to get silence out of a good file.

**§8.1: THE VINTAGE IS WRONG AND IT IS VISIBLE.** COD is `valid_on 2020-04-01` and carries
**522 ADM2 units** against SP2010's 498. Indonesia kept splitting regencies after the
census — Kalimantan Utara became a province in 2012, and a dozen-odd kabupaten were carved
out of others — so a 2020 polygon for a 2010 unit can be SMALLER than the unit the census
counted, and the territory that split away has no census row to fill it. Every 2010 code
matches a polygon, so the join looks perfect; the damage is that some polygons are the wrong
shape and some are unpainted. `report()` names them rather than hiding them.

**The exact fix is the kecamatan layer, not a better ADM2 file.** Sub-district codes are
`regency(4) + kecamatan(3)`, and a new regency is carved out of whole kecamatan — so
dissolving COD's ADM3 by the FIRST FOUR DIGITS OF THE 2010 KECAMATAN CODE reconstructs the
2010 regency exactly, from BPS's own geography rather than from a guess about parentage.
That needs `id.py`'s kecamatan pull, which is the next thing to do anyway.

Usage:
    python sources/id_geo.py --fetch    one 218 MB zip from HDX, then extract
    python sources/id_geo.py            build from data/raw/id_geo/
"""

import csv
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "id_geo")
OUT_DIR = os.path.join(ROOT, "data", "geo", "id")
OUT = os.path.join(OUT_DIR, "id_regencies.gpkg")
OUT_KEC = os.path.join(OUT_DIR, "id_kecamatan.gpkg")
LOOKUP_KEC = os.path.join(OUT_DIR, "id_kecamatan_lookup.csv")
OUT_DRAWN = os.path.join(OUT_DIR, "id_drawn.gpkg")
LOOKUP_DRAWN = os.path.join(OUT_DIR, "id_drawn_lookup.csv")
LOOKUP = os.path.join(OUT_DIR, "id_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "id.csv")

ZIP_URL = ("https://data.humdata.org/dataset/84a1d98a-790b-4d66-9d14-bbfa48500802/"
           "resource/c740a308-0a63-46d6-ab15-b041e62eff58/download/"
           "idn_admin_boundaries.gdb.zip")
ZIP_NAME = "idn_admin_boundaries.gdb.zip"
EXTRACT = os.path.join(RAW, "extracted")
LAYER = "idn_admin2"
LAYER_ADM3 = "idn_admin3"
PCODE_PREFIX = "ID"
EXPECTED_FEATURES = 522
ENGINE = "fiona"          # see the module docstring; pyogrio silently returns nothing

# One drawn unit is not a regency and has no polygon of its own: `65` is Kalimantan Utara,
# recovered as Kalimantan Timur's per-category residual (sources/id.py PROVINCE_RESIDUAL).
# Its geometry is the union of the five regencies that became the province in 2012, which
# is exactly the 2010 territory the residual counts, because they were carved wholly out of
# Kalimantan Timur and nothing else joined them.
RESIDUAL_MEMBERS = {"65": ["6501", "6502", "6503", "6504", "6571"]}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 200_000_000):
        print("GET", ZIP_URL[:100])
        r = requests.get(ZIP_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "religiondots/1.0 (map research)"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(dest):,} bytes")

    # §5a: assert it is a zip, not an HTML error page delivered with a 200
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- {open(dest,'rb').read(80)!r}")

    if not os.path.isdir(EXTRACT):
        print("extracting ...")
        with zipfile.ZipFile(dest) as z:
            z.extractall(EXTRACT)
    print("extracted to", EXTRACT)


def _gdb_path():
    if not os.path.isdir(EXTRACT):
        raise SystemExit(f"missing {EXTRACT} -- run with --fetch first")
    for name in os.listdir(EXTRACT):
        if name.endswith(".gdb"):
            return os.path.join(EXTRACT, name)
    raise SystemExit(f"no .gdb directory inside {EXTRACT}")


def read_boundaries():
    import geopandas as gpd

    gdb = _gdb_path()
    gdf = gpd.read_file(gdb, layer=LAYER, engine=ENGINE)

    # The assertion that matters (§12, Chile and now Indonesia). A zero-feature read here
    # raises nothing at all and reports a plausible CRS.
    if len(gdf) == 0:
        raise SystemExit(
            f"{LAYER} returned ZERO features with engine={ENGINE!r}. The file is probably "
            "fine and the reader is not -- try the other engine before re-downloading.")
    if len(gdf) != EXPECTED_FEATURES:
        print(f"  NOTE {len(gdf)} features, expected {EXPECTED_FEATURES} -- COD has "
              "released a new vintage; the §8.1 discussion in the docstring needs redoing")
    if "adm2_pcode" not in gdf.columns:
        raise SystemExit(f"no adm2_pcode in {list(gdf.columns)}")

    bad = gdf[~gdf["adm2_pcode"].astype(str).str.match(rf"^{PCODE_PREFIX}\d{{4}}$")]
    if len(bad):
        raise SystemExit(f"{len(bad)} pcodes are not {PCODE_PREFIX}+4 digits: "
                         f"{sorted(bad['adm2_pcode'].astype(str))[:8]}")

    gdf["unit"] = gdf["adm2_pcode"].astype(str).str[len(PCODE_PREFIX):]
    if gdf["unit"].duplicated().any():
        dup = sorted(gdf.loc[gdf["unit"].duplicated(), "unit"])
        raise SystemExit(f"duplicate codes in the boundary file: {dup}")

    empty = int(gdf.geometry.is_empty.sum()) + int(gdf.geometry.isna().sum())
    if empty:
        raise SystemExit(f"{empty} empty or null geometries")
    print(f"  {len(gdf)} polygons, crs={gdf.crs}, engine={ENGINE}")
    return gdf


def read_census(*levels):
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/id.py first")
    units = {}
    with open(NORM, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if levels and row["geo_level"] not in levels:
                continue
            units[row["geo_id"]] = row["geo_name"]
    return units


def read_kecamatan_boundaries():
    """COD ADM3, the sub-district tier. Same engine trap as ADM2 — see the docstring."""
    import geopandas as gpd

    gdf = gpd.read_file(_gdb_path(), layer=LAYER_ADM3, engine=ENGINE)
    if len(gdf) == 0:
        raise SystemExit(
            f"{LAYER_ADM3} returned ZERO features with engine={ENGINE!r} -- try the other "
            "engine before re-downloading, the file is probably fine")
    gdf["unit"] = gdf["adm3_pcode"].astype(str).str[len(PCODE_PREFIX):]
    bad = gdf[~gdf["unit"].str.match(r"^\d{7}$")]
    if len(bad):
        raise SystemExit(f"{len(bad)} ADM3 pcodes are not 7 digits")

    # A kecamatan's code carries its regency in the first four digits, and COD agrees with
    # itself on that for all 7,069 — checked rather than assumed, because the whole 2010
    # reconstruction rests on the prefix meaning what it says.
    off = gdf[gdf["unit"].str[:4] != gdf["adm2_pcode"].astype(str).str[len(PCODE_PREFIX):]]
    if len(off):
        raise SystemExit(f"{len(off)} ADM3 codes disagree with their own adm2_pcode")
    print(f"  {len(gdf)} sub-district polygons, crs={gdf.crs}")
    return gdf


def report_kecamatan(gdf, census):
    """What the kecamatan tier joins, and what territory it leaves unpainted."""
    b, c = set(gdf["unit"]), set(census)
    print(f"\n  census kecamatan   : {len(c):,}")
    print(f"  boundary polygons  : {len(b):,}")
    print(f"  matched            : {len(b & c):,}")

    missing = sorted(c - b)
    if missing:
        print(f"\n  CENSUS KECAMATAN WITH NO POLYGON ({len(missing)}) -- fatal:")
        for u in missing[:20]:
            print(f"    {u}  {census[u]}")

    extra = gdf[~gdf["unit"].isin(c)].copy()
    if len(extra):
        # Post-2010 sub-districts. Their area is the honest measure of what the vintage
        # mismatch costs, and it is much smaller here than at ADM2 because a kecamatan
        # split moves far less ground than a regency split.
        area = extra.to_crs(3857).area.sum() / 1e6
        total = gdf.to_crs(3857).area.sum() / 1e6
        by_prov = extra["unit"].str[:2].value_counts().head(8).to_dict()
        print(f"\n  POLYGONS WITH NO CENSUS ROW: {len(extra):,} "
              f"({100*len(extra)/len(gdf):.1f}% of polygons, "
              f"{100*area/total:.1f}% of area)")
        print(f"  these are post-2010 sub-districts; by province: {by_prov}")
        kaltara = extra[extra["unit"].str[:2] == "65"]
        if len(kaltara):
            print(f"  of which {len(kaltara)} are Kalimantan Utara, which has no census "
                  "data on this route at all (id.py's UNRECOVERABLE)")
    return not missing


def report(gdf, census):
    b = set(gdf["unit"])
    c = set(census)
    print(f"\n  census regencies : {len(c)}")
    print(f"  boundary polygons: {len(b)}")
    print(f"  matched          : {len(b & c)}")

    missing = sorted(c - b)
    if missing:
        print(f"\n  CENSUS UNITS WITH NO POLYGON ({len(missing)}) -- this is fatal:")
        for u in missing:
            print(f"    {u}  {census[u]}")

    extra = sorted(b - c)
    if extra:
        names = dict(zip(gdf["unit"], gdf["adm2_name"]))
        print(f"\n  POLYGONS WITH NO CENSUS ROW ({len(extra)}) -- §8.1 vintage drift.")
        print("  These are post-2010 splits and COD's water/forest polygons. Their "
              "territory is\n  unpainted, and their 2010 parent is drawn too small. "
              "See the docstring.")
        for u in extra:
            print(f"    {u}  {names.get(u, '?')}")
    return not missing


def write(gdf, census):
    import geopandas as gpd

    keep = gdf[gdf["unit"].isin(census)].copy()
    # The sixteen regencies rebuilt from a province residual have no name in the census —
    # the residual is arithmetic and carries none — so id.py writes `[code]` as a
    # placeholder. Take COD's name for those rather than putting "[5107]" on the map.
    keep["name"] = [census[u] if not census[u].startswith("[") else n
                    for u, n in zip(keep["unit"], keep["adm2_name"])]
    keep["pcode"] = keep["adm2_pcode"]
    os.makedirs(OUT_DIR, exist_ok=True)
    keep[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="regencies", driver="GPKG")
    print(f"\nwrote {OUT} ({len(keep)} polygons)")

    # geo_id and unit are the same 4-digit BPS code here -- the census and the boundary
    # file genuinely share an identifier, which is rare enough on this map to be worth a
    # column rather than an assumption downstream.
    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit", "name", "pcode", "adm1_pcode", "adm1_name"])
        for _, r in keep.sort_values("unit").iterrows():
            w.writerow([r["unit"], r["unit"], r["name"], r["pcode"],
                        r.get("adm1_pcode", ""), r.get("adm1_name", "")])
    print(f"wrote {LOOKUP}")


def write_kecamatan(gdf, census):
    import geopandas as gpd

    keep = gdf[gdf["unit"].isin(census)].copy()
    keep["name"] = keep["unit"].map(census)
    keep["pcode"] = keep["adm3_pcode"]
    keep["regency"] = keep["unit"].str[:4]
    os.makedirs(OUT_DIR, exist_ok=True)
    keep[["unit", "name", "pcode", "regency", "geometry"]].to_file(
        OUT_KEC, layer="kecamatan", driver="GPKG")
    print(f"\nwrote {OUT_KEC} ({len(keep):,} polygons)")

    with open(LOOKUP_KEC, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit", "name", "pcode", "regency", "adm1_pcode"])
        for _, r in keep.sort_values("unit").iterrows():
            w.writerow([r["unit"], r["unit"], r["name"], r["pcode"], r["regency"],
                        r.get("adm1_pcode", "")])
    print(f"wrote {LOOKUP_KEC}")


def write_drawn(adm2, adm3, drawn_reg, drawn_kec):
    """One layer for the DRAWN tier: kecamatan where they replace their regency, regency
    where they do not. Disjoint by construction, and asserted to be.

    `id.py` decides which is which per unit and records it in `geo_level`; this file only
    has to fetch the right polygon for each and check that the two sets do not overlap —
    a kecamatan drawn inside a regency that is ALSO drawn would double the people there,
    and nothing downstream would notice.
    """
    import geopandas as gpd
    import pandas as pd

    k = adm3[adm3["unit"].isin(drawn_kec)].copy()
    k["name"] = k["unit"].map(drawn_kec)
    k["level"] = "kecamatan"
    k["regency"] = k["unit"].str[:4]

    # The residual unit first: dissolve its member regencies into one polygon.
    import pandas as _pd
    extra = []
    for code, members in RESIDUAL_MEMBERS.items():
        if code not in drawn_reg:
            continue
        part = adm2[adm2["unit"].isin(members)]
        if len(part) != len(members):
            raise SystemExit(f"{code}: {len(part)} of {len(members)} member polygons found "
                             f"({sorted(set(members) - set(part['unit']))} missing)")
        extra.append({"unit": code, "name": drawn_reg[code], "level": "province_residual",
                      "regency": code, "geometry": part.geometry.union_all()})
        print(f"  dissolved {len(part)} regencies into {code} {drawn_reg[code]}")
    drawn_reg = {k: v for k, v in drawn_reg.items() if k not in RESIDUAL_MEMBERS}

    r = adm2[adm2["unit"].isin(drawn_reg)].copy()
    r["name"] = [drawn_reg[u] if not drawn_reg[u].startswith("[") else n
                 for u, n in zip(r["unit"], r["adm2_name"])]
    r["level"] = "regency"
    r["regency"] = r["unit"]

    overlap = set(k["regency"]) & set(r["unit"])
    if overlap:
        raise SystemExit(f"{len(overlap)} regencies are drawn AND have drawn kecamatan: "
                         f"{sorted(overlap)[:5]} -- the tiers are not disjoint")
    missing_k = sorted(set(drawn_kec) - set(k["unit"]))
    missing_r = sorted(set(drawn_reg) - set(r["unit"]))
    if missing_k or missing_r:
        raise SystemExit(f"no polygon for {len(missing_k)} kecamatan and "
                         f"{len(missing_r)} regencies: {(missing_k + missing_r)[:5]}")

    parts = [k[["unit", "name", "level", "regency", "geometry"]],
             r[["unit", "name", "level", "regency", "geometry"]]]
    if extra:
        parts.append(gpd.GeoDataFrame(extra, crs=adm2.crs))
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=adm3.crs)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT_DRAWN, layer="drawn", driver="GPKG")
    print(f"\nwrote {OUT_DRAWN} -- {len(out):,} polygons "
          f"({len(k):,} kecamatan + {len(r):,} regencies)")

    with open(LOOKUP_DRAWN, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit", "name", "level", "regency"])
        for _, x in out.sort_values("unit").iterrows():
            w.writerow([x["unit"], x["unit"], x["name"], x["level"], x["regency"]])
    print(f"wrote {LOOKUP_DRAWN}")


def main():
    if "--fetch" in sys.argv:
        fetch()
        return

    gdf = read_boundaries()
    census = read_census("regency", "regency_covered")
    ok = report(gdf, census)
    if not ok:
        raise SystemExit("a census unit has no polygon -- not writing")
    write(gdf, census)

    drawn_reg = read_census("regency", "province_residual")
    drawn_kec = read_census("kecamatan")
    if drawn_kec:
        print()
        write_drawn(gdf, read_kecamatan_boundaries(), drawn_reg, drawn_kec)

    kec_census = read_census("kecamatan", "kecamatan_partial")
    if kec_census:
        print("\n=== sub-district tier ===")
        k = read_kecamatan_boundaries()
        if not report_kecamatan(k, kec_census):
            raise SystemExit("a census kecamatan has no polygon -- not writing")
        write_kecamatan(k, kec_census)
    else:
        print("\nno kecamatan rows in id.csv -- run sources/id.py --fetch-all first")


if __name__ == "__main__":
    main()
