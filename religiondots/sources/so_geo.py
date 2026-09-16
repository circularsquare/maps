"""Somalia: the 18 regions, COD-AB polygons with the 2026 humanitarian planning population.

Writes data/geo/so/so_regions.gpkg and data/geo/so/so_lookup.csv. `sources/so.md` §4 is the record.

  * **boundaries**: COD-AB `cod-ab-som` v03 (OCHA; valid from 2025-01-08, reviewed 2025-10-30,
    boundaries created 1984-06-23), `som_admin1.geojson`: the 18 regions of 1984, SO11-SO28.
    Somaliland's five regions (Awdal, Woqooyi Galbeed, Togdheer, Sool, Sanaag) are in it as
    regions of Somalia, as in every population source below.
  * **population**: COD-PS `cod-ps-som`, *Somalia District-level population estimates for
    humanitarian response planning, 2026* (version 2026.V1, released 2026-05-25), 90 districts.
    Its read-me: the national figure "was shared by the Government through UNFPA" and is "the
    official population figure"; the district figures are "planning estimates derived through
    remodelling of previous sub-national population data" and "not official district population
    statistics". So the map is drawn at region, the districts summed by p-code.
  * **witness, not used**: UNFPA's Population Estimation Survey (PESS) 2014, 18 regions, from the
    same COD-PS dataset (`somalia-population-statistics.xlsx`). It is the last regional figure a
    field survey measured; the COD-PS notes say its district split was interpolated from 2005 UNDP
    data. Printed beside the 2026 figure.

WHY 2026 AND NOT PESS 2014. Twelve years of displacement and growth (PESS 12,327,528; 2026
19,442,160). Kontur 2023 is no closer to either: half the absolute share difference per region is
0.161 against PESS and 0.154 against 2026 (scratch comparison, 2026-09-15, `sources/so.md` §4).

THE JOIN is on p-code (the district p-code's first four characters are its region's), with the
workbook's own region name for every district as the witness against COD-AB's region name. Inside
Banadir the workbook swaps two pairs of district names against COD-AB (SO2201/SO2202 and
SO2208/SO2209); at region grain that moves nobody, and it is printed.

Usage:
    python sources/so_geo.py --fetch    COD-AB geojson zip, the 2026 workbook and PESS 2014 into data/raw/so/
    python sources/so_geo.py            rebuild from data/raw/so/
"""

import io
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "so")
GEO = os.path.join(ROOT, "data", "geo", "so")
OUT = os.path.join(GEO, "so_regions.gpkg")
LOOKUP = os.path.join(GEO, "so_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
_DS = "https://data.humdata.org/dataset/"
COD_AB_URL = (_DS + "ec140a63-5330-4376-a3df-c7ebf73cfc3c/resource/79f7f826-6028-4650-b8f7-2d5e53032955/"
              "download/som_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "som_admin_boundaries.geojson.zip")
HRP_URL = (_DS + "6cac9c64-2716-4809-9e82-440839d421f6/resource/9096cd51-037b-4c05-b591-ec6777166fbd/"
           "download/somalia-district-level-sex-and-age-disaggregated-population-estimates-for-"
           "humanitarian-response.xlsx")
HRP = os.path.join(RAW, "som_hrp2026_district_pop.xlsx")
HRP_SHEET = "HPC 2026 District Population "
PESS_URL = (_DS + "6cac9c64-2716-4809-9e82-440839d421f6/resource/87bae300-8b66-452a-8da1-439664757ef2/"
            "download/somalia-population-statistics.xlsx")
PESS = os.path.join(RAW, "somalia-population-statistics-2014.xlsx")

EXPECTED_REGIONS = {f"SO{i}" for i in range(11, 29)}
EXPECTED_DISTRICTS = 90
PESS_2014 = 12_327_528           # the 18 region rows summed, asserted
SOMALILAND = {"SO11", "SO12", "SO13", "SO14", "SO15"}
METRIC_AREA = "ESRI:54034"

# District names the 2026 workbook and COD-AB admin2 give the other of a pair, on the same p-code.
# Measured 2026-09-15; both of each pair are in Banadir, so the region sums cannot move.
DISTRICT_NAME_SWAPS = {"SO2201", "SO2202", "SO2208", "SO2209"}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z]", "", s)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst in ((COD_AB_URL, COD_AB), (HRP_URL, HRP), (PESS_URL, PESS)):
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{url} did not return a zip or xlsx (starts {data[:16]!r})")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(data):,} bytes)")


def read_hrp():
    """{district p-code: row} from the 2026 workbook, with the national Total row asserted."""
    raw = pd.read_excel(HRP, sheet_name=HRP_SHEET, header=None)
    hdr = next(i for i in range(10) if str(raw.iloc[i, 0]).strip() == "S/N")
    cols = [str(x).strip() for x in raw.iloc[hdr]]
    body = raw.iloc[hdr + 1:].copy()
    body.columns = cols
    d = body[pd.to_numeric(body["S/N"], errors="coerce").notna()].copy()
    for c in ("State", "Region", "District", "P_Code"):
        d[c] = d[c].astype(str).str.strip()
    d["Total"] = pd.to_numeric(d["Total"]).astype(int)
    if len(d) != EXPECTED_DISTRICTS or d["P_Code"].nunique() != EXPECTED_DISTRICTS:
        raise SystemExit(f"the 2026 workbook has {len(d)} district rows, {d['P_Code'].nunique()} p-codes")
    tot = body[body["S/N"].astype(str).str.strip() == "Total"]
    if len(tot) != 1:
        raise SystemExit(f"the 2026 workbook has {len(tot)} Total rows")
    national = int(pd.to_numeric(tot["Total"]).iloc[0])
    if national != int(d["Total"].sum()):
        raise SystemExit(f"the 2026 districts sum to {int(d['Total'].sum()):,} against its Total row {national:,}")
    print(f"2026 planning estimates: {len(d)} districts, {national:,} people (the Total row, equal to the sum)")
    return d, national


def read_pess():
    p = pd.read_excel(PESS, sheet_name="Region Population Statistics", header=0).iloc[1:]
    p.columns = ["pcode", "name", "total", "urban", "rural", "idp"]
    p = p[p["pcode"].astype(str).str.startswith("SO")].copy()
    p["total"] = pd.to_numeric(p["total"]).round().astype(int)
    if set(p["pcode"]) != EXPECTED_REGIONS or int(p["total"].sum()) != PESS_2014:
        raise SystemExit(f"PESS 2014 is not the 18 regions summing to {PESS_2014:,}: {int(p['total'].sum()):,}")
    return p.set_index("pcode")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in (COD_AB, HRP, PESS)):
        fetch()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("som_admin1.geojson")))
        a2 = gpd.read_file(io.BytesIO(z.read("som_admin2.geojson")))
    print(f"COD-AB admin1: {len(g)} features; admin2: {len(a2)}")
    if set(g["adm1_pcode"]) != EXPECTED_REGIONS or len(g) != 18:
        raise SystemExit(f"COD-AB admin1 is not SO11-SO28: {sorted(g['adm1_pcode'])}")

    d, national = read_hrp()
    d["region"] = d["P_Code"].str[:4]
    if set(d["region"]) != EXPECTED_REGIONS:
        raise SystemExit(f"district p-codes name regions {sorted(set(d['region']))}")
    # witness 1: the workbook's own Region column against COD-AB's name for the p-code's region
    ab1 = dict(zip(g["adm1_pcode"], g["adm1_name"]))
    bad = d[[fold(r) != fold(ab1[p]) for r, p in zip(d["Region"], d["region"])]]
    if len(bad):
        raise SystemExit(f"districts whose Region disagrees with their p-code's region:\n{bad[['District', 'Region', 'P_Code']]}")
    # witness 2: every district p-code is a COD-AB admin2 in the same region
    ab2 = a2.set_index("adm2_pcode")
    missing = sorted(set(d["P_Code"]) - set(ab2.index))
    if missing:
        raise SystemExit(f"2026 districts with no COD-AB admin2: {missing}")
    wrong_parent = [p for p in d["P_Code"] if ab2.loc[p, "adm1_pcode"] != p[:4]]
    if wrong_parent:
        raise SystemExit(f"2026 districts whose COD-AB parent is another region: {wrong_parent}")
    swaps = {p for p, n in zip(d["P_Code"], d["District"]) if fold(n) != fold(ab2.loc[p, "adm2_name"])}
    if swaps != DISTRICT_NAME_SWAPS:
        raise SystemExit(f"district names differing on p-code: {sorted(swaps)}, pinned {sorted(DISTRICT_NAME_SWAPS)}")
    print(f"  {len(d)} districts joined on p-code; every Region column agrees with COD-AB; "
          f"district names swapped inside Banadir on {sorted(swaps)} (region sums unaffected)")
    extra = a2[~a2["adm2_pcode"].isin(d["P_Code"])]
    for _i, r in extra.iterrows():
        print(f"  COD-AB admin2 with no 2026 row: {r['adm2_pcode']} {r['adm2_name']} in {r['adm1_name']}, "
              f"{r['area_sqkm']:.1f} km2 (inside its region's polygon, so nothing is lost at region grain)")

    states = d.groupby("region")["State"].agg(lambda s: "/".join(sorted(set(s))))
    pop = d.groupby("region")["Total"].sum().astype(int)
    pess = read_pess()
    g["unit"] = g["adm1_pcode"]
    g["name"] = g["adm1_name"]
    g["pop"] = g["unit"].map(pop).astype(int)
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("\n  region, state in the workbook, 2026, PESS 2014, 2026 share / PESS share, km2, people per km2:")
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda t: t[0][1]["unit"]):
        u = r["unit"]
        rel = (r["pop"] / national) / (pess.loc[u, "total"] / PESS_2014)
        print(f"      {u}  {r['name']:<16} {states[u]:<21} {r['pop']:>10,} {pess.loc[u, 'total']:>10,}  "
              f"{rel:5.2f}  {a:>8,.0f}  {r['pop'] / a:>7.1f}")
    sl = int(g.loc[g["unit"].isin(SOMALILAND), "pop"].sum())
    print(f"  Somaliland's five regions: {sl:,} ({sl / national:.1%}); PESS 2014 "
          f"{int(pess.loc[sorted(SOMALILAND), 'total'].sum()):,}")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="regions", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (18 regions, {int(g['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
