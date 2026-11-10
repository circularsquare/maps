"""T-2964: Palestine's people by governorate for admin_totals "PS": PCBS's mid-year estimates by governorate (2017 census
base) on COD-AB PSE's 16 governorates, whose Jerusalem holds East Jerusalem as PCBS's does (J1 and J2).

PCBS counts no Israeli settler, but GHS-POP has them where they live, so the CBS 2022 census units beyond the Green Line
with a Jewish majority (tools/country/israel_census.py's il_beyond_line_2022.gpkg: the settlements and East Jerusalem's
Israeli neighbourhoods) are cut out of the governorates and made units of their own, else the governorates' totals
would be spread over them. CBS draws 119 small localities as 0.008 km2 placeholders; those are discs at 4,000 people a
km2 (religiondots/sources/ps.md 6). `--settlements` sets their people (Anita's call, notes/T-2964.md):
- empty: 0, part of neither country's people (the default until she decides)
- counted: CBS's 2022 census count
`--gaza` sets the Gaza Strip's (notes/T-2964.md):
- pcbs2025: PCBS's end-2025 estimate for the Strip, 2,127,034 (press release of 2026-05-12, about 10% under the
  projection after the war), split by the five governorates' mid-2025 projections (the default)
- projection: the mid-2025 projections as published

    python tools/country/palestine_census.py [--settlements empty|counted] [--gaza pcbs2025|projection]

Reads data/raw/admin_totals/PS/pcbs_gov_1997_2026.csv (PCBS "Estimated Population in Palestine Mid-Year by
Governorate, 1997-2026", www.pcbs.gov.ps/papers/statisticsIndicatorsTables_uploads/en2021-5-26-11-11-5-gov.csv: the host
sends an incomplete certificate chain, so curl refuses it; save it from a browser) and COD-AB PSE (HDX, CC BY-IGO;
religiondots' copy when there). Writes data/raw/admin_totals/PS/ps_units_2025.gpkg, layer `units` (code, name, pop).
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import zipfile

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = os.path.dirname(TW)
DIR = os.path.join(TW, "data", "raw", "admin_totals", "PS")
PCBS = os.path.join(DIR, "pcbs_gov_1997_2026.csv")
PCBS_URL = "https://www.pcbs.gov.ps/papers/statisticsIndicatorsTables_uploads/en2021-5-26-11-11-5-gov.csv"
COD = os.path.join(DIR, "pse_admin_boundaries.geojson.zip")
COD_MIRROR = os.path.join(MAPS, "religiondots", "data", "raw", "ps", "pse_admin_boundaries.geojson.zip")
COD_URL = ("https://data.humdata.org/dataset/2caf8373-816f-458c-9913-71bddb9cab7c/resource/ca372385-4c79-4378-abf1-cb506fb98023/"
           "download/pse_admin_boundaries.geojson.zip")
BEYOND = os.path.join(TW, "data", "raw", "admin_totals", "IL", "il_beyond_line_2022.gpkg")
OUT = os.path.join(DIR, "ps_units_2025.gpkg")
YEAR = "2025"
GAZA_END_2025 = 2_127_034  # PCBS, Nakba press release 2026-05-12: 3.43M in the West Bank and 2.13M in the Gaza Strip at end-2025
PLACEHOLDER_KM2 = 0.01
DISC_DENSITY = 4000.0  # people a km2
# PCBS's names to COD-AB's pcodes (six of sixteen spelled differently)
PCODE = {"Jenin": "PS0101", "Tubas & the Northern Valleys": "PS0105", "Tulkarm": "PS0110", "Nablus": "PS0115",
         "Qalqiliya": "PS0120", "Salfit": "PS0125", "Ramallah & Al-Bireh": "PS0130", "Jericho & Al -Aghwar": "PS0135",
         "Jerusalem": "PS0140", "Bethlehem": "PS0145", "Hebron": "PS0150", "North Gaza": "PS0255", "Gaza": "PS0260",
         "Dier al Balah": "PS0265", "Khan Yunis": "PS0270", "Rafah": "PS0275"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--settlements", choices=["empty", "counted"], default="empty")
    ap.add_argument("--gaza", choices=["pcbs2025", "projection"], default="pcbs2025")
    a = ap.parse_args()
    import geopandas as gpd
    import pandas as pd
    import shapely
    from shapely.geometry import shape
    if not os.path.exists(PCBS):
        raise SystemExit(f"{PCBS}: missing; save {PCBS_URL} from a browser")
    rows = {r["Governorate"].strip(): r for r in csv.DictReader(open(PCBS, encoding="utf-8-sig"))}
    pop = {PCODE[n]: float(rows[n][YEAR]) for n in PCODE}
    wb = sum(v for k, v in pop.items() if k.startswith("PS01"))
    gz = sum(v for k, v in pop.items() if k.startswith("PS02"))
    for name, total, parts in (("West Bank", wb, "PS01"), ("Gaza Strip", gz, "PS02")):
        if abs(float(rows[name][YEAR]) - total) > 0.5:
            raise SystemExit(f"{name} {rows[name][YEAR]} against its governorates' {total:,.0f}")
    if a.gaza == "pcbs2025":
        for k in pop:
            if k.startswith("PS02"):
                pop[k] *= GAZA_END_2025 / gz
    path = COD if os.path.exists(COD) else COD_MIRROR if os.path.exists(COD_MIRROR) else None
    if path is None:
        os.makedirs(DIR, exist_ok=True)
        subprocess.run(["curl", "-sS", "-L", "--fail", "-o", COD, COD_URL], check=True)
        path = COD
    fc = json.loads(zipfile.ZipFile(path).read("pse_admin2.geojson"))
    gov = gpd.GeoDataFrame([{"code": f["properties"]["adm2_pcode"], "name": f["properties"]["adm2_name"],
                             "geometry": shapely.make_valid(shape(f["geometry"]))} for f in fc["features"]], crs=4326)
    if sorted(gov.code) != sorted(pop):
        raise SystemExit(f"COD-AB pcodes {sorted(gov.code)} against PCBS's {sorted(pop)}")
    if not os.path.exists(BEYOND):
        raise SystemExit(f"{BEYOND}: missing; run tools/country/israel_census.py")
    b = gpd.read_file(BEYOND, layer="units").to_crs(4326)
    s = b[b.nonarab > 0.5 * b["pop"]].copy()
    # placeholders: a disc at DISC_DENSITY round the polygon's point, in an equal-area projection
    eq = s.to_crs(6933)
    small = eq.geometry.area < PLACEHOLDER_KM2 * 1e6
    eq.loc[small, "geometry"] = [g.representative_point().buffer(1000 * math.sqrt(max(p, 1) / DISC_DENSITY / math.pi))
                                 for g, p in zip(eq.geometry[small], s["pop"][small])]
    s["geometry"] = eq.to_crs(4326).geometry.to_numpy()
    cut = shapely.union_all(s.geometry.to_numpy())
    gov["geometry"] = [shapely.make_valid(g.difference(cut)) for g in gov.geometry]
    gov["pop"] = gov.code.map(pop)
    s = s.assign(code="ILS" + s.code.str.replace("+", "-", regex=False), name=s["name"] + " (CBS)",
                 pop=s["pop"] if a.settlements == "counted" else 0.0)
    out = pd.concat([gov[["code", "name", "pop", "geometry"]], s[["code", "name", "pop", "geometry"]]], ignore_index=True)
    out = gpd.GeoDataFrame(out, crs=4326)
    print(f"governorates: West Bank {wb:,.0f}, Gaza Strip {sum(v for k, v in pop.items() if k.startswith('PS02')):,.0f} "
          f"({a.gaza}); settlement units {len(s)} ({int(small.sum())} placeholder discs), {s['pop'].sum():,.0f} people "
          f"({a.settlements}; CBS counts {b[b.nonarab > 0.5 * b['pop']]['pop'].sum():,.0f}); total {out['pop'].sum():,.0f}")
    os.makedirs(DIR, exist_ok=True)
    tmp = OUT + ".part.gpkg"
    out.to_file(tmp, layer="units", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
