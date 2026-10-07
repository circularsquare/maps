"""Qatar: a placement layer on the 2020 census's eight municipalities, re-keyed from religiondots'
hexes. -> data/geo/qa/qa_hexes.gpkg

    python sources/qa_geo.py

religiondots built Qatar on the ten municipalities of 2004 (`../religiondots/sources/qa_geo.py`,
read-only): Kontur 2023's 400 m hexes cut to COD-AB Qatar v02's zones, each piece carrying its
zone number, and each zone's pieces scaled to the zone's 2004 population. The zones kept their
numbers, so this re-keys the same pieces:

  * unit = the 2020 municipality the zone belongs to in COD-AB (its pcode QATmmmzzz; mmm decoded
    here and asserted: Table 2's zones summed by it equal Table 1's municipalities);
  * pop = each zone's 2020 population (census Table 2) spread over the zone's pieces in
    proportion to religiondots' weights, which inside a zone are Kontur's; a zone religiondots
    left at zero (it had nobody in 2004) is spread by piece area.

Every 2020 zone with people must be a COD-AB zone; asserted.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RD_QA = ROOT.parent / "religiondots" / "data" / "geo" / "qa"
XLSX = ROOT / "data" / "raw" / "qa" / "Census_Final_Results.xlsx"
OUT = ROOT / "data" / "geo" / "qa" / "qa_hexes.gpkg"

# COD-AB Qatar v02's municipality code (pcode digits 4-6) -> Table 1's municipality
MUNI = {"001": "Al Daayen", "002": "Al Khor and Al Thakhira", "003": "Al Rayyan",
        "004": "Al Shamal", "005": "Al Wakra", "006": "Doha", "007": "Umm Slal",
        "008": "Al Sheehaniya"}


def table1():
    t = pd.read_excel(XLSX, sheet_name="1", header=None)
    out = {}
    for _i, r in t.iterrows():
        lab = str(r[0]).strip()
        if lab in MUNI.values() or lab == "Total":
            out[lab] = (int(r[3]), int(r[2]), int(r[1]))      # total, men, women
    if len(out) != 9 or sum(v[0] for k, v in out.items() if k != "Total") != out["Total"][0]:
        raise SystemExit(f"Table 1 not as expected: {out}")
    return out


def table2():
    t = pd.read_excel(XLSX, sheet_name="2", header=None)
    z = {}
    for _i, r in t.iterrows():
        try:
            n = int(str(r[0]).strip())
        except ValueError:
            continue
        z[n] = (int(r[4]), int(r[3]), int(r[2]))
    return z


def main():
    t1 = table1()
    t2 = table2()
    if sum(v[0] for v in t2.values()) != t1["Total"][0]:
        raise SystemExit(f"Table 2's {len(t2)} zones sum to {sum(v[0] for v in t2.values()):,}, "
                         f"Table 1 {t1['Total'][0]:,}")
    zones = gpd.read_file(RD_QA / "qa_zones.gpkg")
    zones["muni"] = zones["adm2_pcode"].str[3:6].map(MUNI)
    if zones["muni"].isna().any():
        raise SystemExit(f"unknown municipality codes: {sorted(zones.loc[zones['muni'].isna(), 'adm2_pcode'])}")
    muni_of = dict(zip(zones["zone"].astype(int), zones["muni"]))
    missing = sorted(z for z, v in t2.items() if v[0] > 0 and z not in muni_of)
    if missing:
        raise SystemExit(f"2020 zones with people and no COD-AB polygon: {missing}")
    by_muni = {}
    for z, v in t2.items():
        by_muni[muni_of[z]] = by_muni.get(muni_of[z], 0) + v[0]
    bad = {m: (by_muni.get(m), t1[m][0]) for m in MUNI.values() if by_muni.get(m) != t1[m][0]}
    if bad:
        raise SystemExit(f"Table 2 zones by COD-AB municipality do not equal Table 1: {bad}")
    print(f"  {len(t2)} zones of 2020 sum by COD-AB's municipality to Table 1's eight, exactly")

    h = gpd.read_file(RD_QA / "qa_hexes.gpkg")
    h["zone"] = h["zone"].astype(int)
    h["unit"] = h["zone"].map(muni_of)
    if h["unit"].isna().any():
        raise SystemExit("religiondots' hex pieces carry zones COD-AB lacks")
    area = h.to_crs(6933).area
    zsum = h.groupby("zone")["pop"].transform("sum")
    asum = area.groupby(h["zone"]).transform("sum")
    share = (h["pop"] / zsum).where(zsum > 0, area / asum)
    h["pop"] = share * h["zone"].map(lambda z: t2.get(z, (0,))[0])
    # zones religiondots gave no piece (nobody there in 2004: 46, 49, 50, 58, 98) are placed on
    # the whole zone polygon, evenly
    no_piece = sorted(z for z, v in t2.items() if v[0] > 0 and z not in set(h["zone"]))
    if no_piece:
        zp = zones[zones["zone"].astype(int).isin(no_piece)].copy()
        zp["zone"] = zp["zone"].astype(int)
        zp["unit"] = zp["muni"]
        zp["pop"] = zp["zone"].map(lambda z: t2[z][0])
        h = pd.concat([h, zp[["unit", "zone", "pop", "geometry"]].to_crs(h.crs)], ignore_index=True)
        print(f"  zones with no religiondots piece, placed on the zone polygon: "
              + ", ".join(f"{z} ({t2[z][0]:,})" for z in no_piece))
        zsum = zsum.reindex(h.index).fillna(1.0)
    by_area = sorted(int(z) for z in h.loc[zsum <= 0, "zone"].unique() if t2.get(z, (0,))[0] > 0)
    got = h.groupby("unit")["pop"].sum()
    for m in MUNI.values():
        if abs(got[m] - t1[m][0]) > 1:
            raise SystemExit(f"{m}: layer {got[m]:,.0f} against Table 1 {t1[m][0]:,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    h[["unit", "zone", "pop", "geometry"]].to_file(OUT, driver="GPKG")
    print(f"wrote {OUT}: {len(h):,} pieces, 8 municipalities; zones spread by area "
          f"(no 2004 weight): {by_area} ({sum(t2[z][0] for z in by_area):,} people)")


if __name__ == "__main__":
    main()
