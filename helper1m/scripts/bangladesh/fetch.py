"""Bangladesh populations for helper1m -> data/bangladesh/population.csv

Units are the OCHA COD-AB v03 (2023) boundaries in maps/data/asia1m/bangladesh/:
8 divisions, 64 zilas, and 507 level-3 units = 495 upazilas + 12 city
corporations. That level-3 split is exactly how the 2022 census reports, so 2022
joins one-to-one.

2022  Population and Housing Census 2022, National Report (Volume I), BBS, Nov
      2023. Table P35 (upazila, outside city corporations) and Table P33 (city
      corporation, by thana). Enumerated population, male + female. See report.py.
2011  Population and Housing Census 2011, union/ward level (5,161 units), as
      transcribed by the US Census Bureau (bangladesh_uscb_202107.xlsx, Age-Sex
      sheet, with matching 2011 polygons in Bangladesh.gdb). The 2011 units are
      not the 2022 ones (city corporations were created or enlarged, upazilas
      split), so each 2011 union is moved WHOLE onto one v03 unit (new upazilas
      and city corporations are formed from whole unions and wards):
        - its home is the v03 upazila carrying the same BBS geocode (2011 NSO
          code 100409 -> v03 BD10040009), which holds for 482 of the 544 2011
          upazilas/thanas;
        - it leaves home only for a unit that did not exist in 2011 (13 new
          upazilas, 12 city corporations), when at least MOVE of its Kontur 2023
          population falls inside that unit;
        - a union with no home (the 61 metropolitan thanas of 2011, and
          Dakshin Sunamganj, renamed Shantiganj with a new code) goes to the v03
          unit holding most of its Kontur population.
      Geometry alone is too noisy for the rest: the 2011 USCB polygons and the
      v03 lines differ by a few hundred metres, which moves 10-20% of the hex
      centres of many border unions across the line.

Levels 2 and 1 are the sums of level 3, for both years. Level 4 (unions,
paurashavas, city corporation wards/thanas) is built by level4.py from the
level-3 figures computed here, and adds up to them exactly.

Usage:  C:\\Python39\\python.exe helper1m/scripts/bangladesh/fetch.py [--download]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import re
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

import report

REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "helper1m"
DATA = HELPER / "data/bangladesh"
RAW = DATA / "raw"

V03 = REPO / "data/asia1m/bangladesh/bgd_admin{n}.shp"
USCB_XLSX = REPO / "religiondots/data/raw/bd/bangladesh_uscb_202107.xlsx"
USCB_GDB = REPO / "religiondots/data/raw/bd/Bangladesh.gdb"
KONTUR = RAW / "kontur_population_BD_20231101.gpkg"
KONTUR_GZ = REPO / "religiondots/data/geo/kontur/kontur_population_BD_20231101.gpkg.gz"

# The full 517-page report. bbs.portal.gov.bd is retired and its successor
# bbs.gov.bd fails TLS from here, so the Wayback copy is the reliable route.
PDF_URL = ("https://web.archive.org/web/20231125020331if_/https://bbs.portal.gov.bd/"
           "sites/default/files/files/bbs.portal.gov.bd/page/"
           "b343a8b4_956b_45ca_872f_4cf9b2f1a6e0/"
           "2023-11-20-05-20-e6676a7993679bfd72a663e39ef0cca7.pdf")

# Population and Housing Census 2022: Union Statistics (BBS, May 2025), linked
# from the census page of the new bbs.gov.bd (static-pages/6922e073933eb65569e27220).
UNION_PDF = RAW / "phc2022_union_statistics.pdf"
UNION_URL = ("https://objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/n/axvjbnqprylg/b/"
             "V2Ministry/o/office-bbs/2024/12/f376ca7e7ee5405f8311de650415b9d8.pdf")

CHECK_FILES = {
    # UNRCO Bangladesh's upload of BBS's PHC 2022 district dataset (64 zilas).
    "unrco_phc2022_admin02.xlsx":
        "https://data.humdata.org/dataset/a6fedebe-72fe-4fc2-8657-1580acfa32c6/resource/"
        "72eaaa6c-6a30-4efd-bad9-02133b316ea8/download/"
        "bangladesh_bbs_population-and-housing-census-dataset_2022_admin-02.xlsx",
    # COD-PS: USCB's 2022 projection from the 2011 census, on the 544 2011 units.
    "bgd_admpop_2022.xlsx":
        "https://data.humdata.org/dataset/fdf0606c-8a3b-421a-b3e8-903301e5b2ff/resource/"
        "d3ee1ccb-9efb-412f-9323-cd4dc9606f7d/download/bgd_admpop_2022.xlsx",
    # Thakurgaon's PHC 2022 District Report: an independent print of its unions.
    "zila2022/thakurgaon.pdf":
        "https://web.archive.org/web/20250512071726id_/http://203.112.218.101/storage/files/1/"
        "Publications/PHCensus/Rangpur/District%20Report%20Thakurgaon%20Full.pdf",
}

# A 2011 union leaves its home upazila for a unit that is new since 2011 when at
# least this share of its Kontur population lies inside that new unit.
MOVE = 0.5

# 2011 units whose geocode did not survive but whose successor is known.
HOME_ALIAS = {"609027": "BD60900087"}  # Dakshin Sunamganj -> Shantiganj (renamed)

OUT = DATA / "population.csv"
LINEAGE = DATA / "unions_2011_to_v03.csv"


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    if not report.PDF.exists():
        print("downloading the National Report Vol I (11 MB) from the Wayback Machine")
        subprocess.run(["curl", "-sL", "-m", "600", PDF_URL, "-o", str(report.PDF)], check=True)
    if not UNION_PDF.exists():
        print("downloading the Union Statistics report (15 MB) from BBS's object storage")
        subprocess.run(["curl", "-sL", "-m", "600", UNION_URL, "-o", str(UNION_PDF)], check=True)
    # Only check.py reads these two.
    for name, url in CHECK_FILES.items():
        if not (RAW / name).exists():
            print(f"downloading {name}")
            (RAW / name).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-sL", "-m", "300", url, "-o", str(RAW / name)], check=True)
    if not KONTUR.exists():
        import gzip, shutil
        print("decompressing Kontur BD")
        tmp = KONTUR.with_suffix(".part")
        with gzip.open(KONTUR_GZ, "rb") as f, open(tmp, "wb") as g:
            shutil.copyfileobj(f, g, 1 << 20)
        os.replace(tmp, KONTUR)


def load_v03(n):
    return gpd.read_file(str(V03).format(n=n))


def pop_2022(v3):
    """{adm3_pcode: pop} straight from the report tables."""
    recs = pd.DataFrame(report.parse())
    ups = recs[(recs.table == "P35") & (recs.name != "") & (recs.name != "TOTAL")]
    ccs = recs[(recs.table == "P33") & (recs.name == "")]
    up_pop = {(norm(r.district), norm(r.name)): r.pop for r in ups.itertuples()}
    cc_pop = {norm(r.cc): r.pop for r in ccs.itertuples()}
    out, missing = {}, []
    for r in v3.itertuples():
        if r.adm3_name.endswith("City Corporation"):
            p = cc_pop.pop(norm(r.adm3_name), None)
        else:
            p = up_pop.pop((norm(r.adm2_name), norm(r.adm3_name)), None)
        if p is None:
            missing.append((r.adm2_name, r.adm3_name))
        else:
            out[r.adm3_pcode] = int(p)
    assert not missing, f"v03 units with no 2022 row: {missing}"
    assert not up_pop and not cc_pop, f"2022 rows with no v03 unit: {up_pop} {cc_pop}"
    # The parsed tables must add up to their own printed totals.
    total = recs[(recs.table == "P35") & (recs.name == "TOTAL")]["pop"].iloc[0]
    assert ups["pop"].sum() == total
    assert sum(out.values()) == total + ccs["pop"].sum()
    return out


def unions_2011():
    x = pd.read_excel(USCB_XLSX, sheet_name="Age-Sex", skiprows=[1])
    national = int(x[x.ADM_LEVEL.astype(str) == "0"].BTOTL.iloc[0])
    x = x[x.ADM_LEVEL.astype(str) == "4"][["GEO_MATCH", "BTOTL"]]
    g = gpd.read_file(USCB_GDB, layer="BD_GEOG_ADM4_2011_uscb_202107")
    g = g[["GEO_MATCH", "ADM2_NAME", "ADM3_NAME", "ADM4_NAME", "geometry"]].merge(
        x, on="GEO_MATCH", how="outer", indicator=True)
    assert (g._merge == "both").all(), g[g._merge != "both"]
    assert len(g) == 5161
    g = g.drop(columns="_merge")
    g["BTOTL"] = g.BTOTL.astype(int)
    # The Age-Sex sheet's unions add up to its own national row, which is one
    # person above BBS's published 144,043,696 (the Religion sheet has it exact).
    assert g.BTOTL.sum() == national, (g.BTOTL.sum(), national)

    # The union's 2011 upazila/thana, for its BBS geocode.
    g3 = gpd.read_file(USCB_GDB, layer="BD_GEOG_ADM3_2011_uscb_202107", ignore_geometry=True)
    g["up_match"] = g.GEO_MATCH.str.rsplit("_", n=1).str[0]
    g = g.merge(g3[["GEO_MATCH", "NSO_CODE", "NSO_NAME"]].rename(
        columns={"GEO_MATCH": "up_match", "NSO_CODE": "up_nso", "NSO_NAME": "up_name"}),
        on="up_match", how="left")
    assert g.up_nso.notna().all()
    return gpd.GeoDataFrame(g, geometry="geometry", crs="EPSG:4326")


def kontur_shares(unions, v3):
    """Long table GEO_MATCH, adm3_pcode, share: where each 2011 union's Kontur
    2023 population sits among the v03 units (by hex centre). Unions with no hex
    centre inside them (city wards smaller than one 0.74 km2 hex) get share 1 in
    the unit holding a point inside the union."""
    hexes = gpd.read_file(KONTUR)[["population", "geometry"]]  # EPSG:3857
    pts = hexes.copy()
    pts["geometry"] = hexes.geometry.centroid
    pts = pts.to_crs("EPSG:4326")
    pts = gpd.sjoin(pts, unions[["GEO_MATCH", "geometry"]], how="inner", predicate="within")
    pts = pts.drop(columns="index_right")
    pts = gpd.sjoin(pts, v3[["adm3_pcode", "geometry"]], how="inner", predicate="within")
    w = pts.groupby(["GEO_MATCH", "adm3_pcode"]).population.sum().reset_index()
    w = w[w.population > 0]
    w["share"] = w.population / w.groupby("GEO_MATCH").population.transform("sum")
    w = w[["GEO_MATCH", "adm3_pcode", "share"]]

    left = unions[~unions.GEO_MATCH.isin(w.GEO_MATCH)].copy()
    left["geometry"] = left.geometry.representative_point()
    j = gpd.sjoin(left[["GEO_MATCH", "geometry"]], v3[["adm3_pcode", "geometry"]],
                  how="left", predicate="within")
    if j.adm3_pcode.isna().any():
        # A point just off the v03 coastline: take the nearest unit.
        far = j[j.adm3_pcode.isna()].drop(columns=["adm3_pcode", "index_right"])
        near = gpd.sjoin_nearest(far.to_crs(3106), v3[["adm3_pcode", "geometry"]].to_crs(3106))
        j = pd.concat([j[j.adm3_pcode.notna()], near.to_crs(4326)])
    j = j[["GEO_MATCH", "adm3_pcode"]].assign(share=1.0)
    return pd.concat([w, j], ignore_index=True), set(left.GEO_MATCH)


def recut_2011(unions, v3):
    """Move each 2011 union whole onto one v03 level-3 unit (rules in the module
    docstring). Returns ({pcode: pop}, lineage table)."""
    v3codes = set(v3.adm3_pcode)
    unions = unions.copy()
    home = "BD" + unions.up_nso.str[:4] + unions.up_nso.str[4:].str.zfill(4)
    home = home.where(home.isin(v3codes))
    for nso, code in HOME_ALIAS.items():
        home[unions.up_nso == nso] = code
    unions["home"] = home
    new_units = v3codes - set(home.dropna())

    shares, by_point = kontur_shares(unions, v3)
    shares["new"] = shares.adm3_pcode.isin(new_units)

    rows = []
    sh ={gm: g.sort_values("share", ascending=False) for gm, g in shares.groupby("GEO_MATCH")}
    for u in unions.itertuples():
        s = sh[u.GEO_MATCH]
        top = s.iloc[0]
        new = s[s.new]
        in_home = s[s.adm3_pcode == u.home].share.sum() if isinstance(u.home, str) else 0.0
        if isinstance(u.home, str):
            if len(new) and new.share.iloc[0] >= MOVE:
                dest, how = new.adm3_pcode.iloc[0], "moved to new unit"
            else:
                dest, how = u.home, "home"
        else:
            dest, how = top.adm3_pcode, "no home: largest share"
        rows.append(dict(GEO_MATCH=u.GEO_MATCH, adm2_2011=u.ADM2_NAME, unit_2011=u.up_name,
                         union_2011=u.ADM4_NAME, pop_2011=u.BTOTL, home=u.home,
                         adm3_pcode=dest, how=how,
                         share_in_dest=round(float(s[s.adm3_pcode == dest].share.sum()), 3),
                         share_in_home=round(float(in_home), 3),
                         by_point=u.GEO_MATCH in by_point))
    lin = pd.DataFrame(rows)
    assert lin.pop_2011.sum() == unions.BTOTL.sum()
    by_unit = lin.groupby("adm3_pcode").pop_2011.sum()
    return by_unit, lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch missing raw inputs first")
    args = ap.parse_args()
    if args.download or not report.PDF.exists() or not KONTUR.exists() or not UNION_PDF.exists() \
            or not all((RAW / n).exists() for n in CHECK_FILES):
        download()

    v1, v2, v3 = load_v03(1), load_v03(2), load_v03(3)
    print(f"v03 units: {len(v1)} divisions, {len(v2)} zilas, {len(v3)} upazilas/city corporations")

    p22 = pop_2022(v3)
    print(f"2022: {len(p22)} units, {sum(p22.values()):,}")

    unions = unions_2011()
    by_unit, lin = recut_2011(unions, v3)
    lin.to_csv(LINEAGE, index=False)
    missing = set(v3.adm3_pcode) - set(by_unit.index)
    assert not missing, f"v03 units with no 2011 union: {missing}"
    p11 = by_unit.astype(int)
    print(f"2011: {len(p11)} units, {p11.sum():,}; unions by rule: "
          + ", ".join(f"{k} {v}" for k, v in lin.how.value_counts().items())
          + f"; {lin.by_point.sum()} placed by a point (no hex centre inside)")
    # Unions kept home although most of their Kontur population lies in another
    # pre-existing upazila: digitising noise or a real transfer, not told apart.
    odd = lin[(lin.how == "home") & (lin.share_in_home < 0.5)]
    print(f"  kept home with < 50% of Kontur inside home: {len(odd)} unions, "
          f"{odd.pop_2011.sum():,} people")

    rows = []
    for year, pops in ((2011, p11.to_dict()), (2022, p22)):
        s3 = pd.Series(pops)
        for code, pop in s3.items():
            rows.append((code, 3, year, int(pop)))
        par = v3.set_index("adm3_pcode")
        s2 = s3.groupby(par.adm2_pcode).sum()
        s1 = s3.groupby(par.adm1_pcode).sum()
        assert set(s2.index) == set(v2.adm2_pcode) and set(s1.index) == set(v1.adm1_pcode)
        rows += [(c, 2, year, int(p)) for c, p in s2.items()]
        rows += [(c, 1, year, int(p)) for c, p in s1.items()]
    # Level 4: unions, paurashavas, city corporation wards/thanas (level4.py).
    import level4
    rows += level4.build(v1, v2, v3, p22, p11.to_dict())

    out = pd.DataFrame(rows, columns=["code", "level", "year", "pop"]).sort_values(
        ["level", "code", "year"])
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
