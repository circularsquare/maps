"""Mongolia — boundaries for the 22 aimags and the 339 soums.

Writes data/geo/mn/mn_aimags.gpkg, data/geo/mn/mn_soums.gpkg and data/geo/mn/mn_lookup.csv.

COD-AB Mongolia (`cod-ab-mng`, OCHA FISS, refreshed 2026-01-26, CC BY-IGO), one shapefile
bundle, 5.8 MB, one GET from HDX, no wall.

**THIS IS THE RARE COUNTRY WHERE THERE IS NO NAME JOIN AT ALL, AND THE MODULE'S JOB IS TO
PROVE IT.** `[[reference_name_join_wrong_neighbour]]` is the single biggest silent-failure
source in this project, and it needs two things to bite: a name key, and two places that could
plausibly answer to the same name. Mongolia has neither, because **COD's pcodes ARE the
Mongolian official administrative codes the census itself keys on**:

    adm1_pcode = "MN" + the two-digit aimag code      MN11 Ulaanbaatar, MN45 Darkhan-Uul
    adm2_pcode = "MN" + aimag code + soum code        MN4804 Adaacag, in MN48 Dundgovi

The census's own AIMAG variable (V3 of the PHC 2020 DDI, `catalog/ddi/175`) is that same
two-digit code, and its value labels are the same Cyrillic names COD carries in `adm1_name1`.
Seventeen of the twenty-two codes are printed in the DDI and **all seventeen match COD on both
the code and the Cyrillic name, with zero mismatches**; the module asserts that below rather
than repeating the claim. So the join is numeric, and a wrong pairing is not available.

**THE SOUM TIER IS BUILT WHETHER OR NOT IT IS DRAWN.** 339 soums (including Ulaanbaatar's nine
düüregs) at 9,700 people each would be a far better map than 22 aimags at 150,000, and which
one gets used is decided by what NSO publishes, not by this file. Both layers are written, both
are checked, and `sources/mn.md` records what the tabulation actually reaches. The structural
check here is the one `kz_geo.py` uses: every soum's pcode must begin with its own parent
aimag's pcode, so the hierarchy cannot be silently wrong.

Usage:
    python sources/mn_geo.py --fetch    one 5.8 MB zip from HDX
    python sources/mn_geo.py            rebuild from data/raw/mn/
"""

import os
import re
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mn")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mn")
OUT_AIMAG = os.path.join(OUT_DIR, "mn_aimags.gpkg")
OUT_SOUM = os.path.join(OUT_DIR, "mn_soums.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mn_lookup.csv")

ZIP_URL = ("https://data.humdata.org/dataset/a9b0a8a6-cb14-448e-b35c-aa5eb51b0557/"
           "resource/2ec00922-5b9b-47fc-aa36-0a8b59a877df/download/"
           "mng_admin_boundaries.shp.zip")
ZIP_NAME = "mng_admin_boundaries.shp.zip"

EXPECTED_AIMAGS = 22
EXPECTED_SOUMS = 339

# The census's own aimag codes and Cyrillic labels, read out of V3 of the PHC 2020 DDI at
# http://web.nso.mn/nada/index.php/catalog/ddi/175 . The DDI prints only the seventeen that
# appear in its public-use sample's value labels; the five western aimags (81 Zavkhan,
# 82 Govi-Altai, 83 Bayan-Olgii, 84 Khovd, 85 Uvs) are absent from that list and are NOT
# asserted here. Seventeen exact matches is already enough to establish that the two files
# share a coding scheme rather than merely a length.
CENSUS_AIMAG = {
    11: "Улаанбаатар", 21: "Дорнод", 22: "Сүхбаатар", 23: "Хэнтий",
    41: "Төв", 42: "Говьсүмбэр", 43: "Сэлэнгэ", 44: "Дорноговь",
    45: "Дархан-Уул", 46: "Өмнөговь", 48: "Дундговь", 61: "Орхон",
    62: "Өвөрхангай", 63: "Булган", 64: "Баянхонгор", 65: "Архангай",
    67: "Хөвсгөл",
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and zipfile.is_zipfile(dest):
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: HTTP 200 is not a download.
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def _read(layer, expected):
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    names = [i.filename for i in zipfile.ZipFile(src).infolist()
             if re.search(rf"mng_{layer}\.shp$", i.filename, re.I)]
    if len(names) != 1:
        raise SystemExit(f"expected one {layer} shapefile, found {names}")
    g = gpd.read_file(f"zip://{src}!{names[0]}")
    if len(g) != expected:
        raise SystemExit(f"{names[0]}: {len(g)} features, expected {expected}. "
                         "COD has re-issued Mongolia; check whether a soum was split "
                         "before changing the constant.")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {layer} {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    a1 = _read("admin1", EXPECTED_AIMAGS)
    a2 = _read("admin2", EXPECTED_SOUMS)
    print(f"COD-AB: {len(a1)} aimags, {len(a2)} soums")

    # ---- 1. the pcodes are well-formed and are the official codes
    bad = [p for p in a1["adm1_pcode"] if not re.fullmatch(r"MN\d\d", str(p))]
    if bad:
        raise SystemExit(f"adm1_pcode not MN##: {bad[:5]}")
    bad = [p for p in a2["adm2_pcode"] if not re.fullmatch(r"MN\d\d\d\d", str(p))]
    if bad:
        raise SystemExit(f"adm2_pcode not MN####: {bad[:5]}")
    if not a1["adm1_pcode"].is_unique or not a2["adm2_pcode"].is_unique:
        raise SystemExit("pcodes are not unique")

    # ---- 2. the hierarchy: a soum's pcode begins with its own parent's pcode
    wrong = [(r["adm2_pcode"], r["adm1_pcode"])
             for _, r in a2.iterrows()
             if str(r["adm2_pcode"])[:4] != str(r["adm1_pcode"])]
    if wrong:
        raise SystemExit(f"{len(wrong)} soums whose pcode does not sit under their own "
                         f"aimag: {wrong[:5]}")
    orphan = set(a2["adm1_pcode"]) - set(a1["adm1_pcode"])
    if orphan:
        raise SystemExit(f"soums under an aimag that is not in admin1: {sorted(orphan)}")
    childless = set(a1["adm1_pcode"]) - set(a2["adm1_pcode"])
    if childless:
        raise SystemExit(f"aimags with no soums: {sorted(childless)}")
    print(f"  every soum's pcode sits under its own aimag; all {EXPECTED_AIMAGS} aimags "
          f"have children")

    # ---- 3. THE JOIN ASSERTION: COD's codes are the census's codes
    by_code = {int(str(p)[2:]): n for p, n in zip(a1["adm1_pcode"], a1["adm1_name1"])}
    missing = sorted(k for k in CENSUS_AIMAG if k not in by_code)
    if missing:
        raise SystemExit(f"census aimag codes absent from COD: {missing}")
    mism = [(k, CENSUS_AIMAG[k], by_code[k]) for k in sorted(CENSUS_AIMAG)
            if by_code[k] != CENSUS_AIMAG[k]]
    if mism:
        for k, want, got in mism:
            print(f"    MN{k:02d}  census {want!r}  COD {got!r}")
        raise SystemExit(f"{len(mism)} of {len(CENSUS_AIMAG)} aimag codes disagree with "
                         "the census DDI on the Cyrillic name -- the two files do NOT "
                         "share a coding scheme and every join below is suspect")
    print(f"  all {len(CENSUS_AIMAG)} aimag codes the PHC 2020 DDI prints match COD on the "
          f"code AND the Cyrillic name")

    # ---- 4. write
    os.makedirs(OUT_DIR, exist_ok=True)
    aim = a1[["adm1_pcode", "adm1_name", "adm1_name1", "geometry"]].copy()
    aim.columns = ["unit", "name_en", "name_mn", "geometry"]
    aim = aim.sort_values("unit").reset_index(drop=True)
    aim.to_file(OUT_AIMAG, driver="GPKG", layer="aimags")

    soum = a2[["adm2_pcode", "adm2_name", "adm2_name1",
               "adm1_pcode", "adm1_name", "geometry"]].copy()
    soum.columns = ["unit", "name_en", "name_mn", "aimag", "aimag_en", "geometry"]
    soum = soum.sort_values("unit").reset_index(drop=True)
    soum.to_file(OUT_SOUM, driver="GPKG", layer="soums")

    lut = pd.concat([
        pd.DataFrame({"level": "aimag", "unit": aim["unit"], "name_en": aim["name_en"],
                      "name_mn": aim["name_mn"], "parent": ""}),
        pd.DataFrame({"level": "soum", "unit": soum["unit"], "name_en": soum["name_en"],
                      "name_mn": soum["name_mn"], "parent": soum["aimag"]}),
    ], ignore_index=True)
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")

    per = soum.groupby("aimag").size().sort_values(ascending=False)
    print(f"\nwrote {OUT_AIMAG} ({len(aim)} aimags)")
    print(f"wrote {OUT_SOUM} ({len(soum)} soums)")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")
    print(f"\n  soums per aimag: {per.max()} (MN{per.idxmax()[2:]}) down to "
          f"{per.min()} (MN{per.idxmin()[2:]})")


if __name__ == "__main__":
    main()
