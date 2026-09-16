"""Tunisia: boundaries and populations for the 24 governorates.

Writes data/geo/tn/tn_governorates.gpkg and data/geo/tn/tn_lookup.csv.

Two sources:

  * **boundaries**: COD-AB `cod-ab-tun` v01 ADM2 (OCHA, source Institut National de la
    Statistique, valid from 2022-11-15, reviewed 2024-12-19), 24 features. ADM1 in that file is the
    six old economic regions (North East, North West, Centre East, Centre West, South East, South
    West), not the governorates, and ADM3 is 264 delegations.
  * **populations**: INS, *Recensement Général de la Population et de l'Habitat 2024, Bilan
    Démographique* (May 2025), p.15, *Evolution de la population par gouvernorat selon les
    recensements (1994-2024)*: the census count of each governorate on 6 November 2024
    (11,972,169), with 1994, 2004 and 2014 beside it and the five districts as subtotals.

## THE KEY IS THE INS GOVERNORATE CODE

COD's `adm2_pcode` is `TN` followed by INS's two-digit governorate code (Tunis 11 to Kébili 63),
the same code INS's own 2024 workbooks carry as `Code_Gouvernorat`. So the join is an authored
table, `GOVERNORATES`: INS code -> the name this map prints -> the Bilan's spelling -> COD's
`adm2_ref_name`. It is checked four ways:

  1. the table covers COD's 24 p-codes and the Bilan's 24 governorate rows exactly;
  2. **the Bilan's own district subtotals**: each district row equals the sum of the governorates
     printed above it, in all four census years, and the five districts sum to its `Total`, which
     is also p.7's national figure for 2024 and 2014;
  3. **COD's ADM1 parent** of each governorate is the economic region the first digit of the INS
     code names (1 North East ... 6 South West);
  4. **Tunis's area**: p.21 prints Tunis's 2024 density, 3,734 per km2, so INS's area is 288 km2;
     COD's polygon must be within 0.8-1.2x of it. The rank witness that pins the whole join,
     Kontur people per governorate against the census, is in `sources/tn_grid.py`.

Usage:
    python sources/tn_geo.py --fetch    COD-AB geojson zip (54.8 MB) and the INS report (5.0 MB)
    python sources/tn_geo.py            rebuild from data/raw/tn/
"""

import io
import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tn")
OUT_DIR = os.path.join(ROOT, "data", "geo", "tn")
OUT = os.path.join(OUT_DIR, "tn_governorates.gpkg")
LOOKUP = os.path.join(OUT_DIR, "tn_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

COD_URL = ("https://data.humdata.org/dataset/e47eda48-8f83-4858-b739-dbffb8a50c47/resource/"
           "ebe19593-9438-4d14-8291-e65f811e2671/download/tun_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "tun_admin_boundaries.geojson.zip")
INS_URL = ("https://www.ins.tn/sites/default/files-ftp3/files/2025-05/"
           "Bilan%20D%C3%A9mographique_0.pdf")
INS_PDF = os.path.join(RAW, "ins_rgph2024_bilan_demographique.pdf")

TOTAL_2024 = 11_972_169       # p.15 Total row, and p.7
TOTAL_2014 = 10_982_754       # p.15 Total row, and p.7
# p.15's Total row minus the sum of its five district rows, for 1994, 2004, 2014 and 2024,
# measured 2026-09-15. 2014's Total is the official census figure and the governorates under it
# add to 10,982,476; every district subtotal equals its own governorates, so the 278 is INS's
# and not a misread. 2024, the year drawn, closes exactly.
TOTAL_MINUS_DISTRICTS = (0, 0, 278, 0)
TUNIS_DENSITY_2024 = 3734     # p.21, habitants par km2
TUNIS_AREA_BAND = (0.8, 1.2)

# INS code -> (the name this map prints, the Bilan's p.15 spelling, COD's adm2_ref_name)
GOVERNORATES = {
    11: ("Tunis", "Tunis", "Tunis"),
    12: ("Ariana", "Ariana", "Ariana"),
    13: ("Ben Arous", "Ben Arous", "Ben Arous"),
    14: ("Manouba", "Manouba", "Manubah"),
    15: ("Nabeul", "Nabeul", "Nabeul"),
    16: ("Zaghouan", "Zaghouan", "Zaghouan"),
    17: ("Bizerte", "Bizerte", "Bizerte"),
    21: ("Béja", "Béja", "Beja"),
    22: ("Jendouba", "Jendouba", "Jendouba"),
    23: ("Le Kef", "Le Kef", "Le Kef"),
    24: ("Siliana", "Siliana", "Siliana"),
    31: ("Sousse", "Sousse", "Sousse"),
    32: ("Monastir", "Monastir", "Monastir"),
    33: ("Mahdia", "Mahdia", "Mahdia"),
    34: ("Sfax", "Sfax", "Sfax"),
    41: ("Kairouan", "Kairouan", "Kairouan"),
    42: ("Kasserine", "Kasserine", "Kasserine"),
    43: ("Sidi Bouzid", "Sidi Bouzid", "Sidi Bou Zid"),
    51: ("Gabès", "Gabes", "Gabes"),
    52: ("Médenine", "Médenine", "Medenine"),
    53: ("Tataouine", "Tataouine", "Tataouine"),
    61: ("Gafsa", "Gafsa", "Gafsa"),
    62: ("Tozeur", "Tozeur", "Tozeur"),
    63: ("Kébili", "Kébili", "Kebili"),
}
ECONOMIC_REGION = {1: "North East", 2: "North West", 3: "Centre East", 4: "Centre West",
                   5: "South East", 6: "South West"}

EQUAL_AREA = "EPSG:6933"


def geo_id(code):
    return f"TN{code}"


def fold(s):
    s = str(s).lower()
    for a, b in (("é", "e"), ("è", "e"), ("ï", "i"), ("â", "a"), ("ô", "o")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z]", "", s)


def _get(url, dst, magic, minsize):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r:
        data = r.read()
    if not data.startswith(magic):
        raise SystemExit(f"{url} did not return the expected file; starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(COD_URL, COD_ZIP, b"PK", 10_000_000)
    _get(INS_URL, INS_PDF, b"%PDF", 1_000_000)


def bilan_table():
    """p.15 as {name: (1994, 2004, 2014, 2024)} for governorates, districts and Total, in order."""
    import fitz

    with open(INS_PDF, "rb") as fh:
        if b"%%EOF" not in fh.read()[-2048:]:
            raise SystemExit(f"{INS_PDF} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(INS_PDF)
    hits = [i for i, p in enumerate(doc)
            if "Evolution de la population par gouvernorat" in p.get_text()]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} pages carry p.15's title, expected 1")
    lines = [ln.strip() for ln in doc[hits[0]].get_text().splitlines() if ln.strip()]
    start = next(i for i, ln in enumerate(lines) if ln == "2024") + 1
    rows, i = [], start
    while i < len(lines):
        name = lines[i]
        nums = lines[i + 1:i + 5]
        if not all(re.fullmatch(r"\d+", x) for x in nums):
            raise SystemExit(f"p.15 row {name!r} is not followed by four counts: {nums}")
        rows.append((name, tuple(int(x) for x in nums)))
        i += 5
        if name == "Total":
            break
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, INS_PDF):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")

    with zipfile.ZipFile(COD_ZIP) as zf:
        g = gpd.read_file(io.BytesIO(zf.read("tun_admin2.geojson")))
    if len(g) != 24:
        raise SystemExit(f"{len(g)} COD-AB ADM2 features, expected 24")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read COD-AB ADM2: {len(g)} governorates, {g.crs}")

    rows = bilan_table()
    govs = [(n, v) for n, v in rows if not n.startswith("District") and n != "Total"]
    if len(govs) != 24:
        raise SystemExit(f"p.15 has {len(govs)} governorate rows, expected 24")

    # ---- witness 2: district subtotals and the Total, every census year ----
    acc = [0, 0, 0, 0]
    districts = 0
    for name, v in rows:
        if name.startswith("District"):
            if tuple(acc) != v:
                raise SystemExit(f"p.15 {name}: governorates above sum to {acc}, printed {v}")
            districts += 1
            acc = [0, 0, 0, 0]
        elif name == "Total":
            tot = v
        else:
            acc = [a + x for a, x in zip(acc, v)]
    dsum = [sum(v[k] for n, v in rows if n.startswith("District")) for k in range(4)]
    gap = tuple(t - d for t, d in zip(tot, dsum))
    if districts != 5 or gap != TOTAL_MINUS_DISTRICTS or tot[3] != TOTAL_2024 \
            or tot[2] != TOTAL_2014:
        raise SystemExit(f"p.15's districts or Total do not close as pinned: {districts} "
                         f"districts, {dsum} against {tot} (Total minus districts {gap}, "
                         f"pinned {TOTAL_MINUS_DISTRICTS})")
    print(f"  INS p.15: 24 governorates in 5 districts; every district subtotal equals its "
          f"governorates in all four years; the Total equals the districts in 1994, 2004 and "
          f"2024 and is {gap[2]} above them in 2014 (pinned); {TOTAL_2024:,} people in 2024")

    # ---- witness 1: the authored table covers both sides exactly ----
    bilan = {fold(n): v for n, v in govs}
    if sorted(fold(b) for _m, b, _c in GOVERNORATES.values()) != sorted(bilan):
        raise SystemExit(f"GOVERNORATES and p.15 disagree: "
                         f"{sorted(set(fold(b) for _m, b, _c in GOVERNORATES.values()) ^ set(bilan))}")
    if sorted(geo_id(c) for c in GOVERNORATES) != sorted(g["adm2_pcode"]):
        raise SystemExit("GOVERNORATES' codes and COD's p-codes do not agree")
    ref = dict(zip(g["adm2_pcode"], g["adm2_ref_name"]))
    bad = [(c, cod, ref[geo_id(c)]) for c, (_m, _b, cod) in GOVERNORATES.items()
           if fold(cod) != fold(ref[geo_id(c)])]
    if bad:
        raise SystemExit(f"COD's name under a p-code is not the one GOVERNORATES expects: {bad}")
    print("  witness 1: the table covers COD's 24 p-codes and p.15's 24 rows; COD's name under "
          "each p-code is the expected one")

    # ---- witness 3: COD's ADM1 parent is the INS code's economic region ----
    par = dict(zip(g["adm2_pcode"], g["adm1_name"]))
    bad = [(c, par[geo_id(c)]) for c in GOVERNORATES if par[geo_id(c)] != ECONOMIC_REGION[c // 10]]
    if bad:
        raise SystemExit(f"COD's ADM1 parent does not match the INS code's first digit: {bad}")
    print("  witness 3: every governorate sits in the economic region its INS code's first digit "
          "names")

    lut = pd.DataFrame([dict(code=c, geo_id=geo_id(c), name=m, ins_name=b, cod_name=cod,
                             pop=bilan[fold(b)][3], pop_2014=bilan[fold(b)][2])
                        for c, (m, b, cod) in GOVERNORATES.items()])
    g = g.merge(lut, left_on="adm2_pcode", right_on="geo_id", how="inner", validate="1:1")
    g["cod_area"] = g.to_crs(EQUAL_AREA).geometry.area / 1e6

    # ---- witness 4: Tunis's area against p.21's density ----
    t = g[g["code"] == 11].iloc[0]
    ins_area = t["pop"] / TUNIS_DENSITY_2024
    ratio = t["cod_area"] / ins_area
    print(f"  witness 4: Tunis {t['cod_area']:.0f} km2 on COD against {ins_area:.0f} km2 from "
          f"p.21's density, {ratio:.2f}x")
    if not TUNIS_AREA_BAND[0] <= ratio <= TUNIS_AREA_BAND[1]:
        raise SystemExit("Tunis's polygon is outside the band around INS's area")

    g["unit"] = g["geo_id"]
    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "geo_id", "code", "pop", "geometry"]].to_file(
        OUT, layer="governorates", driver="GPKG")
    print(f"\nwrote {OUT} (24 polygons)")
    lk = g[["geo_id", "unit", "code", "name", "pop", "pop_2014", "cod_area"]].sort_values("code")
    lk.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} (24 rows, {int(lk['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
