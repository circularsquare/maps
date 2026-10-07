"""Azerbaijan: the 74 COD-AB units with the 2019 census's EXISTING (de facto) population.

Writes data/geo/az/az_units.gpkg and data/geo/az/az_lookup.csv. `sources/az.md` §4 is the record.

  * **boundaries**: COD-AB `cod-ab-aze` v01 (OCHA; valid from 2023-10-17, reviewed 2024-01-31,
    boundaries created 2020-11-24), `aze_admin1.geojson`: 74 units, every rayon plus the cities of
    republican subordination (Baku, Ganja, Sumgayit, Mingachevir, Shirvan, Naftalan, Nakhchivan,
    Khankendi). Lankaran, Shaki and Yevlakh are one polygon each with their city, as in every
    population table below.
  * **population**: *Population Census in the Republic of Azerbaijan 2019*, Volume A (State
    Statistical Committee, 2022), Table 3, *Number of permanent and existing population by economic
    regions, administrative-territorial units and settlements* (printed pp.37-56). The PERMANENT
    column is the headline 9,951,409 and is de jure: the people displaced from the districts that
    were under Armenian control in 2019 are counted in the district they come from (Kalbajar 71,039,
    every one of them `temporarily absent`). The EXISTING column, 9,943,958, counts people where the
    census found them, so those districts hold nobody and Baku holds its 293,487 temporary residents.
    **The map is drawn on EXISTING**, because a dot is a place. The occupied territories' own
    residents were not enumerated at all (the table's footnote), so the Karabakh Armenians are in
    neither column.

THE JOIN. Table 3 prints an English name beside each Azerbaijani one; the English is matched to
COD-AB's `adm1_name` through `ALIAS` and a fold, every one of the 74 exactly once. Two witnesses
that use no names: each unit's PERMANENT population against the State Statistical Committee's own
table 1.17 (2019 column, thousands, to the rounding), and each unit's count against its COD-AB area
rank in `az_grid.py` (Kontur).

Usage:
    python sources/az_geo.py --fetch    COD-AB geojson zip and the census Volume A zip into data/raw/az/
    python sources/az_geo.py            rebuild from data/raw/az/
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
RAW = os.path.join(ROOT, "data", "raw", "az")
GEO = os.path.join(ROOT, "data", "geo", "az")
OUT = os.path.join(GEO, "az_units.gpkg")
LOOKUP = os.path.join(GEO, "az_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/ecb2cb09-1270-4a7f-9187-f093efebdef7/resource/"
              "526a0aa4-3011-4cea-bcd4-d9799061617b/download/aze_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "aze_admin_boundaries.geojson.zip")
VOL_A_URL = ("https://www.stat.gov.az/menu/6/statistical_yearbooks/source/"
             "Siyahiyaalinma-2019,%20Cild%20A.zip")
VOL_A_ZIP = os.path.join(RAW, "census2019_A.zip")
VOL_A = os.path.join(RAW, "Siyahiyaalinma-2019, Cild A.pdf")
T117_URL = "https://www.stat.gov.az/source/demoqraphy/en/001_17en.xls"
T117 = os.path.join(RAW, "001_17en.xls")

# Table 3's English spelling -> table 1.17's, where they differ.
T117_ALIAS = {"Sumgait": "Sumgayit", "Jabrail": "Jabrayil", "Khankandi": "Khankendi",
              "Shamakhy": "Shamakhi"}
# Table 1.17 prints thousands to one decimal, and four units (Shahbuz, Aghjabadi, Bilasuvar,
# Kalbajar) differ from Table 3 by 59 to 76 people, more than rounding: a revision after the
# volume. Measured 2026-10-03; the band is 0.1 thousand, so a wrong join (units differ by
# thousands) still fails.
T117_TOL = 100

TABLE3_PAGES = range(36, 57)          # 0-based: printed pp.37-56
PERMANENT = 9_951_409
EXISTING = 9_943_958
N_UNITS = 74

# Table 3's English name -> COD-AB adm1_name, where a fold does not already match.
ALIAS = {
    "Baku": "Baku", "Sumgait": "Sumgait", "Khizi": "Khizy", "Aghjabadi": "Agdjabadi",
    "Aghdash": "Agdash", "Imishli": "Imishly", "Saatli": "Saatly", "Ismayilli": "Ismayilly",
    "Shamakhy": "Shamakhy", "Gubadli": "Gubadly", "Jabrail": "Jabrayil", "Khojavend": "Khojavand",
    "Khankandi": "Khankendi", "Yardimli": "Yardimly", "Masalli": "Masally", "Kangarli": "Kengerli",
    "Babak": "Babek", "Gakh": "Gakh", "Zagatala": "Zagatala", "Lankaran": "Lankaran",
    "Mingachevir": "Mingechevir", "Gusar": "Gusar", "Guba": "Guba", "Siyazan": "Siyazan",
    "Shabran": "Shabran", "Khachmaz": "Khachmaz",
}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z]", "", s)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst in ((COD_AB_URL, COD_AB), (VOL_A_URL, VOL_A_ZIP), (T117_URL, T117)):
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False                     # stat.gov.az's chain is incomplete
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=900, context=ctx) as r:
            data = r.read()
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(data):,} bytes)")
    if not os.path.exists(VOL_A):
        with zipfile.ZipFile(VOL_A_ZIP) as z:
            z.extractall(RAW)


def read_table3():
    """[(English name, permanent, existing)] for every row of Table 3, in printed order."""
    import fitz

    d = fitz.open(VOL_A)
    txt = re.sub(r"\s+", " ", " ".join(d[p].get_text() for p in TABLE3_PAGES))
    n = r"(\d+|-)"
    pat = re.compile(r"((?:[A-Z][a-z'\-]+ ){0,1}[A-Z][a-z'\-]+ |)(district|region|city|Republic of Azerbaijan)"
                     r"\s+" + r"\s+".join([n] * 6) + r"\s+([\d,]+|,0)\s+" + r"\s+".join([n] * 6))
    rows = []
    for m in pat.finditer(txt):
        g = [m.group(i) for i in range(3, 16)]
        v = [0 if x == "-" else int(x) for x in (g[0], g[3], g[7])]
        rows.append((m.group(1).strip(), m.group(2), v[0], v[1], v[2]))
    return rows


def read_t117():
    """{folded English name: 2019 permanent population in thousands} from table 1.17."""
    x = pd.read_excel(T117, header=None)
    out = {}
    for _i, r in x.iloc[6:134].iterrows():          # the urban-and-rural block only
        name = str(r[1]).strip()
        if name == "nan" or pd.isna(r[6]) or "economic region" in name or "including" in name:
            continue
        key = re.sub(r"\s+(district|city)(\s+-\s+total)?$", "", name)
        out[fold(key)] = float(r[6])
    return out


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in (COD_AB, VOL_A, T117)):
        fetch()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("aze_admin1.geojson")))
    if len(g) != N_UNITS or g["adm1_pcode"].nunique() != N_UNITS:
        raise SystemExit(f"COD-AB admin1 has {len(g)} features, expected {N_UNITS}")
    by_fold = {fold(n): p for n, p in zip(g["adm1_name"], g["adm1_pcode"])}
    if len(by_fold) != N_UNITS:
        raise SystemExit("two COD-AB names fold to the same key")

    rows = read_table3()
    nat = [r for r in rows if r[1] == "Republic of Azerbaijan"]
    if not nat or (nat[0][2], nat[0][4]) != (PERMANENT, EXISTING):
        raise SystemExit(f"Table 3's national row is {nat[:1]}, not {PERMANENT:,} / {EXISTING:,}")

    # Baku's and Ganja's districts are printed under them and are not units here; take the first
    # row for each COD-AB unit, which is the unit's own total because the city total precedes its
    # districts and no district shares a name with a unit.
    got = {}
    for name, _kind, perm, absent_or_ex, exist in rows:
        key = fold(ALIAS.get(name, name))
        if key in by_fold and by_fold[key] not in got:
            got[by_fold[key]] = (name, perm, exist)
    missing = sorted(set(g["adm1_pcode"]) - set(got))
    if missing:
        raise SystemExit(f"COD-AB units with no Table 3 row: "
                         f"{[g.loc[g['adm1_pcode'] == p, 'adm1_name'].iloc[0] for p in missing]}")
    perm_sum = sum(v[1] for v in got.values())
    ex_sum = sum(v[2] for v in got.values())
    print(f"Table 3: {len(rows)} rows read; {len(got)} units matched; permanent {perm_sum:,}, "
          f"existing {ex_sum:,}")
    if (perm_sum, ex_sum) != (PERMANENT, EXISTING):
        raise SystemExit(f"the 74 units sum to {perm_sum:,} / {ex_sum:,}, not {PERMANENT:,} / {EXISTING:,}")

    # witness: each unit's permanent population against table 1.17's 2019 column (thousands)
    t117 = read_t117()
    bad = []
    for p, (name, perm, _ex) in got.items():
        k = fold(T117_ALIAS.get(name, name))
        if k not in t117:
            bad.append((name, "absent from 1.17"))
        elif abs(t117[k] * 1000 - perm) > T117_TOL:
            bad.append((name, perm, t117[k]))
    if bad:
        raise SystemExit(f"Table 3 against table 1.17 (2019, thousands): {bad}")
    print(f"  witness: all {len(got)} permanent figures equal table 1.17's 2019 column to the rounding")

    g["unit"] = g["adm1_pcode"]
    g["name"] = g["adm1_name"]
    g["name_az"] = g["adm1_name1"]
    g["pop"] = g["unit"].map({p: v[2] for p, v in got.items()}).astype(int)
    g["perm"] = g["unit"].map({p: v[1] for p, v in got.items()}).astype(int)
    empty = g[g["pop"] == 0].sort_values("name")
    print(f"  units with nobody in the EXISTING column (Armenian-held at the census): {len(empty)}, "
          f"permanent {int(empty['perm'].sum()):,}: {', '.join(empty['name'])}")
    for _i, r in g[(g["pop"] > 0) & (g["pop"] < 0.75 * g["perm"])].iterrows():
        print(f"  partly held at the census: {r['name']:<12} permanent {r['perm']:>8,}  existing {r['pop']:>8,}")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "name_az", "pop", "perm", "geometry"]].to_file(OUT, layer="units", driver="GPKG")
    lut = g[["unit", "name", "name_az", "pop", "perm"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} ({N_UNITS} units, {int(g['pop'].sum()):,} existing, "
          f"{int(g['perm'].sum()):,} permanent)")


if __name__ == "__main__":
    main()
