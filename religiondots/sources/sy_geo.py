"""Syria: the 14 governorates, COD-AB polygons with the Central Bureau of Statistics' estimate of the
people living in each at the end of 2011.

Writes data/geo/sy/sy_governorates.gpkg and data/geo/sy/sy_lookup.csv. `sources/sy.md` §4 is the
record.

  * **boundaries**: COD-AB `cod-ab-syr` v02 (OCHA Syria, UN Cartographic Section; valid from
    2020-12-17, reviewed 2024-04-01), `syr_admin1.geojson`: 14 governorates SY01-SY14.
  * **population**: CBS, *Statistical Abstract 2012*, Table 3/2, "Estimates of population actually
    living in Syria by governorate and sex (000) on 31/12/2011", footnoted "The number doesn't
    include Syrian population abroad"; 21,377 thousand. Republished by OCHA Syria on HDX
    (`syrian-arab-republic-other-0`, `syr_pop_2011.xls`, sheet `EST. POPULATION 12-2011`).

WHY 2011, AND NOT NOW. The population base is a choice between a pre-war official estimate and a
current one nobody may use:

  * OCHA's current baseline (Population Task Force, August 2025, admin 1-4,
    `syrian-arab-republic-baseline-population`) is marked confidential on HDX, "intended for
    operational and programmatic use only and cannot be shared for academic or research purposes".
    Not used.
  * The public humanitarian files (HNO 2024 and 2025, HNRP 2026, JIAF 2025) print people in need
    by sub-district and no total population.
  * The US Census Bureau package (`syria_uscb_201811.xlsx`) carries 2014 and 2016 estimates from
    wartime reports, already out of date by 2018.
  * Kontur, WorldPop and GHSL spread census-era admin totals over buildings, so they do not see
    displacement either.

So the dots stand where people lived at the end of 2011, by the state's own last pre-war estimate,
and the note says how far that is from now. The national mix is Pew's for 2020 (21,049,429
people), within 2% of this total.

THE GOLAN IS CLIPPED. COD-AB's Quneitra includes the part Israel has administered since 1967, which
this map draws inside Israel (`sources/il_geo.py`, Anita 2026-09-07). Natural Earth's
`ne_10m_admin_0_disputed_areas` names it (`BRK_NAME` "Golan Heights", "Admin. By Israel; Claimed by
Syria"), the same layer `country_shapes.py` cuts with, and the same publisher whose country outline
already leaves it out of Syria. CBS's Quneitra figure (90,000) counts only the Syrian-held part.
The UNDOF zone stays in Syria, as Natural Earth has it. The area removed is printed and pinned.

The join is on the CBS table's row order against COD p-codes SY01-SY14 (the same order CBS has
used since the 1980s), with both the English and the Arabic names as witnesses.

Usage:
    python sources/sy_geo.py --fetch    COD-AB geojson zip and the CBS 2011 workbook into data/raw/sy/
    python sources/sy_geo.py            rebuild from data/raw/sy/
"""

import io
import json
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
RAW = os.path.join(ROOT, "data", "raw", "sy")
GEO = os.path.join(ROOT, "data", "geo", "sy")
OUT = os.path.join(GEO, "sy_governorates.gpkg")
LOOKUP = os.path.join(GEO, "sy_lookup.csv")
DISPUTED = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_disputed_areas.geojson")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/356a63e9-90aa-4b9c-a938-58ef24469c00/resource/"
              "ab4b6f19-3854-4b2b-8f5c-b5dc474a4c0c/download/syr_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "syr_admin_boundaries.geojson.zip")
CBS_URL = ("https://data.humdata.org/dataset/c7441fdb-2f35-477e-80cb-c6242aac9aee/resource/"
           "103d57e9-cf7c-402a-891a-c2d59d820144/download/syr_pop_2011.xls")
CBS = os.path.join(RAW, "syr_pop_2011.xls")
CBS_SHEET = "EST. POPULATION 12-2011"
CBS_TOTAL_000 = 21_377

GOLAN = "Golan Heights"
QUNEITRA = "SY14"
GOLAN_KM2 = (1_000, 1_400)       # clipped from SY14, pinned; measured 2026-10-03 (printed)
METRIC_AREA = "ESRI:54034"

# CBS English label (folded) -> COD-AB adm1_name (folded), where they differ. A label not here must
# fold to the COD name itself; the Arabic names are a second witness.
CBS_TO_COD = {"alrakka": "arraqqa", "daraa": "dara", "alsweida": "assweida",
              "alquneitra": "quneitra"}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z]", "", s)


def fold_ar(s):
    """Arabic letters only, tatweel dropped, alef and teh marbuta forms unified."""
    s = re.sub(r"[^ء-ي]", "", str(s).replace("ـ", ""))
    return s.translate(str.maketrans({"إ": "ا", "أ": "ا", "آ": "ا",
                                      "ة": "ه"}))


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst, magic in ((COD_AB_URL, COD_AB, b"PK"), (CBS_URL, CBS, b"\xd0\xcf\x11\xe0")):
        if os.path.exists(dst) and os.path.getsize(dst) > 40_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if not data.startswith(magic):
            raise SystemExit(f"{url} did not return the expected file (starts {data[:8]!r})")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(data):,} bytes)")


def cbs_2011():
    """[(english, arabic, people)] in the table's row order, and the printed total, both asserted."""
    t = pd.read_excel(CBS, sheet_name=CBS_SHEET, header=None)
    title = " ".join(str(x) for x in t.iloc[:3, 0])
    if "ACTUALLY LIVING IN SYRIA" not in title or "31 / 12 / 2011" not in title:
        raise SystemExit(f"{CBS_SHEET} title changed: {title[:200]!r}")
    rows = []
    for _i, r in t.iterrows():
        name, tot = str(r[0]).strip(), r[1]
        if name == "TOTAL":
            if int(tot) != CBS_TOTAL_000:
                raise SystemExit(f"CBS 2011 TOTAL is {tot}, not {CBS_TOTAL_000}")
            break
        if isinstance(tot, (int, float)) and not pd.isna(tot) and name not in ("nan", "GOVERNORATE"):
            rows.append((name, str(r[4]), int(tot) * 1000))
    if len(rows) != 14 or sum(p for _e, _a, p in rows) != CBS_TOTAL_000 * 1000:
        raise SystemExit(f"CBS 2011: {len(rows)} governorates summing to "
                         f"{sum(p for _e, _a, p in rows):,}")
    return rows


def main():
    import geopandas as gpd
    from shapely.geometry import shape

    if "--fetch" in sys.argv or not (os.path.exists(COD_AB) and os.path.exists(CBS)):
        fetch()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("syr_admin1.geojson")))
    print(f"COD-AB admin1: {len(g)} features")
    want = {f"SY{i:02d}" for i in range(1, 15)}
    if len(g) != 14 or set(g["adm1_pcode"]) != want:
        raise SystemExit(f"COD-AB admin1 is not SY01-SY14: {sorted(g['adm1_pcode'])}")

    # ---- the Golan: Natural Earth's Israeli-administered polygon, out of Quneitra ----
    ne = json.load(open(DISPUTED, encoding="utf-8"))
    golan = [f for f in ne["features"] if f["properties"].get("BRK_NAME") == GOLAN]
    if len(golan) != 1 or "Admin. By Israel" not in str(golan[0]["properties"].get("NOTE_BRK")):
        raise SystemExit(f"{DISPUTED} no longer has one '{GOLAN}' administered by Israel")
    gol = gpd.GeoSeries([shape(golan[0]["geometry"])], crs=4326).to_crs(g.crs).iloc[0]
    before = g.to_crs(METRIC_AREA).area / 1e6
    g["geometry"] = g.geometry.difference(gol)
    after = g.to_crs(METRIC_AREA).area / 1e6
    cut = before - after
    for i, c in cut.items():
        if c > 1.0:
            print(f"  Golan clip: {g.loc[i, 'adm1_name']} loses {c:,.0f} km2 "
                  f"({before[i]:,.0f} -> {after[i]:,.0f})")
    q = g.index[g["adm1_pcode"] == QUNEITRA][0]
    if not GOLAN_KM2[0] <= cut[q] <= GOLAN_KM2[1]:
        raise SystemExit(f"the Golan clip took {cut[q]:,.0f} km2 from Quneitra, outside {GOLAN_KM2}")
    if (cut.drop(q) > 5.0).any():
        raise SystemExit(f"the Golan clip touched another governorate: {cut.drop(q)[cut.drop(q) > 5]}")

    # ---- CBS 2011, by row order, with both names as witnesses ----
    rows = cbs_2011()
    by_pc = g.set_index("adm1_pcode")
    pop, cbs_name = {}, {}
    for k, (en, ar, people) in enumerate(rows, start=1):
        pc = f"SY{k:02d}"
        cod_en, cod_ar = by_pc.loc[pc, "adm1_name"], by_pc.loc[pc, "adm1_name1"]
        if CBS_TO_COD.get(fold(en), fold(en)) != fold(cod_en):
            raise SystemExit(f"row {k}: CBS {en!r} against COD {pc} {cod_en!r}")
        if fold_ar(ar) != fold_ar(cod_ar):
            raise SystemExit(f"row {k}: CBS Arabic {ar!r} against COD {pc} {cod_ar!r}")
        pop[pc], cbs_name[pc] = people, en
    print("  14 governorates joined by row order; English (4 respellings pinned) and Arabic agree")

    g["unit"] = g["adm1_pcode"]
    g["name"] = g["adm1_name"]
    g["pop"] = g["unit"].map(pop).astype(int)
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("\n  governorate, CBS end-2011 estimate, area km2 (after the clip), people per km2:")
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda t: -t[0][1]["pop"]):
        print(f"      {r['unit']}  {r['name']:<15} {r['pop']:>10,}  {a:>9,.0f}  {r['pop'] / a:>8.1f}")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="governorates", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (14 governorates, {int(g['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
