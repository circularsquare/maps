"""Ethiopia 2007 census, mother tongue by zone -> data/normalized/et.csv.

    python sources/et_uscb.py [--fetch]

SOURCE. The U.S. Census Bureau's "Ethiopia Subnational Population and Housing Data Tables with
Administrative Boundaries" on HDX (CC BY), the same release religiondots draws Ethiopia's
religion from (religiondots/sources/et.py). Its `Language` sheet transcribes CSA's 2007 census
Table 3.2, "Population by Urban-Rural Residence, Sex, and Mother Tongue", from the eleven
regional reports, for the country, 13 regions and 94 zones. Both sexes, urban and rural together.
The question: mother tongue, "the language used during childhood when speaking with family
members" (the metadata, quoting CSA).

ZONES, NOT WOREDAS. The queue row said woreda; that was the religion table. USCB's Language and
Ethnicity sheets stop at ADM2, because CSA's regional reports tabulate mother tongue by zone
(its religion table goes to woreda, which is why religiondots has 738 units). 93 zones carry
figures, about 793,000 people each; Finfinne Zuria special zone is a row with no data and no
polygon (it was not a 2007 tabulation unit).

LABELS. USCB renamed every column to an ISO 639-3 language name "where feasible", and some of
those are wrong in ways that matter (CSA's `Mossigna` became Mossi of Burkina Faso, `UPO` became
Ignaciano of Bolivia, `Shegna` became She of China). The Data Dictionary sheet keeps CSA's own
field name for every column ("Original field name: \"Mother Tongue, Oromigna.\""), so the
normalised `source_category` is CSA's spelling, and USCB's name is carried only as a column for
reference. taxonomy/et2007.py maps from CSA's names.

CHECKS (all must pass):
  1. every LNG_ column has exactly one CSA name in the dictionary, and the names are unique
  2. the 91 categories sum to 73,750,932 nationally, the census population (Age-Sex, ETH_00)
  3. every zone's categories sum to that zone's Age-Sex total (USCB: the Age-Sex total is the
     universe of the Language sheet at ADM0-2)
  4. zones, and separately regions, sum to the national figure category by category
  5. every zone's total equals the sum of its woredas in the Religion sheet (CSA Table 3.4, a
     different table, read by religiondots); this is also what proves the woreda -> zone nesting
     used by countries/et.py
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "et"
XLSX = RAW / "ethiopia_uscb_202308.xlsx"
URL = ("https://data.humdata.org/dataset/5438946a-51b7-44d6-9b76-aeafdb4dc4d3/resource/"
       "f112d5fa-d90e-4352-b0c1-54cffbbad62d/download/ethiopia_uscb_202308.xlsx")
OUT = HERE / "data" / "normalized" / "et.csv"

NATIONAL = 73_750_932
N_ZONES = 93            # with data; Finfinne Zuria is the 94th row and has none
N_CATEGORIES = 91
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers=UA)
    data = urllib.request.urlopen(req, timeout=300).read()
    if data[:4] != b"PK\x03\x04" or len(data) < 1_000_000:
        raise SystemExit(f"{XLSX.name}: not an xlsx ({len(data):,} bytes, starts {data[:16]!r})")
    XLSX.write_bytes(data)
    print(f"wrote {XLSX} ({len(data):,} bytes)")


def sheet(name):
    # row 0 is the field code, row 1 the field's English description
    return pd.read_excel(XLSX, sheet_name=name, header=0, skiprows=[1])


def csa_names():
    d = pd.read_excel(XLSX, sheet_name="Data Dictionary", header=None)
    out = {}
    for r in d.itertuples(index=False):
        v = [str(x) for x in r if pd.notna(x)]
        if len(v) >= 3 and v[0].startswith("LNG_"):
            m = re.search(r'Original field name: "Mother Tongue, (.*?)\.?"', v[2])
            if not m:
                raise SystemExit(f"{v[0]}: no original field name in the dictionary")
            if v[0] in out:
                raise SystemExit(f"{v[0]} twice in the dictionary")
            out[v[0]] = (m.group(1).strip(), v[1].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not XLSX.exists():
        fetch()

    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Ethiopia, 2007 census mother tongue (CSA Table 3.2 via USCB)\n")
    names = csa_names()
    lang = sheet("Language")
    cols = [c for c in lang.columns if c.startswith("LNG_")]
    csa = [names[c][0] for c in cols]
    report(len(cols) == N_CATEGORIES and set(cols) == set(names) and len(set(csa)) == len(csa),
           f"{len(cols)} language columns, each with one CSA name, all distinct")

    for c in cols:
        lang[c] = pd.to_numeric(lang[c], errors="coerce")
    neg = int((lang[cols] < 0).sum().sum())
    report(neg == 0, f"{neg} negative cells (USCB's -999 sentinel does not occur in this sheet)")
    lang["TOT"] = lang[cols].sum(axis=1, min_count=1)

    nat = lang[lang["ADM_LEVEL"] == 0].iloc[0]
    report(int(nat["TOT"]) == NATIONAL,
           f"categories sum to {int(nat['TOT']):,} nationally (census population {NATIONAL:,})")

    age = sheet("Age-Sex").set_index("GEO_MATCH")["BTOTL"]
    zones = lang[(lang["ADM_LEVEL"] == 2) & lang["TOT"].notna()].copy()
    empty = lang[(lang["ADM_LEVEL"] == 2) & lang["TOT"].isna()]
    report(len(zones) == N_ZONES and list(empty["GEO_MATCH"]) == ["ETH_08_21"],
           f"{len(zones)} zones with data; no data only for "
           f"{', '.join(empty['AREA_NAME'])} ({', '.join(empty['GEO_MATCH'])})")
    off = (zones.set_index("GEO_MATCH")["TOT"] - age).dropna()
    off = off[zones["GEO_MATCH"].tolist()]
    report((off == 0).all(), f"every zone's languages sum to its Age-Sex population "
                             f"({int((off != 0).sum())} differ)")

    for lv, label in ((2, "zones"), (1, "regions")):
        sub = lang[(lang["ADM_LEVEL"] == lv) & lang["TOT"].notna()]
        bad = [c for c in cols if int(sub[c].sum()) != int(nat[c])]
        report(not bad, f"{label} sum to the national figure on all {len(cols)} categories "
                        f"({len(bad)} differ{': ' + ', '.join(bad[:5]) if bad else ''})")

    # 5. the religion table's woredas, summed to their zone
    rel = sheet("Religion")
    rel = rel[rel["ADM_LEVEL"] == 3].copy()
    rcols = [c for c in rel.columns if c.startswith("RLG_") and c.endswith("_B")]
    for c in rcols:
        rel[c] = pd.to_numeric(rel[c], errors="coerce").mask(lambda s: s < 0)
    rel["TOT"] = rel[rcols].sum(axis=1, min_count=1)
    rel["zone"] = rel["GEO_MATCH"].str.rsplit("_", n=1).str[0]
    names_ok = (rel["zone"].map(lang.set_index("GEO_MATCH")["AREA_NAME"]) == rel["ADM2_NAME"]).all()
    report(names_ok, "every woreda id's zone prefix names the woreda's own zone (ADM2_NAME)")
    wsum = rel.groupby("zone")["TOT"].sum()
    d = zones.set_index("GEO_MATCH")["TOT"] - wsum.reindex(zones["GEO_MATCH"]).fillna(0)
    report((d == 0).all(), f"every zone's total equals its woredas' total in the religion table "
                           f"({int((d != 0).sum())} differ, largest {int(d.abs().max()):,})")
    if (d != 0).any():
        print(d[d != 0].to_string())

    if not ok:
        raise SystemExit("\nreconciliation FAILED; nothing written")

    long = zones.melt(id_vars=["GEO_MATCH", "AREA_NAME", "ADM1_NAME"], value_vars=cols,
                      var_name="uscb_field", value_name="count")
    long = long[long["count"] > 0].copy()
    long["count"] = long["count"].astype("int64")
    long["source_category"] = long["uscb_field"].map(lambda c: names[c][0])
    long["uscb_name"] = long["uscb_field"].map(lambda c: names[c][1])
    long["geo_level"] = "zone"
    out = long.rename(columns={"GEO_MATCH": "geo_id", "AREA_NAME": "geo_name", "ADM1_NAME": "region"})
    out = out[["geo_id", "geo_level", "geo_name", "region", "source_category", "count",
               "uscb_field", "uscb_name"]].sort_values(["geo_id", "count"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}: {len(out):,} rows, {out['geo_id'].nunique()} zones, "
          f"{out['count'].sum():,} people, {out['source_category'].nunique()} languages with speakers")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
