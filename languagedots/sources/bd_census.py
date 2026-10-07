"""Bangladesh 2011 census, ethnic population by upazila -> data/normalized/bd.csv.

    python sources/bd_census.py [--fetch]

NO LANGUAGE QUESTION. Bangladesh's censuses (2011, 2022) ask ethnic group, not language, so this
is the ethnicity table, and taxonomy/bd2011.py turns each ethnic group into the language it is
taken to speak. Every row is written `tier=derived`. Anita allowed this proxy on 2026-10-05
(relayed by the languagedots supervisor); sources/bd.md is the record.

The table: U.S. Census Bureau, `bangladesh_uscb_202107.xlsx` on HDX (CC BY), sheet
`Religion and Ethnicity`, which carries BBS's 2011 census tabulation down to the 544 upazilas
and thanas: the total population, the total ethnic population, 27 named ethnic groups and
`Other ethnicity`. religiondots draws its religion columns from the same sheet and keys its
Kontur hex layer on the same GEO_MATCH ids, so the join is an identity.

Categories written: the 28 ethnic columns by their USCB English label, plus
`Not an ethnic minority` = total population - total ethnic population, which is Bengali here.

Checks, all asserted:
  * the 28 ethnic columns sum to the published total ethnic population, every row;
  * the 544 upazilas sum to the national row, column by column, and to 144,043,696;
  * 544 distinct GEO_MATCH ids, none with zero population; no nulls, no negatives.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "bd" / "bangladesh_uscb_202107.xlsx"
URL = ("https://data.humdata.org/dataset/862d607b-a195-4f69-b0cb-bd2fe6ac858f/resource/"
       "1fdc9232-9fb4-49d2-b66a-b7a375472f72/download/bangladesh_uscb_202107.xlsx")
OUT = HERE / "data" / "normalized" / "bd.csv"
NATIONAL = 144_043_696
REMAINDER = "Not an ethnic minority"


def fetch():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    RAW.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(req, timeout=300).read()
    if data[:2] != b"PK":
        raise SystemExit("bd: the download is not an xlsx (no PK header)")
    RAW.write_bytes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not RAW.exists():
        fetch()

    df = pd.read_excel(RAW, sheet_name="Religion and Ethnicity", header=0, skiprows=[1])
    eth = [c for c in df.columns if c.startswith("ETH_") and c != "ETH_ETOTL"]
    labels = pd.read_excel(RAW, sheet_name="Religion and Ethnicity", header=0, nrows=1)
    label = {c: str(labels.loc[0, c]).strip() for c in eth}
    if len(eth) != 28:
        raise SystemExit(f"bd: {len(eth)} ethnic columns, expected 28")
    num = ["RLG_TPOP", "ETH_ETOTL"] + eth
    if df[num].isna().any().any() or (df[num] < 0).any().any():
        raise SystemExit("bd: nulls or negatives in the ethnic columns")
    df[num] = df[num].astype("int64")
    bad = df[df[eth].sum(axis=1) != df["ETH_ETOTL"]]
    if len(bad):
        raise SystemExit(f"bd: {len(bad)} rows whose groups do not sum to ETH_ETOTL")

    nat = df[df["ADM_LEVEL"] == 0]
    up = df[df["ADM_LEVEL"] == 3].copy()
    if len(nat) != 1 or int(nat["RLG_TPOP"].iloc[0]) != NATIONAL:
        raise SystemExit("bd: national row missing or not 144,043,696")
    if up["GEO_MATCH"].nunique() != 544 or len(up) != 544 or (up["RLG_TPOP"] <= 0).any():
        raise SystemExit("bd: expected 544 distinct upazilas with population")
    diff = up[num].sum() - nat[num].iloc[0]
    if diff.abs().sum():
        raise SystemExit(f"bd: upazilas do not sum to the national row: {diff[diff != 0]}")
    if (up["ETH_ETOTL"] > up["RLG_TPOP"]).any():
        raise SystemExit("bd: an upazila with more ethnic population than people")

    rows = []
    for r in up.itertuples(index=False):
        r = r._asdict()
        base = dict(geo_id=r["GEO_MATCH"], geo_level="upazila",
                    geo_name=f"{r['ADM2_NAME']} / {r['ADM3_NAME']}", tier="derived")
        rows.append({**base, "source_category": REMAINDER,
                     "count": r["RLG_TPOP"] - r["ETH_ETOTL"]})
        for c in eth:
            if r[c]:
                rows.append({**base, "source_category": label[c], "count": r[c]})
    out = pd.DataFrame(rows)[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier"]]
    if out["count"].sum() != NATIONAL:
        raise SystemExit("bd: written rows do not sum to the census population")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    tot = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"bd.csv: {len(out):,} rows, 544 upazilas, {out['count'].sum():,} people")
    print(f"  ethnic minority total {int(nat['ETH_ETOTL'].iloc[0]):,} "
          f"({nat['ETH_ETOTL'].iloc[0] / NATIONAL:.2%})")
    for k, v in tot.items():
        print(f"  {k:28} {v:>12,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
