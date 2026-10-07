"""Pakistan Digital Census 2023, Table 11 (population by mother tongue) -> data/normalized/pk.csv.

    python sources/pk_t11.py [--fetch]

PBS published Table 11 as per-province PDFs whose URLs now 404. The CRAN package PakPC2023
(GPL-2, by Muhammad Yaseen and colleagues) carries every 2023 census table as data;
TABLE_11.RData is read straight out of its source tarball, no R needed (`pip install rdata`).

Rows are (tehsil, language) with all sexes / female / male / transgender, overall / rural /
urban; only ALL_SEXES_OVERALL is kept. Fourteen named languages plus OTHERS and a TOTAL row,
which is checked against the languages and then dropped.

GEOGRAPHY. Tehsils are summed to districts and matched to religiondots' 2023 district ids
(PK23-<province>/<district>) on name, because religiondots' placement hexes are cut by district.
Its district set is the census's own, so the match is checked to be one-to-one and complete.
Tehsil resolution is available in this table and is a later step (the hexes would need a
tehsil key first).
"""
import argparse
import re
import sys
import tarfile
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
RAW = HERE / "data" / "raw" / "pk" / "PakPC2023_0.2.0.tar.gz"
URL = "https://cran.r-project.org/src/contrib/PakPC2023_0.2.0.tar.gz"
OUT = HERE / "data" / "normalized" / "pk.csv"

PROVINCE = {"BALOCHISTAN": "balochistan", "ISLAMABAD": "islamabad", "KPK": "khyber-pakhtunkhwa",
            "PUNJAB": "punjab", "SINDH": "sindh"}


# religiondots' id is "<district>-district" except where the census unit is not a district
ALIAS = {"malakand": "malakand-protected-area", "tando-allah-yar": "tando-allahyar-district"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def read_table():
    import rdata
    with tarfile.open(RAW) as t:
        raw = t.extractfile("PakPC2023/data/TABLE_11.RData").read()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = rdata.conversion.convert(rdata.parser.parse_data(raw))
    df = conv["TABLE_11"]
    df.columns = [str(c) for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(urllib.request.urlopen(URL, timeout=120).read())

    df = read_table()
    for c in ("PROVINCE", "DIVISION", "DISTRICT", "TEHSIL", "ADMIN_UNIT", "LANGUAGE"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    print("ADMIN_UNIT:", df["ADMIN_UNIT"].value_counts(dropna=False).to_dict())
    df["count"] = df["ALL_SEXES_OVERALL"].fillna(0).astype("int64")
    # The package lost PROVINCE and DIVISION on some tehsils (Tando Allahyar's three talukas
    # among them). Fill from the district's other rows, else from this table of known ones.
    known = df.dropna(subset=["PROVINCE"]).drop_duplicates("DISTRICT").set_index("DISTRICT")["PROVINCE"]
    known = {**{"TANDO ALLAHYAR": "SINDH"}, **known.to_dict()}
    lost = df["PROVINCE"].isna()
    df.loc[lost, "PROVINCE"] = df.loc[lost, "DISTRICT"].map(known)
    if df["PROVINCE"].isna().any():
        raise SystemExit(f"districts with no province: {sorted(df.loc[df['PROVINCE'].isna(), 'DISTRICT'].unique())}")
    print(f"  filled the province on {int(lost.sum())} rows: "
          f"{sorted(df.loc[lost, 'DISTRICT'].unique())}")
    df = df[df["TEHSIL"].notna()].copy()

    key = ["PROVINCE", "DISTRICT", "TEHSIL", "ADMIN_UNIT"]
    tot = df[df["LANGUAGE"] == "TOTAL"].groupby(key)["count"].sum()
    parts = df[df["LANGUAGE"] != "TOTAL"].groupby(key)["count"].sum()
    diff = (tot - parts.reindex(tot.index).fillna(0)).abs()
    if diff.max() > 0:
        raise SystemExit(f"tehsils whose languages do not add to TOTAL:\n{diff[diff > 0].head()}")
    df = df[df["LANGUAGE"] != "TOTAL"]
    print(f"  {df.groupby(key).ngroups} tehsils, {df['count'].sum():,} people, "
          f"languages {sorted(df['LANGUAGE'].unique())}")

    dist = df.groupby(["PROVINCE", "DISTRICT", "LANGUAGE"], as_index=False)["count"].sum()
    dist["geo_id"] = [f"PK23-{PROVINCE[p]}/{ALIAS.get(slug(d), slug(d) + '-district')}"
                      for p, d in zip(dist["PROVINCE"], dist["DISTRICT"])]

    rd = pd.read_csv(RD / "data" / "normalized" / "pk.csv", dtype=str)
    rd = rd[(rd["geo_level"] == "district") & ~rd["geo_id"].str.contains("azad", case=False)]
    rd_ids = set(rd["geo_id"])
    ours = set(dist["geo_id"])
    if ours != rd_ids:
        only_ours = sorted(ours - rd_ids)
        only_rd = sorted(rd_ids - ours)
        raise SystemExit(f"district ids differ from religiondots'.\n  only here: {only_ours}\n"
                         f"  only in religiondots: {only_rd}")

    # The same census's religion table (Table 9, read by religiondots from PBS's own PDFs) must
    # give every district the same population: two independent transcriptions, one check.
    rd["count"] = pd.to_numeric(rd["count"])
    rel = rd.groupby("geo_id")["count"].sum()
    lang = dist.groupby("geo_id")["count"].sum()
    off = (lang - rel.reindex(lang.index)).abs()
    if off.max() > 0:
        raise SystemExit(f"district totals differ from religiondots' Table 9:\n{off[off > 0].head(10)}")
    print(f"  all {len(lang)} district totals equal Table 9's (religiondots/data/normalized/pk.csv)")

    out = dist.rename(columns={"DISTRICT": "geo_name", "LANGUAGE": "source_category"})
    out["geo_level"] = "district"
    out = out[out["count"] > 0][["geo_id", "geo_level", "geo_name", "source_category", "count"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(out):,} rows, {out['geo_id'].nunique()} districts, "
          f"{out['count'].sum():,} people")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
