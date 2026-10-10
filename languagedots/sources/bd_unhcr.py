"""Rohingya refugees registered in Bangladesh's camps, by camp (UNHCR) -> data/normalized/bd_rohingya.csv.

    python sources/bd_unhcr.py [--fetch]

The camps (Ukhia and Teknaf, Cox's Bazar, and Bhasan Char) are in no census. Anita asked for them
to be drawn (2026-10-07, relayed by the languagedots supervisor): a counts proxy that ADDS people
the census does not have, drawn as Rohingya, every row `derived`. sources/bd.md §5 is the record.

Source: UNHCR Operational Data Portal, Bangladesh, "Rohingya refugees and asylum-seekers" by
settlement ("Government, UNHCR": the joint Government of Bangladesh and UNHCR registration), the
JSON behind the portal's camp widget:
  https://data.unhcr.org/population/get/sublocation/root?widget_id=691150&geo_id=591
      &population_group=5556&forcesublocation=true&fromDate=1900-01-01
Open, no login. Every row is dated (`date`, 2026-08-31 on the 2026-10-07 fetch).

Join: each Cox's Bazar camp to the ISCG A1 camp outline of the same name (sources/bd_camps.py),
asserted. Two rows have no outline: `Bhasan Char` (the island camp in the Meghna estuary, placed
on its own coordinates by countries/bd.py) and `Other Camp` (369 people, UNHCR's own remainder,
put on the camp outline holding its point). Kutupalong RC and Nayapara RC, the two registered
camps from the 1990s, have outlines but no row of their own on the portal; they get no people.

Checks: every row is one date; counts are positive integers; the total is between 0.9 and 1.4
million; every Cox's Bazar row joins to exactly one outline.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "bd" / "unhcr_rohingya_camps.json"
URL = ("https://data.unhcr.org/population/get/sublocation/root?widget_id=691150&geo_id=591"
       "&population_group=5556&forcesublocation=true&fromDate=1900-01-01")
CAMPS = HERE / "data" / "geo" / "bd" / "bd_camps.gpkg"
OUT = HERE / "data" / "normalized" / "bd_rohingya.csv"


def key(name):
    """'Camp 1E' / 'Camp 01E' / 'Camp 20 Extension' -> a comparable key."""
    s = name.strip().lower().replace("extension", "x")
    s = re.sub(r"camp\s*0*(\d+)\s*", r"c\1", s)
    return re.sub(r"\s+", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not RAW.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    rows = json.loads(RAW.read_text(encoding="utf-8"))["data"]
    df = pd.DataFrame([{"camp": r["name"], "date": r["date"], "count": int(r["individuals"]),
                        "lon": float(r["centroid_lon"]), "lat": float(r["centroid_lat"])}
                       for r in rows])
    if df["date"].nunique() != 1:
        raise SystemExit(f"bd_unhcr: several dates {sorted(df['date'].unique())}")
    if (df["count"] <= 0).any() or df["camp"].duplicated().any():
        raise SystemExit("bd_unhcr: a non-positive count or a duplicated camp")
    tot = int(df["count"].sum())
    if not 900_000 <= tot <= 1_400_000:
        raise SystemExit(f"bd_unhcr: total {tot:,} outside 0.9-1.4 million")

    import geopandas as gpd
    out = gpd.read_file(CAMPS)
    okeys = {key(n): n for n in out["CampName"]}
    if len(okeys) != len(out):
        raise SystemExit("bd_unhcr: two outlines share a key")
    df["outline"] = df["camp"].map(lambda n: okeys.get(key(n)))
    special = {"Bhasan Char", "Other Camp"}
    bad = df[df["outline"].isna() & ~df["camp"].isin(special)]
    if len(bad):
        raise SystemExit(f"bd_unhcr: camps with no outline: {bad['camp'].tolist()}")
    if df["outline"].dropna().duplicated().any():
        raise SystemExit("bd_unhcr: two camps joined to one outline")
    if set(df["camp"]) & special != special:
        raise SystemExit("bd_unhcr: Bhasan Char or Other Camp missing; recheck the join rules")
    unused = sorted(set(out["CampName"]) - set(df["outline"].dropna()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"bd_rohingya.csv: {len(df)} camps, {tot:,} people, dated {df['date'].iloc[0]}")
    print(f"  Bhasan Char {int(df.loc[df.camp == 'Bhasan Char', 'count'].iloc[0]):,}; "
          f"outlines with no UNHCR row: {unused}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
