"""Nepal NPHC 2021, Table 3 (population by mother tongue) -> data/normalized/np.csv.

    python sources/np_nphc.py [--fetch]

One workbook, Mother_Tongues_NPHC_2021.xlsx, sheet `Prov_District_local level`: a block per
area, the area's name in one of four indented columns (country, province, district, local
level), then one row per mother tongue with total/male/female. Each district ends with an
`INSTITUTIONAL` block (barracks, prisons, hospitals, hostels, monasteries) that has no finer
geography; it is written as geo_level `institutional` and not drawn, as religiondots does.

GEOGRAPHY. geo_id is religiondots' own Nepal id (NP-<prov>-<dist>-<local>), found by joining
(district, local level name) to religiondots/data/normalized/np.csv, which comes from the same
NSO census and spells the 753 names identically. That reuses its 753-unit join to COD-AB
without redoing it.
"""
import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
RAW = HERE / "data" / "raw" / "np" / "Mother_Tongues_NPHC_2021.xlsx"
URL = "https://censusresults.nsonepal.gov.np/files/caste/Mother_Tongues_NPHC_2021.xlsx"
OUT = HERE / "data" / "normalized" / "np.csv"


def fetch():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(urllib.request.urlopen(req, context=ctx, timeout=120).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not RAW.exists():
        fetch()

    df = pd.read_excel(RAW, sheet_name="Prov_District_local level", header=None, dtype=str)
    rows = []
    prov = dist = local = None
    level = None
    for r in df.itertuples(index=False):
        c0, c1, c2, c3, mt, tot = r[0], r[1], r[2], r[3], r[4], r[5]
        # Provinces sit in column 0, except Bagmati, which is one column to the right; read
        # as anything else, its 6.1M people land in the previous district's institutional block.
        head = c0 if isinstance(c0, str) and c0.strip() else c1
        if isinstance(head, str) and head.strip() and not head.startswith("Table") and head != "Area":
            if head.strip() == "0":
                continue
            prov, dist, local = head.strip(), None, None
            level = "nation" if prov == "NEPAL" else "province"
            continue
        if isinstance(c2, str) and c2.strip():
            dist, local, level = c2.strip(), None, "district"
            continue
        if isinstance(c3, str) and c3.strip():
            local = c3.strip()
            level = "institutional" if local == "INSTITUTIONAL" else "local"
            continue
        if isinstance(mt, str) and mt.strip() and isinstance(tot, str) and tot.strip():
            if mt.strip() in ("Mother Tongue",):
                continue
            rows.append(dict(level=level, province=prov, district=dist, local=local,
                             source_category=mt.strip(), count=int(float(tot))))
    out = pd.DataFrame(rows)
    is_total = out["source_category"].str.startswith("All M")
    totals = out[is_total]
    out = out[~is_total]

    # every block's mother tongues add up to its own "All MTongues" row
    key = ["level", "province", "district", "local"]
    tot = totals.fillna("").groupby(key)["count"].sum()
    parts = out.fillna("").groupby(key)["count"].sum()
    diff = (tot - parts.reindex(tot.index).fillna(0)).abs()
    if diff.max() > 0:
        raise SystemExit(f"blocks whose mother tongues do not add up:\n{diff[diff > 0].head()}")

    loc = out[out["level"] == "local"]
    n_local = loc[["district", "local"]].drop_duplicates()
    if len(n_local) != 753:
        raise SystemExit(f"{len(n_local)} local levels, expected 753")
    nat = out.loc[out["level"] == "nation", "count"].sum()
    inst = out.loc[out["level"] == "institutional", "count"].sum()
    if loc["count"].sum() + inst != nat:
        raise SystemExit(f"local levels {loc['count'].sum():,} + institutional {inst:,} != nation {nat:,}")

    # religiondots' ids for the same 753 local levels
    rd = pd.read_csv(RD / "data" / "normalized" / "np.csv", dtype=str)
    rd = rd[rd["geo_level"] == "local"].drop_duplicates("geo_id")
    rd["district"] = rd["note"].str.extract(r"district=([^;]+)")[0].str.strip()
    ids = dict(zip(zip(rd["district"], rd["geo_name"]), rd["geo_id"]))
    loc = loc.copy()
    # The two Nawalparasi districts are "Nawalparasi (Bardaghat Susta East)" here and spelled
    # differently in the religion file; for those, match the local level inside any district
    # sharing the first word, and only where that is unique.
    def lookup(d, l):
        if (d, l) in ids:
            return ids[(d, l)]
        hits = [g for (dd, ll), g in ids.items()
                if ll == l and dd.split()[0].lower() == d.split()[0].lower()]
        return hits[0] if len(hits) == 1 else None
    loc["geo_id"] = [lookup(d, l) for d, l in zip(loc["district"], loc["local"])]
    if loc.drop_duplicates(["district", "local"])["geo_id"].duplicated().any():
        raise SystemExit("two local levels matched the same religiondots id")
    miss = loc.loc[loc["geo_id"].isna(), ["district", "local"]].drop_duplicates()
    if len(miss):
        raise SystemExit(f"{len(miss)} local levels have no religiondots id:\n{miss.head(10)}")

    loc = loc.rename(columns={"local": "geo_name"}).assign(geo_level="local")
    inst_rows = out[out["level"] == "institutional"].assign(
        geo_level="institutional", geo_name="INSTITUTIONAL",
        geo_id=lambda d: "NP-INST-" + d["district"])
    res = pd.concat([loc, inst_rows], ignore_index=True)
    res = res[res["count"] > 0]
    res = res[["geo_id", "geo_level", "geo_name", "district", "source_category", "count"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(res):,} rows; 753 local levels hold {loc['count'].sum():,}, "
          f"institutional {inst:,}, nation {nat:,}; "
          f"{res['source_category'].nunique()} mother tongues")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
