"""Canada, Census of Population 2021, mother tongue by dissemination area -> data/normalized/ca.csv.

    python sources/ca_census.py [--fetch]

SOURCE. Statistics Canada, Census Profile 2021, catalogue 98-401-X2021006 (dissemination areas,
with the country, provinces, census divisions and subdivisions above them), Statistics Canada
Open Licence. Six regional zips of one long CSV each (latin-1): every geography repeats 2,631
characteristics. Mother tongue is characteristics 393-723, "Total - Mother tongue for the total
population excluding institutional residents - 100% data": the short form, everyone.

The profile's mother-tongue list is the full 2021 classification, indented by leading spaces:
single responses (English, French, then 70-odd Indigenous and 200-odd other languages, with
group rows such as "Cree languages" above their members) and five multiple-response rows
("English and French", "English and non-official language(s)", ...). This script writes the
LEAVES of the single responses (a row with no deeper row under it) and the five multiple rows;
group rows are only used to check that the members add up.

WHERE THE ZIPS COME FROM. ancestrydots/canada/data/raw/profile_<region>.zip already holds all six
(downloaded 2026-07-13 for the ancestry map); they are read in place, read-only. Otherwise
--fetch downloads them into data/raw/ca/.

GEOGRAPHY. Rows run in geographic order: a census subdivision's row, then its dissemination
areas. Each DA's subdivision is taken from that order (DAs nest in CSDs by construction) and
checked against the DA count each CSD has in the same file.

Counts are randomly rounded to a multiple of 5 by StatCan, so sums agree only to within a few
units per cell.
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ANC = HERE.parent / "ancestrydots" / "canada" / "data" / "raw"
RAW = HERE / "data" / "raw" / "ca"
OUT = HERE / "data" / "normalized" / "ca.csv"
OUT_UNITS = HERE / "data" / "normalized" / "ca_units.csv"
URL = ("https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/"
       "download-telecharger/comp/GetFile.cfm?Lang=E&FILETYPE=CSV&GEONO=006_{region}")
# (ancestrydots' file stem, StatCan's GEONO region)
REGIONS = [("Territories_Territoires", "Territories_Territoires"), ("Atlantic", "Atlantic"),
           ("Quebec", "Quebec"), ("Ontario", "Ontario"), ("Prairies", "Prairies"),
           ("BC", "British_Columbia")]

FIRST, LAST = 393, 723          # the mother-tongue block
POP = 1                         # Population, 2021
EXPECT = {393: "Total - Mother tongue for the total population excluding institutional residents",
          394: "Single responses", 718: "Multiple responses",
          719: "English and French", 723: "Multiple non-official languages",
          724: "Total - All languages spoken at home"}
LEVEL = {"Country": "country", "Province": "province", "Territory": "province",
         "Census division": "cd", "Census subdivision": "csd", "Dissemination area": "da"}


def zip_path(stem, region, fetch):
    p = ANC / f"profile_{stem}.zip"
    if p.exists():
        return p
    p = RAW / f"profile_{stem}.zip"
    if p.exists():
        return p
    if not fetch:
        raise SystemExit(f"no profile zip for {region} -- run with --fetch")
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL.format(region=region), headers={"User-Agent": "Mozilla/5.0"})
    tmp = p.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=3600) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(p)
    return p


def read_region(path):
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if "_CSV_data_" in n)
    cols = [1, 3, 4, 5, 7, 8, 9, 11]
    keep = []
    with z.open(name) as f:
        for ch in pd.read_csv(f, encoding="latin-1", usecols=cols, chunksize=2_000_000,
                              dtype={"DGUID": str, "GEO_LEVEL": str, "GEO_NAME": str,
                                     "TNR_SF": str, "DATA_QUALITY_FLAG": str,
                                     "CHARACTERISTIC_ID": "int32", "CHARACTERISTIC_NAME": str,
                                     "C1_COUNT_TOTAL": "float64"}):
            m = ch["CHARACTERISTIC_ID"].between(FIRST, LAST) | ch["CHARACTERISTIC_ID"].isin(
                [POP, 724])
            keep.append(ch[m])
    df = pd.concat(keep, ignore_index=True)
    df["region"] = path.stem.replace("profile_", "")
    return df


def tree(names):
    """cid -> (name, depth, is_leaf, children) from the national rows' indentation."""
    cids = sorted(names)
    depth = {c: len(names[c]) - len(names[c].lstrip(" ")) for c in cids}
    out = {}
    for i, c in enumerate(cids):
        below = []
        for d in cids[i + 1:]:
            if depth[d] <= depth[c]:
                break
            below.append(d)
        kids = [d for d in below if depth[d] == min(depth[k] for k in below)] if below else []
        out[c] = (names[c].strip(), depth[c], not kids, kids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    paths = [zip_path(s, r, a.fetch) for s, r in REGIONS]
    with ProcessPoolExecutor(max_workers=3) as ex:
        parts = list(ex.map(read_region, paths))

    # the country and its rows repeat in every regional file; keep the first copy
    nat = parts[0][parts[0]["GEO_LEVEL"] == "Country"]
    rest = [p[p["GEO_LEVEL"] != "Country"] for p in parts]
    df = pd.concat([nat] + rest, ignore_index=True)
    df["level"] = df["GEO_LEVEL"].map(LEVEL)
    if df["level"].isna().any():
        raise SystemExit(f"unknown GEO_LEVEL {df.loc[df['level'].isna(), 'GEO_LEVEL'].unique()}")

    # ---- the block is where we think it is ----
    names = dict(zip(nat["CHARACTERISTIC_ID"], nat["CHARACTERISTIC_NAME"]))
    for cid, want in EXPECT.items():
        if not names[cid].strip().startswith(want):
            raise SystemExit(f"characteristic {cid} is {names[cid]!r}, expected {want!r}")
    block = {c: n for c, n in names.items() if FIRST <= c <= LAST}
    t = tree(block)

    # ---- every group row is the sum of its members, nationally (random rounding: 5 per member)
    natv = dict(zip(nat["CHARACTERISTIC_ID"], nat["C1_COUNT_TOTAL"]))
    worst = 0
    for c, (name, _, leaf, kids) in t.items():
        if leaf:
            continue
        s = sum(natv[k] for k in kids)
        slack = 5 * (len(kids) + 1)
        worst = max(worst, abs(s - natv[c]) / slack)
        if abs(s - natv[c]) > slack:
            raise SystemExit(f"{c} {name}: members sum to {s:,.0f}, row says {natv[c]:,.0f}")
    single = [c for c, v in t.items() if v[2] and 395 <= c <= 717]
    multiple = list(range(719, 724))
    labels = [t[c][0] for c in single + multiple]
    if len(set(labels)) != len(labels):
        raise SystemExit("leaf labels repeat; the mapping keys on the label")
    print(f"mother tongue: {len(single)} single-response leaves, {len(multiple)} multiple rows; "
          f"group sums agree with their members (worst {worst:.2f} of the rounding slack)")
    print(f"  Canada: total {natv[393]:,.0f}, single {natv[394]:,.0f}, "
          f"multiple {natv[718]:,.0f} ({natv[718] / natv[393]:.1%})")

    # ---- each DA's subdivision, from row order ----
    geo = df.drop_duplicates("DGUID")[["DGUID", "level", "GEO_NAME", "region"]].reset_index(drop=True)
    csd, cur = [], None
    for lvl, g in zip(geo["level"], geo["DGUID"]):
        if lvl == "csd":
            cur = g
        elif lvl in ("country", "province", "cd"):
            cur = None
        csd.append(cur if lvl == "da" else (g if lvl == "csd" else None))
    geo["csd"] = csd
    nda = geo[geo["level"] == "da"]
    if nda["csd"].isna().any():
        raise SystemExit(f"{int(nda['csd'].isna().sum())} DAs before any subdivision row")
    # a DA's DGUID carries its province (2021S0512 + PR..), and so must its CSD's (2021A0005 + PR..)
    bad = nda[nda["DGUID"].str[9:11] != nda["csd"].str[9:11]]
    if len(bad):
        raise SystemExit(f"{len(bad)} DAs paired with a subdivision in another province")
    print(f"geography: {int((geo.level == 'province').sum())} provinces and territories, "
          f"{int((geo.level == 'csd').sum()):,} subdivisions, {len(nda):,} dissemination areas")

    # ---- the units file: totals, population, quality ----
    w = df.pivot_table(index="DGUID", columns="CHARACTERISTIC_ID", values="C1_COUNT_TOTAL",
                       aggfunc="first")
    tot = df[df["CHARACTERISTIC_ID"] == 393].drop_duplicates("DGUID").set_index("DGUID")
    units = geo.set_index("DGUID").join(pd.DataFrame({
        "total": w[393], "pop": w[POP], "home_total": w[724],
        "tnr_sf": tot["TNR_SF"], "flag": tot["DATA_QUALITY_FLAG"]}))
    units.index.name = "geo_id"
    da = units[units["level"] == "da"]
    print(f"  DAs with no mother-tongue total (suppressed): {int(da['total'].isna().sum())} "
          f"(population {da.loc[da['total'].isna(), 'pop'].sum():,.0f})")
    same = (da["total"] == da["home_total"]) | da["total"].isna()
    if not same.all():
        raise SystemExit(f"{int((~same).sum())} DAs whose home-language total differs from the "
                         "mother-tongue total (same population base, should match)")
    print(f"  DA totals {da['total'].sum():,.0f} against Canada's {natv[393]:,.0f} "
          f"({da['total'].sum() / natv[393]:.4f}); against DA population "
          f"{da['pop'].sum():,.0f} ({da['total'].sum() / da['pop'].sum():.4f}, the rest is "
          "institutional residents and suppressed DAs)")
    prov = units[units["level"] == "province"]
    da_prov = da.groupby(da.index.str[9:11])["total"].sum()
    for g, row in prov.iterrows():
        s = da_prov.get(g[-2:], 0)
        if abs(s - row["total"]) > 0.002 * row["total"] + 500:
            raise SystemExit(f"{row['GEO_NAME']}: DAs sum to {s:,.0f}, province row {row['total']:,.0f}")
    print("  every province's DAs sum to its own row (within 0.2% + 500)")

    # ---- the long table ----
    keep = df[df["CHARACTERISTIC_ID"].isin(single + multiple)
              & df["level"].isin(["country", "province", "csd", "da"])].copy()
    keep = keep[keep["C1_COUNT_TOTAL"].fillna(0) > 0]
    keep["source_category"] = keep["CHARACTERISTIC_ID"].map(lambda c: t[c][0])
    keep["kind"] = keep["CHARACTERISTIC_ID"].map(lambda c: "multiple" if c >= 719 else "single")
    keep = keep.merge(geo[["DGUID", "csd"]], on="DGUID", how="left")
    out = keep.rename(columns={"DGUID": "geo_id", "level": "geo_level", "GEO_NAME": "geo_name",
                               "CHARACTERISTIC_ID": "cid", "C1_COUNT_TOTAL": "count"})[
        ["geo_id", "geo_level", "geo_name", "csd", "cid", "kind", "source_category", "count"]]
    out["count"] = out["count"].astype(int)

    # per DA, leaves + multiple against the total
    d = out[out["geo_level"] == "da"].groupby("geo_id")["count"].sum()
    diff = (d.reindex(da.index).fillna(0) - da["total"].fillna(0)).abs()
    print(f"  per DA, single leaves + multiple rows minus the total: median {diff.median():.0f}, "
          f"p99 {diff.quantile(.99):.0f}, max {diff.max():.0f} (random rounding)")
    # national: DAs against the country row, per category
    dsum = out[out["geo_level"] == "da"].groupby("cid")["count"].sum()
    ratio = (dsum / pd.Series(natv)).reindex(single + multiple)
    big = [c for c in single + multiple if natv[c] >= 5000]
    r = ratio[big]
    print(f"  DA sums / Canada row, {len(big)} categories of 5,000+: min {r.min():.3f} "
          f"({t[r.idxmin()][0]}), median {r.median():.3f}, max {r.max():.3f} ({t[r.idxmax()][0]})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    units.reset_index().rename(columns={"GEO_NAME": "geo_name", "level": "geo_level"}).to_csv(
        OUT_UNITS, index=False)
    print(f"wrote {OUT} ({len(out):,} rows) and {OUT_UNITS} ({len(units):,} units)")


if __name__ == "__main__":
    sys.exit(main())
