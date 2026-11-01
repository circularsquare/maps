"""Download Census 2011 PCA TV workbooks for the districts level4.py needs.

PCA TV ("Primary census abstract at town, village and ward level") is one
workbook per district on censusindia.gov.in/nada, catalogue idno
PC11_PCA-TV-<ss><dd> (state code, district number within the state). The
catalogue id and the file's download id are not derivable from the codes, so
this searches the catalogue once, then reads each study's related-materials
page for its .xlsx link.

Saves maps/data/asia1m/india/wards/pca_tv/pca_tv_<pc11_d_id>.xlsx and skips
files already there. Each download is checked: it must open and its district
column must be the district asked for.

The site serves an incomplete certificate chain, so verification is off for
this host only (the bytes are checked instead).

    python fetch_pca_tv.py              districts with a sub-district over level4's SPLIT_OVER
    python fetch_pca_tv.py 157 164      just these
"""
from __future__ import annotations

import re
import sys
import time

import pandas as pd
import requests
import urllib3

import level4

urllib3.disable_warnings()
NADA = "https://censusindia.gov.in/nada/index.php"
DISTRICTS = level4.REPO / "religiondots" / "data" / "geo" / "in" / "shrug-subdistrict-pc11.parquet"
S = requests.Session()
S.verify = False
S.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def get(url, **kw):
    for attempt in range(4):
        try:
            r = S.get(url, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 3:
                raise
            print("   retry", url, e)
            time.sleep(5 * (attempt + 1))


def catalogue():
    """idno -> catalogue id for every PCA TV study."""
    out, page = {}, 1
    while True:
        j = get(f"{NADA}/api/catalog/search",
                params={"sk": '"town, village and ward level"', "ps": 1000, "page": page}).json()
        rows = j["result"]["rows"]
        for r in rows:
            if r["idno"].startswith("PC11_PCA-TV-"):
                out[r["idno"]] = r["id"]
        if len(rows) < 1000:
            return out
        page += 1


def wanted():
    p = pd.read_csv(level4.HELPER / "data" / "india" / "population.csv", dtype={"code": str})
    big = p[(p.level == 3) & (p.year == level4.SPLIT_YEAR) & (p["pop"] > level4.SPLIT_OVER)]
    return sorted(set(big.code.str[:3]))


def idno_of():
    """pc11_d_id -> PC11_PCA-TV-<ss><dd>; dd counts the state's districts in code order."""
    d = pd.read_parquet(DISTRICTS, columns=["pc11_s_id", "pc11_d_id"]).drop_duplicates()
    d = d[d.pc11_d_id != "000"].sort_values("pc11_d_id")   # "000" is a placeholder
    d["n"] = d.groupby("pc11_s_id").cumcount() + 1
    return {r.pc11_d_id: f"PC11_PCA-TV-{r.pc11_s_id}{r.n:02d}" for r in d.itertuples()}


def check(path, d):
    df = pd.read_excel(path, dtype=str)
    got = set(df["District"].str.zfill(3)) - {"000"}
    if got != {d}:
        raise ValueError(f"{path.name}: district column {sorted(got)[:5]}, wanted {d}")
    return len(df)


def main(ds):
    level4.PCA.mkdir(parents=True, exist_ok=True)
    todo = [d for d in ds if not (level4.PCA / f"pca_tv_{d}.xlsx").exists()]
    print(f"{len(ds)} districts, {len(todo)} to download")
    if not todo:
        return
    cat, ids = catalogue(), idno_of()
    print(f"catalogue: {len(cat)} PCA TV studies")
    failed = []
    for k, d in enumerate(todo, 1):
        idno = ids.get(d)
        cid = cat.get(idno)
        if not cid:
            print(f"  {d}: {idno} not in the catalogue")
            failed.append(d)
            continue
        page = get(f"{NADA}/catalog/{cid}/related-materials").text
        links = re.findall(rf'href="({re.escape(NADA)}/catalog/{cid}/download/\d+)"[^>]*'
                           r'data-filename="([^"]+\.xlsx?)"', page)
        if not links:
            print(f"  {d}: no workbook on catalogue/{cid}")
            failed.append(d)
            continue
        url, name = links[0]
        out = level4.PCA / f"pca_tv_{d}.xlsx"
        tmp = out.with_suffix(".part")
        tmp.write_bytes(get(url).content)
        try:
            tmp_x = tmp.with_suffix(".check.xlsx")
            tmp.replace(tmp_x)
            n = check(tmp_x, d)
            tmp_x.replace(out)
        except Exception as e:
            print(f"  {d}: {name} failed the check: {e}")
            failed.append(d)
            continue
        print(f"  [{k}/{len(todo)}] {d} {idno} {name}: {n:,} rows")
        time.sleep(1)
    if failed:
        print("FAILED:", ",".join(failed))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:] or wanted())
