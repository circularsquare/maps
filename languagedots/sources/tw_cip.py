"""Taiwan: registered indigenous population by people (族別), by township, end of October 2020.

    python sources/tw_cip.py --fetch    download into data/raw/tw/, then normalise
    python sources/tw_cip.py            normalise what is on disk

Used only to share each township's census 原住民族語 (indigenous languages, one answer in the
2020 census, sources/tw_census.py) across the indigenous peoples registered there (ask 012,
Anita 2026-10-05). The census counts stay as they are; this file only decides which language.

Source: Council of Indigenous Peoples (原住民族委員會), 原住民人口數統計資料 (monthly), the
October 2020 release (published 2020-11-11): 10910台閩縣市鄉鎮市區原住民族人口-按性別族別.xls,
the household register's RCRPC1F0 月統計報表之７, 臺閩地區各縣市鄉鎮市區現住原住民人口數按
性別、原住民身分及族別分. End of October 2020 is the month-end nearest the census reference date
(8 November 2020). Listing page:
https://www.cip.gov.tw/zh-tw/news/data-list/940F9579765AC6A0/2D9680BFECBE80B676534BCF44D8EC98-info.html

The sheet holds three blocks (不分平地山地 = all, 平地原住民 = lowland status, 山地原住民 =
mountain status), each with a 計/男/女 row per area; only the first block's 計 rows are read.
Columns: 總計, the 16 recognised peoples, 尚未申報 (people registered as indigenous whose people
is not yet declared). All ages, registered residents.

Writes data/normalized/tw_peoples.csv: geo_id "<county>|<township>" (the census's own keys),
people (the table's label), count.

CHECKS, all asserted:
  * the national row's total is 575,967 and the peoples plus 尚未申報 sum to 總計 on every row;
  * townships sum to their county row in every column, counties to the national row;
  * 368 townships, joined both ways to the census townships (data/normalized/tw.csv), exactly.
"""
import csv
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tw" / "cip_10910_town_people.xls"
CENSUS = ROOT / "data" / "normalized" / "tw.csv"
OUT = ROOT / "data" / "normalized" / "tw_peoples.csv"
URL = ("https://www.cip.gov.tw/data/news/document/202011/1605057200215-2.xls?s=3F2C0B628540808B"
       "&c=9C13926BD891316C380DBDB607C91D2F&fn=57F3C98579E2DDEF870D865A60846592C60ED141A3AEC4633D3CD6"
       "06F1172413BD7076C11633C29B340A5E8F0999A40AD0636733C6861689")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
NATIONAL = 575967
N_TOWNS = 368
PROVINCES = {"臺灣省", "福建省"}


def clean(v):
    return "" if v is None else str(v).replace("　", "").replace(" ", "").strip()


def main():
    if "--fetch" in sys.argv:
        req = urllib.request.Request(URL, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if data[:4] != b"\xd0\xcf\x11\xe0":
            raise SystemExit(f"not an xls ({len(data)} bytes)")
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(data)
    import pandas as pd
    df = pd.read_excel(RAW, header=None)
    head = df.index[df[0].astype(str).str.strip() == "不分平地山地"][0]
    end = df.index[df[0].astype(str).str.strip() == "平地原住民"][0]
    cols = [clean(c) for c in df.iloc[head]]
    assert cols[1:4] == ["區域別", "性別", "總計"], cols
    peoples = [c for c in cols[4:] if c]
    assert len(peoples) == 17 and peoples[-1] == "尚未申報", peoples
    block = df.iloc[head + 1:end]
    block = block[block[2].map(clean) == "計"]

    census_towns = {}
    with open(CENSUS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["geo_level"] == "town":
                c, t = r["geo_id"].split("|")
                census_towns.setdefault(c, set()).add(t)
    counties = set(census_towns)

    national, county_rows, towns = None, {}, {}
    county = None
    for _, r in block.iterrows():
        name = clean(r[1])
        vals = [int(r[i]) for i in range(3, 3 + 1 + len(peoples))]
        if sum(vals[1:]) != vals[0]:
            raise SystemExit(f"{name}: peoples sum {sum(vals[1:])} != total {vals[0]}")
        if name == "總計":
            national = vals
        elif name in PROVINCES:   # 臺灣省, 福建省: subtotals over some counties, skipped
            county = None
        elif name in counties:
            county = name
            county_rows[county] = vals
        else:
            if county is None or name not in census_towns[county]:
                raise SystemExit(f"township {county}|{name} not in the census")
            towns[f"{county}|{name}"] = vals
    assert national and national[0] == NATIONAL, national
    assert set(county_rows) == counties, set(county_rows) ^ counties
    for c, cv in county_rows.items():
        s = [sum(v[i] for k, v in towns.items() if k.startswith(c + "|")) for i in range(len(cv))]
        assert s == cv, (c, s, cv)
    assert [sum(cv[i] for cv in county_rows.values()) for i in range(len(national))] == national
    want = {f"{c}|{t}" for c, ts in census_towns.items() for t in ts}
    assert len(towns) == N_TOWNS and set(towns) == want, set(towns) ^ want

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geo_id", "people", "count"])
        for k, v in towns.items():
            for p, n in zip(peoples, v[1:]):
                w.writerow([k, p, n])
    print(f"{len(towns)} townships, {national[0]:,} registered indigenous people; "
          + ", ".join(f"{p} {n:,}" for p, n in zip(peoples, national[1:])))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
