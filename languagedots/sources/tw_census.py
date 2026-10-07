"""Taiwan: 2020 Population and Housing Census, language, by the 368 townships and districts.

    python sources/tw_census.py --fetch    download into data/raw/tw/, then normalise
    python sources/tw_census.py            normalise what is on disk

Source: DGBAS (Directorate-General of Budget, Accounting and Statistics), 109年人口及住宅普查
總報告統計結果表, 縣市別報告統計表: one page per county or city on www.stat.gov.tw, each with ~46
tables as XLSX. Two of them are language tables of the 6+ resident nationals, by township:

    t006  表6 6歲以上本國籍常住人口使用語言情形      language used now: main (one answer) and
                                                      secondary language
    t007  表7 6歲以上本國籍常住人口兒時最早學會語言情形  language learned earliest in childhood
                                                      (one answer)

Each table gives the township's 6+ resident nationals as a count and the languages as shares
per hundred, to one decimal. Counts here are share x population, so a township count carries
up to +-0.05% of its population of rounding (Taipei's 100,000-people districts: +-50).

The ws.dgbas.gov.tw certificate chain does not verify from here, so the fetch skips
verification (a public statistics file; nothing is sent).

Writes data/normalized/tw.csv: geo_level=town (and county, the table's own total row),
geo_id = "<county name>|<township name>" in the table's own characters, question = earliest
(t007, drawn) or main (t006, recorded for comparison), one row per category.

CHECKS, all asserted:
  * each file's title names its county and its table;
  * the townships' populations sum to the county row, in both tables, and the two tables agree
    on every township's population;
  * every township's shares sum to 100 within rounding (+-0.6 over six one-decimal cells);
  * the county rows of t006 against SEGIS's county dataset of the same census (MOI's copy, also
    shares to one decimal, so a transcription check, not an independent one): population equal,
    each main-language share equal;
  * 368 townships, and the national 6+ resident nationals equal SEGIS's sum.
"""
import csv
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tw"
OUT = ROOT / "data" / "normalized" / "tw.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# the county pages' `s` ids, from the 縣市別報告統計表 listing
# https://www.stat.gov.tw/News_Content.aspx?Create=1&n=2755&state=1327FD6AD8DCDA52&s=230879&ccms_cs=1&sms=11065
COUNTIES = {
    "新北市": 230886, "臺北市": 230887, "桃園市": 230883, "基隆市": 230888, "新竹市": 230889,
    "宜蘭縣": 230890, "新竹縣": 230892, "臺中市": 230893, "苗栗縣": 230894, "彰化縣": 230895,
    "南投縣": 230896, "雲林縣": 230897, "臺南市": 230898, "高雄市": 230899, "嘉義市": 230900,
    "嘉義縣": 230901, "屏東縣": 230902, "澎湖縣": 230903, "臺東縣": 230904, "花蓮縣": 230905,
    "金門縣": 230906, "連江縣": 230907,
}
FILE_URL = "https://ws.dgbas.gov.tw/001/Upload/463/relfile/11065/{s}/{t}.xlsx"
TABLES = {"t006": ("main", "使用語言情形"), "t007": ("earliest", "兒時最早學會語言情形")}
# SEGIS (MOI Social Economic Data Service), 109年6歲以上本國籍常住人口使用語言情形普查統計, by county
SEGIS = ("https://segisws.moi.gov.tw/STATWSSTData/OpenService.asmx/GetAdminSTDataForOpenCode?"
         "oCode=88641597DE2A496B0BD262FF6FF7387358872164398953095540FAA96511A9BA9894FCEE7AC686A3D8BBECC4B34231B0")
SEGIS_MAIN = {"FLD02": "國語", "FLD03": "閩南語", "FLD04": "客語", "FLD05": "原住民族語", "FLD06": "其他"}
N_TOWNS = 368


def get(url, insecure=False):
    ctx = ssl._create_unverified_context() if insecure else None
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        return r.read()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for county, s in COUNTIES.items():
        for t in TABLES:
            p = RAW / f"{s}_{t}.xlsx"
            if p.exists() and p.stat().st_size > 5000:
                continue
            data = get(FILE_URL.format(s=s, t=t), insecure=True)
            if data[:2] != b"PK":
                raise SystemExit(f"{county} {t}: not an xlsx ({len(data)} bytes)")
            p.write_bytes(data)
            print(f"  {county} {t}: {len(data):,} bytes")
    (RAW / "segis_county_main.json").write_bytes(get(SEGIS))


def clean(v):
    return "" if v is None else str(v).replace("　", "").replace(" ", "").replace("\n", "").strip()


def read_table(path, county, t):
    import openpyxl
    q, title_key = TABLES[t]
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows, cats, title_ok = [], None, False
    for ws in wb.worksheets:
        in_town = False
        for r in ws.iter_rows(values_only=True):
            cells = [clean(v) for v in r]
            joined = "".join(cells)
            if county in joined and title_key in joined:
                title_ok = True
            if cats is None and "國語" in cells and "閩南語" in cells:
                # the first header row of language names; for t006 the first five are `main`
                idx = [i for i, c in enumerate(cells) if c]
                names = [(i, cells[i]) for i in idx if i >= 2]
                if t == "t006":
                    names = names[:5]
                cats = names
            if cells[1:2] == ["按鄉鎮市區別分"] or joined.startswith("按鄉鎮市區別分"):
                in_town = True
                continue
            if joined.startswith("按"):
                in_town = False
                continue
            if in_town and cells[1] and isinstance(r[2], (int, float)):
                rows.append((cells[1], int(r[2]), {name: float(r[i] or 0) for i, name in cats}))
    if not title_ok:
        raise SystemExit(f"{path.name}: title does not name {county} / {title_key}")
    if not rows or rows[0][0] != county:
        raise SystemExit(f"{path.name}: first township-section row is {rows[:1]}, not {county}")
    return q, cats, rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    segis = json.loads((RAW / "segis_county_main.json").read_text(encoding="utf-8"))
    rows_key = next(k for k, v in segis.items() if isinstance(v, list) and v and "COUNTY" in v[0])
    seg = {r["COUNTY"]: r for r in segis[rows_key]}
    if set(seg) != set(COUNTIES):
        raise SystemExit(f"SEGIS counties differ: {sorted(set(seg) ^ set(COUNTIES))}")

    out, towns, pops, worst_seg = [], set(), {}, 0.0
    for county, s in COUNTIES.items():
        for t in TABLES:
            q, cats, rows = read_table(RAW / f"{s}_{t}.xlsx", county, t)
            head, town_rows = rows[0], rows[1:]
            if sum(p for _, p, _ in town_rows) != head[1]:
                raise SystemExit(f"{county} {t}: townships sum to {sum(p for _, p, _ in town_rows)}, "
                                 f"county row {head[1]}")
            for name, pop, shares in rows:
                tot = sum(shares.values())
                if abs(tot - 100) > 0.6:
                    raise SystemExit(f"{county} {name} {t}: shares sum to {tot}")
                key = f"{county}|{name}"
                if name != county:
                    towns.add(key)
                    if pops.setdefault(key, pop) != pop:
                        raise SystemExit(f"{key}: t006 and t007 populations differ")
                level = "county" if name == county else "town"
                for cat, sh in shares.items():
                    out.append({"geo_level": level, "geo_id": key if level == "town" else county,
                                "geo_name": name, "question": q, "source_category": cat,
                                "share": sh, "pop6": pop, "count": round(pop * sh / 100, 1)})
            if t == "t006":
                sr = seg[county]
                if int(float(sr["FLD01"])) != head[1]:
                    raise SystemExit(f"{county}: SEGIS 6+ {sr['FLD01']} vs table {head[1]}")
                for fld, cat in SEGIS_MAIN.items():
                    d = abs(float(sr[fld]) - head[2][cat])
                    worst_seg = max(worst_seg, d)
                    if d > 0.05:
                        raise SystemExit(f"{county} {cat}: SEGIS {sr[fld]} vs table {head[2][cat]}")
    if len(towns) != N_TOWNS:
        raise SystemExit(f"{len(towns)} townships, expected {N_TOWNS}")
    nat = sum(pops.values())
    seg_nat = sum(int(float(r["FLD01"])) for r in seg.values())
    if nat != seg_nat:
        raise SystemExit(f"townships hold {nat:,}, SEGIS {seg_nat:,}")
    print(f"  {len(towns)} townships, {nat:,} resident nationals aged 6+ (= SEGIS); "
          f"SEGIS county main-language shares differ from the tables by at most {worst_seg:.2f}")
    for q in ("earliest", "main"):
        tot = {}
        for r in out:
            if r["question"] == q and r["geo_level"] == "town":
                tot[r["source_category"]] = tot.get(r["source_category"], 0) + r["count"]
        print(f"  {q}: " + ", ".join(f"{k} {v:,.0f} ({100 * v / nat:.1f}%)" for k, v in tot.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"  wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
