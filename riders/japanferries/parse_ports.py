"""Parse 港湾統計（年報）第2表 船舶乗降人員表 into one row per port.

Reads the major-port (甲種港湾, 第2部) and minor-port (乙種港湾, 第3部)
sheets from raw/ and writes data/ports_<year>.csv. Each sheet is laid out in
side-by-side column blocks, each headed
    都道府県 | (grade mark) | 港湾 | | 種別 | 計 | 乗込人員 | 上陸人員
and read top to bottom, block by block. 種別 is 計 / 外 (international) /
内 (domestic); minor ports mostly carry a 内 row only.
"""
import csv
import os
import sys
from collections import defaultdict

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
DATA = os.path.join(HERE, "data")
YEAR = 2024
FILES = [
    ("major", f"port{YEAR}_koushu_t2.xlsx"),
    ("minor", f"port{YEAR}_otsushu_t2.xlsx"),
]
KINDS = ("計", "外", "内")


def clean(v):
    if v is None:
        return ""
    return str(v).replace("　", "").replace(" ", "").strip()


def read_records(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = [w for w in wb.worksheets if not w.title.startswith("タイトル")][0]
    rows = [list(r) + [None] * 8 for r in ws.iter_rows(values_only=True)]
    starts = sorted({i for r in rows for i, v in enumerate(r) if clean(v) == "都道府県"})
    records = []
    pref = port = mark = ""
    for s in starts:
        for r in rows:
            c_pref, c_mark, c_port, c_port2, c_kind = (clean(v) for v in r[s:s + 5])
            nums = r[s + 5:s + 8]
            if c_pref == "都道府県":
                continue
            if c_pref:
                pref, port, mark = c_pref, "", ""
            if c_port or c_port2:
                port, mark = c_port + c_port2, c_mark
            if c_kind not in KINDS:
                continue
            if not all(isinstance(x, (int, float)) for x in nums):
                continue
            total, board, land = (int(x) for x in nums)
            if total != board + land:
                print(f"  ! {pref} {port} {c_kind}: {total} != {board} + {land}")
            records.append((pref, mark, port, c_kind, total, board, land))
    return records


def main():
    os.makedirs(DATA, exist_ok=True)
    out = []
    for table, name in FILES:
        records = read_records(os.path.join(RAW, name))
        ports = defaultdict(dict)
        pref_totals = defaultdict(dict)
        grand = {}
        for pref, mark, port, kind, total, board, land in records:
            if not port:
                if pref == "総計":
                    grand[kind] = total
                continue
            if port == "合計":
                pref_totals[pref][kind] = total
                continue
            ports[(pref, port)][kind] = (total, board, land)
            ports[(pref, port)]["mark"] = mark

        # checks: ports sum to prefecture totals and to the grand total
        dom_sum = 0
        by_pref = defaultdict(int)
        for (pref, port), k in ports.items():
            dom = k.get("内") or (k["計"] if "外" not in k and "計" in k else (0, 0, 0))
            dom_sum += dom[0]
            by_pref[pref] += dom[0]
        print(f"{table}: {len(ports)} ports, domestic sum {dom_sum:,} vs 総計 内 {grand.get('内', 0):,}")
        for pref, t in pref_totals.items():
            if "内" in t and t["内"] != by_pref[pref]:
                print(f"  ! {pref}: ports {by_pref[pref]:,} vs 合計 {t['内']:,}")

        for (pref, port), k in ports.items():
            dom = k.get("内") or (k["計"] if "外" not in k and "計" in k else (0, 0, 0))
            intl = k.get("外", (0, 0, 0))
            out.append({
                "table": table, "prefecture": pref, "mark": k["mark"], "port": port,
                "dom_total": dom[0], "dom_boarding": dom[1], "dom_landing": dom[2],
                "intl_total": intl[0], "intl_boarding": intl[1], "intl_landing": intl[2],
            })

    path = os.path.join(DATA, f"ports_{YEAR}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}: {len(out)} ports")

    print("\nbusiest ports, domestic boarding + landing:")
    for row in sorted(out, key=lambda r: -r["dom_total"])[:40]:
        print(f"  {row['dom_total']:>10,}  {row['prefecture']} {row['port']} ({row['table']})")


if __name__ == "__main__":
    main()
