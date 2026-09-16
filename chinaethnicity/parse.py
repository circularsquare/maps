"""Read each province's 2020 table 1-4 into one line per table row, checked by arithmetic.

Every table is shaped the same way underneath its formatting: a province total, then
prefectures, each followed by its counties (some provinces also have county-level units
directly under the province, such as Hubei's Xiantao or Hainan's Danzhou). The shape is
recovered from the row labels' indentation where the table has any, and from the numbers
themselves where it does not (Ningxia's PDF): a row is a prefecture when the rows right
after it add up to it exactly, in all 59 columns.

Either way nothing is trusted until it adds up:
  * every prefecture row equals the sum of its own counties, in every column;
  * the top-level rows add up to the province total;
  * the province total equals the province's row in the NATIONAL table, in every column —
    so a province's own yearbook is checked against a different publisher's copy;
  * for every row and column, the total equals male plus female.

Writes data/work/table_2020.csv:
    prov, seq, name, kind (agg | leaf | zero), parent (the name of the aggregate row it
    sits under, empty at top level), total, then one column per nationality key.

Usage:
    python parse.py              # every province fetch.py has downloaded
    python parse.py henan hubei
"""
import csv
import glob
import io
import os
import re
import subprocess
import sys
import zipfile

import numpy as np

from common import CODE_OF, GROUPS, KEYS, NCAT, PROVINCES, RAW, WORK, norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UNRAR = [r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files\WinRAR\UnRAR.exe"]
OUT = os.path.join(WORK, "table_2020.csv")

# province key -> (how to open it, file name from fetch.py)
TABLES = {
    "beijing": ("xls", "beijing_A0106.xls"),
    "neimenggu": ("xls", "neimenggu_A0104.xls"),
    "jilin": ("xls", "jilin_A0104.xls"),
    "heilongjiang": ("xls", "heilongjiang_A0104.xls"),
    "shanghai": ("xls", "shanghai_ANJ-1-04.xls"),
    "jiangsu": ("xls", "jiangsu_A0104.xls"),
    "zhejiang": ("xls", "zhejiang_A0104.xls"),
    "fujian": ("xls", "fujian_a0104.xls"),
    "shandong": ("xls", "shandong_A0104.xls"),
    "henan": ("xls", "henan_A0104.xls"),
    "hubei": ("zip", "hubei_2020.zip"),
    "guangxi": ("xls", "guangxi_A1-04.xls"),
    "hainan": ("xls", "hainan_A0104.xls"),
    "yunnan": ("rar", "yunnan_2020.rar"),
    "qinghai": ("xls", "qinghai_A0104.xls"),
    "ningxia": ("pdf", "ningxia_rkpc2021.pdf"),
}
NATIONAL = "national_A0104.xls"

# Ningxia's yearbook PDF: table 1-4 and its 19 continuation pages, 0-based page numbers.
# The first page carries the unit total and Han; each continuation page three groups.
NINGXIA_PAGES = range(29, 49)

# the member of a whole-yearbook archive that is table 1-4 itself, not its 1-4a/b/c
# (city / town / rural) variants and not table 1-40-something
MEMBER = re.compile(r"^a?1-4(?![0-9a-z])")


# ---------------------------------------------------------------------------------------
# reading cells

def num(v):
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(round(v))
    s = norm(v).replace(",", "")
    if s in ("", "-", "—", "–"):
        return 0
    return int(float(s))


def is_num(v):
    if isinstance(v, (int, float)):
        return True
    return bool(re.fullmatch(r"\d+(\.0+)?", norm(v).replace(",", "")))


def xlsx_rows(src):
    import openpyxl
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    return [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]


def sheet_rows(kind, fn):
    path = os.path.join(RAW, fn)
    if kind == "xls":
        import xlrd
        sh = xlrd.open_workbook(path).sheet_by_index(0)
        return [sh.row_values(r) for r in range(sh.nrows)]
    if kind == "zip":
        z = zipfile.ZipFile(path)
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:          # not flagged UTF-8: GBK, read as cp437
                name = name.encode("cp437").decode("gbk", "replace")
            base = os.path.basename(name)
            if (MEMBER.match(base) and "各地区分性别、民族的人口" in base
                    and "（" not in base and base.endswith(".xlsx")):
                print(f"    member {name}")
                return xlsx_rows(io.BytesIO(z.read(info)))
        raise SystemExit(f"{fn}: no table 1-4 member")
    if kind == "rar":
        out = os.path.join(RAW, os.path.splitext(fn)[0])
        if not glob.glob(os.path.join(out, "**", "*.xlsx"), recursive=True):
            os.makedirs(out, exist_ok=True)
            # Not Windows' bsdtar: it reads the archive but cannot decode the Chinese member
            # names and skips every file ("empty or unreadable filename").
            tool = next((t for t in UNRAR if os.path.exists(t)), None)
            if tool is None:
                raise SystemExit(f"{fn}: needs 7-Zip or WinRAR to unpack; looked for {UNRAR}")
            args = ([tool, "x", "-y", f"-o{out}", path] if tool.endswith("7z.exe")
                    else [tool, "x", "-y", path, out + os.sep])
            subprocess.run(args, check=True, capture_output=True)
        for p in sorted(glob.glob(os.path.join(out, "**", "*.xlsx"), recursive=True)):
            base = os.path.basename(p)
            if MEMBER.match(base) and "各地区分性别、民族的人口" in base and "（" not in base:
                print(f"    member {os.path.relpath(p, out)}")
                return xlsx_rows(p)
        raise SystemExit(f"{fn}: no table 1-4 member")
    raise ValueError(kind)


def find_header(rows, where):
    for r, row in enumerate(rows[:12]):
        if any(norm(c) == "汉族" for c in row[1:]):
            return r
    raise SystemExit(f"{where}: no header row naming 汉族")


def check_header(labels, where):
    """The 56 named nationalities must sit in census order. The two residual columns are
    labelled several ways between yearbooks, so they are shown rather than matched."""
    bad = [(i, have, GROUPS[i - 1][2]) for i, have in enumerate(labels)
           if 1 <= i <= 56 and have != GROUPS[i - 1][2]]
    if bad:
        raise SystemExit(f"{where}: header is not in census order, e.g. {bad[:4]}")
    return labels[57:59]


def lines_from_sheet(rows, where):
    header = find_header(rows, where)
    width = 1 + 3 * NCAT
    labels = [norm(rows[header][1 + 3 * g]) if 1 + 3 * g < len(rows[header]) else ""
              for g in range(NCAT)]
    tail = check_header(labels, where)
    out = []
    for row in rows[header + 1:]:
        row = list(row) + [None] * (width - len(row))
        label = row[0]
        name = norm(label)
        if not name or name in ("甲", "乙") or not is_num(row[1]):
            continue
        raw = str(label)
        out.append(dict(
            name=name,
            indent=len(raw) - len(raw.lstrip()),
            v=np.array([num(row[1 + 3 * g]) for g in range(NCAT)], dtype=np.int64),
            m=np.array([num(row[2 + 3 * g]) for g in range(NCAT)], dtype=np.int64),
            f=np.array([num(row[3 + 3 * g]) for g in range(NCAT)], dtype=np.int64)))
    return out, tail


# ---------------------------------------------------------------------------------------
# Ningxia's PDF

def lines_from_ningxia(fn):
    import fitz

    doc = fitz.open(os.path.join(RAW, fn))
    pages = []
    for k, pno in enumerate(NINGXIA_PAGES):
        cols = [0, 1] if k == 0 else [2 + 3 * (k - 1) + j for j in range(3)]
        # Group words into printed lines by vertical gap. Rows are ~20pt apart, and a label
        # can sit a point or two off its own numbers, so fixed-width bands split rows.
        words = sorted(doc[pno].get_text("words"), key=lambda w: (w[1] + w[3]) / 2)
        lines, cy = {}, None
        for w in words:
            y = (w[1] + w[3]) / 2
            if cy is None or y - cy > 5:
                cy = y
            lines.setdefault(cy, []).append(w)
        bands = sorted(lines)
        # header: the band holding the 男 / 女 column heads
        hb = next(b for b in bands if sum(w[4] == "男" for w in lines[b]) == len(cols))
        heads = sorted((w for w in lines[hb] if w[4] in ("合计", "小计", "男", "女")),
                       key=lambda w: w[0])
        if len(heads) != 3 * len(cols):
            raise SystemExit(f"ningxia page {pno + 1}: {len(heads)} column heads")
        centres = np.array([(w[0] + w[2]) / 2 for w in heads])
        text_above = norm("".join(w[4] for b in bands if b < hb for w in lines[b]))
        for c in cols:
            if 1 <= c <= 56 and GROUPS[c - 1][2] not in text_above:
                raise SystemExit(f"ningxia page {pno + 1}: expected {GROUPS[c - 1][2]}")

        rows = []
        for b in bands:
            if b <= hb:
                continue
            ws = sorted(lines[b], key=lambda w: w[0])
            label = norm("".join(w[4] for w in ws if not is_num(w[4])))
            nums = [w for w in ws if is_num(w[4])]
            if not label or not nums:
                continue                                  # page number, footers
            cells = np.zeros(3 * len(cols), dtype=np.int64)
            seen = set()
            for w in nums:
                j = int(np.argmin(np.abs(centres - (w[0] + w[2]) / 2)))
                if j in seen:
                    raise SystemExit(f"ningxia page {pno + 1} {label}: two numbers in one column")
                seen.add(j)
                cells[j] = int(w[4])
            rows.append((label, cells))
        pages.append((cols, rows))

    # The first page carries the unit totals, which are never blank, so it defines the rows.
    # A continuation page leaves a row out entirely when none of its three groups live
    # there; that row is zeros for those columns. A label a page has and the first page
    # does not is an error, and so is a label printed twice.
    names = [r[0] for r in pages[0][1]]
    index = {n: i for i, n in enumerate(names)}
    if len(index) != len(names):
        raise SystemExit("ningxia: a row label repeats on the first page")
    V = np.zeros((len(names), NCAT), dtype=np.int64)
    M, Fm = V.copy(), V.copy()
    for cols, rows in pages:
        seen = set()
        for label, cells in rows:
            if label not in index or label in seen:
                raise SystemExit(f"ningxia columns {cols}: unexpected row {label!r}")
            seen.add(label)
            i = index[label]
            for j, c in enumerate(cols):
                V[i, c], M[i, c], Fm[i, c] = cells[3 * j], cells[3 * j + 1], cells[3 * j + 2]
    out = [dict(name=n, indent=0, v=V[i], m=M[i], f=Fm[i]) for i, n in enumerate(names)]
    return out, ("(pdf)", "(pdf)")


# ---------------------------------------------------------------------------------------
# structure

def build_tree(lines, where):
    """Label each line total / agg / leaf / zero and give it a parent. Returns the total
    vector and the remaining lines."""
    for l in lines:
        bad = np.nonzero(l["v"] != l["m"] + l["f"])[0]
        if len(bad):
            raise SystemExit(f"{where} {l['name']}: total != male + female in "
                             f"{[('total' if c == 0 else KEYS[c - 1]) for c in bad[:4]]}")

    vmax = max(l["v"][0] for l in lines)
    first = next(i for i, l in enumerate(lines) if l["v"][0] == vmax)
    total = lines[first]["v"]
    rest = lines[first + 1:]
    while rest and np.array_equal(rest[0]["v"], total):   # Hubei prints 合计 then 湖北
        rest = rest[1:]

    indents = {l["indent"] for l in rest}
    if len(indents) > 1:
        how = "indentation"
        stack = []
        for i, l in enumerate(rest):
            while stack and l["indent"] <= rest[stack[-1]]["indent"]:
                stack.pop()
            l["parent"] = stack[-1] if stack else None
            nxt = rest[i + 1]["indent"] if i + 1 < len(rest) else -1
            if nxt > l["indent"]:
                l["kind"] = "agg"
                stack.append(i)
            else:
                l["kind"] = "leaf"
    else:
        how = "sums"
        i = 0
        while i < len(rest):
            target, acc, found = rest[i]["v"], np.zeros(NCAT, dtype=np.int64), None
            j = i + 1
            while target[0] > 0 and j < len(rest) and acc[0] + rest[j]["v"][0] <= target[0]:
                acc = acc + rest[j]["v"]
                if np.array_equal(acc, target):
                    found = j
                    break
                j += 1
            rest[i]["parent"] = None
            if found is not None:
                rest[i]["kind"] = "agg"
                for k in range(i + 1, found + 1):
                    rest[k]["parent"], rest[k]["kind"] = i, "leaf"
                i = found + 1
            else:
                rest[i]["kind"] = "leaf"
                i += 1

    for l in rest:
        if l["kind"] == "leaf" and l["v"][0] == 0:
            l["kind"] = "zero"

    errors = 0
    for a, l in enumerate(rest):
        if l["kind"] != "agg":
            continue
        s = sum((c["v"] for c in rest if c["parent"] == a), np.zeros(NCAT, dtype=np.int64))
        if not np.array_equal(s, l["v"]):
            d = np.nonzero(s != l["v"])[0]
            print(f"  !! {where} {l['name']}: children do not add up in "
                  f"{len(d)} columns (total {s[0]:,} vs {l['v'][0]:,})")
            errors += 1
    top = sum((l["v"] for l in rest if l["parent"] is None), np.zeros(NCAT, dtype=np.int64))
    if not np.array_equal(top, total):
        print(f"  !! {where}: top-level rows add to {top[0]:,}, province total {total[0]:,}")
        errors += 1
    return total, rest, how, errors


def national_table():
    import xlrd
    sh = xlrd.open_workbook(os.path.join(RAW, NATIONAL)).sheet_by_index(0)
    rows = [sh.row_values(r) for r in range(sh.nrows)]
    lines, _ = lines_from_sheet(rows, "national")
    by_name = {PROVINCES[c][2]: c for c in PROVINCES}
    out = {}
    for l in lines:
        if l["name"] in by_name:
            out[by_name[l["name"]]] = l["v"]
    if len(out) != 31:
        raise SystemExit(f"national table: found {len(out)} provinces")
    return out


def main():
    keys = sys.argv[1:] or [k for k, (_, fn) in TABLES.items()
                            if os.path.exists(os.path.join(RAW, fn))]
    nat = national_table()
    print(f"national table: 31 provinces, {sum(v[0] for v in nat.values()):,} people")

    out_rows, failed = [], []
    for key in keys:
        code = CODE_OF[key]
        kind, fn = TABLES[key]
        where = PROVINCES[code][1]
        print(f"\n{where} ({fn})")
        if kind == "pdf":
            lines, tail = lines_from_ningxia(fn)
        else:
            lines, tail = lines_from_sheet(sheet_rows(kind, fn), where)
        total, rest, how, errors = build_tree(lines, where)
        if not np.array_equal(total, nat[code]):
            d = np.nonzero(total != nat[code])[0]
            print(f"  !! does not match the national table in {len(d)} columns "
                  f"(total {total[0]:,} vs {nat[code][0]:,})")
            errors += 1
        n_leaf = sum(l["kind"] == "leaf" for l in rest)
        n_agg = sum(l["kind"] == "agg" for l in rest)
        print(f"  {total[0]:,} people; {n_agg} prefecture rows, {n_leaf} county rows "
              f"(structure by {how}); residual columns labelled {tail}")
        if errors:
            failed.append(where)
            continue
        print("  every prefecture adds up, and the total matches the national table")
        for seq, l in enumerate(rest):
            parent = rest[l["parent"]]["name"] if l["parent"] is not None else ""
            out_rows.append([code, seq, l["name"], l["kind"], parent] + l["v"].tolist())

    os.makedirs(WORK, exist_ok=True)
    # merge with provinces already parsed, so one province can be re-run on its own
    kept = []
    done = {r[0] for r in out_rows}
    if os.path.exists(OUT) and sys.argv[1:]:
        with open(OUT, encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            next(rd)
            kept = [r for r in rd if r[0] not in done]
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prov", "seq", "name", "kind", "parent", "total"] + KEYS)
        w.writerows(kept + out_rows)
    print(f"\nwrote {OUT}: {len(kept) + len(out_rows):,} rows")
    if failed:
        raise SystemExit(f"NOT written, failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
