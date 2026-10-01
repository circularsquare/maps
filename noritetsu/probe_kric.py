"""What the open Korean rail datasets hold, before a register reader is built on them.

    python probe_kric.py --fetch              # download the KRIC files listed in FILES
    python probe_kric.py --show <id> [--rows N]   # columns, row count and sample rows of one
    python probe_kric.py --dump [<id> ...]    # every sheet as UTF-8 CSV, to data/raw/kr/csv/
    python probe_kric.py --report             # the whole survey, to data/raw/kr/probe_report.txt
    python probe_kric.py --distance-table     # Korail's 각 선구별 거리표 (data.go.kr 15137040)
                                              # as ordered station chains with section km,
                                              # to data/raw/kr/probe_distance_table.txt

What was found is written up in data/raw/kr/SOURCES.md.

Korea has no open line-geometry register like N02 or the Swiss Schienennetz. What is open,
with no login or key, is KRIC's rail portal (data.kric.go.kr, run by 국가철도공단): station
lists per operator with coordinates, line lists, and for Busan, Daegu and Daejeon the
station order with section km. The intercity network's line lengths are in Korail's
철도통계연보, read by riders/koreariders/lines.py.

Download pattern, no login:
    https://data.kric.go.kr/rips/dataset/download.file?type=filedata&id=<ID>&operation=1
A bad id comes back as HTTP 200 with an HTML page, so every file is checked for the PK zip
signature before it is kept.
"""
import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "kr"
URL = "https://data.kric.go.kr/rips/dataset/download.file?type=filedata&id=%d&operation=1"
DETAIL = "https://data.kric.go.kr/rips/M_01_01/detail.do?id=%d"
UA = "Mozilla/5.0 (noritetsu register survey)"

# id -> short name. Titles are KRIC's own.
FILES = {
    18: "line_all",            # 표준데이터 노선정보(전체 기관)
    32: "station_all",         # 표준데이터 역사정보(전체 기관)
    1294: "station_national",  # 전국 도시광역철도 역사정보 (29 fields)
    31: "line_korail",         # 표준데이터 노선정보(코레일 기관)
    45: "station_korail",      # 표준데이터 역사정보(코레일 기관)
    1264: "station_korail29",  # 코레일 역사정보 (29 fields)
    216: "pos_korail",         # 코레일 역위치
    79: "stninfo_korail",      # 코레일 표준데이터의 역정보 (English, romanised)
    602: "lineinfo_korail",    # 코레일 호선정보
    624: "order_busan",        # 부산교통공사 호선구성역정보 (구간키로, 기점키로)
    629: "order_daegu",        # 대구도시철도공사 호선구성역정보
    633: "order_daejeon",      # 대전도시철도공사 호선구성역정보
}


def fetch(ids):
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for i in ids:
        short = FILES.get(i, "file")
        out = RAW / f"kric_{i}_{short}.xlsx"
        if out.exists() and out.stat().st_size > 0:
            print(f"  {i:>5}  have {out.name}")
            continue
        r = requests.get(URL % i, headers={"User-Agent": UA}, timeout=180)
        body = r.content
        if r.status_code == 200 and body[:2] == b"PK":
            out.write_bytes(body)
            print(f"  {i:>5}  {len(body)/1e6:6.2f} MB  {out.name}")
        else:
            bad = RAW / f"kric_{i}_{short}.error.html"
            bad.write_bytes(body)
            print(f"  {i:>5}  NOT A ZIP (HTTP {r.status_code}, {len(body)} bytes, "
                  f"starts {body[:40]!r}); saved {bad.name}")
        time.sleep(0.5)


def read(path, header=0):
    """All sheets of one workbook as {sheet: DataFrame}, stylesheet dropped for speed."""
    import pandas as pd
    src = zipfile.ZipFile(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename != "xl/styles.xml":
                dst.writestr(info, src.read(info.filename))
    buf.seek(0)
    try:
        return pd.read_excel(buf, sheet_name=None, header=header, dtype=str)
    except Exception:
        # openpyxl needs the stylesheet for a normal load; fall back to the whole file
        return pd.read_excel(path, sheet_name=None, header=header, dtype=str)


def show(i, rows):
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_colwidth", 30)
    hits = sorted(RAW.glob(f"kric_{i}_*.xlsx"))
    if not hits:
        print(f"no file for id {i}; run --fetch")
        return
    for path in hits:
        print(f"=== {path.name}")
        for sheet, df in read(path).items():
            print(f"--- sheet {sheet!r}: {len(df)} rows x {len(df.columns)} cols")
            for c in df.columns:
                vals = df[c].dropna()
                print(f"    {str(c)[:40]:<40} {vals.nunique():>6} distinct  "
                      f"e.g. {', '.join(map(str, vals.unique()[:3]))[:90]}")
            print(df.head(rows).to_string())


def dump(i, out_dir, header):
    """Every sheet of one file as UTF-8 CSV, for reading whole."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW.glob(f"kric_{i}_*.xlsx")):
        for sheet, df in read(path, header=header).items():
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(sheet))[:30]
            dst = out_dir / f"{path.stem}__{safe}.csv"
            df.to_csv(dst, index=False, encoding="utf-8")
            print(f"  {dst}  {len(df)} rows")


def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def sheets(i):
    path = next(RAW.glob(f"kric_{i}_*.xlsx"))
    return read(path)


def report(out):
    """Per operator and line: stations, coordinates, English names, summed section km."""
    import re
    import pandas as pd
    w = []

    # 18: the national line table, ordered station list packed in one cell
    df = next(iter(sheets(18).values()))
    w.append("=== 18 표준데이터 노선정보(전체 기관): %d rows" % len(df))
    for _, r in df.iterrows():
        seq = str(r["정거장구성"])
        seq = seq.strip().strip('"')
        parts = [p for p in re.split(r"[,+\n]", seq) if p.strip()]
        w.append("  %-6s %-28s %-22s n=%-3d len=%-8s  %s ~ %s" % (
            r["노선번호"], r["노선명"], str(r["운영기관명"])[:22], len(parts),
            r["노선연장"], r["기점명"], r["종점명"]))

    # 32 and 1294: stations
    for i, sheet_pick in ((32, 0), (1294, 0)):
        df = list(sheets(i).values())[sheet_pick]
        w.append("\n=== %d: %d rows, columns: %s" % (i, len(df), ", ".join(map(str, df.columns))))
        if i == 32:
            op, ln, lat, lon, en = "운영기관명", "노선명", "역위도", "역경도", "영문역사명"
            up = dn = None
        else:
            op, ln, lat, lon, en = "철도운영기관명", "운영노선", "역 위치(위도)", "역 위치(경도)", "역명(영어)"
            up, dn = "상행거리", "하행거리"
        g = df.groupby([op, ln], sort=False)
        for (o, l), s in g:
            has_xy = s[lat].map(num).notna().sum()
            has_en = s[en].notna().sum()
            extra = ""
            if up:
                u = s[up].map(num)
                d = s[dn].map(num)
                extra = "  up-km %d/%d sum %.1f  down-km %d/%d sum %.1f" % (
                    u.notna().sum(), len(s), u.sum(), d.notna().sum(), len(s), d.sum())
            w.append("  %-22s %-24s n=%-3d xy=%-3d en=%-3d%s" % (
                str(o)[:22], str(l)[:24], len(s), has_xy, has_en, extra))

    # 45, 1264, 216, 79: Korail's own
    for i, col in ((45, "노선명"), (1264, "운영노선"), (216, "선명"), (79, "선명")):
        df = list(sheets(i).values())[0]
        w.append("\n=== %d: %d rows, columns: %s" % (i, len(df), ", ".join(map(str, df.columns))))
        for l, s in df.groupby(col, sort=False):
            w.append("  %-24s n=%d" % (l, len(s)))

    # 624/629/633: order and chainage
    for i in (624, 629, 633):
        df = list(sheets(i).values())[0]
        w.append("\n=== %d: %d rows, columns: %s" % (i, len(df), ", ".join(map(str, df.columns))))
        for l, s in df.groupby("선명", sort=False):
            seg = s["구간키로"].map(num)
            ch = s["기점키로"].map(num)
            w.append("  %-10s n=%-3d 구간키로 %d given sum %.1f   last 기점키로 %s" % (
                l, len(s), seg.notna().sum(), seg.sum(), ch.dropna().iloc[-1] if ch.notna().any() else "-"))

    Path(out).write_text("\n".join(w), encoding="utf-8")
    print(out)


def distance_table(path, out):
    """Korail's 각 선구별 거리표 (data.go.kr 15137040): one triangular distance matrix per 선구.

    Every triangle puts its stations on a down-right diagonal, (r, c), (r+1, c+1), ..., with
    the pairwise distances either to the right of each name on its row (upper triangle) or
    below it in its column (lower triangle). So a station sequence is a diagonal chain of text
    cells, and the section km between neighbours is the cell beside the diagonal. The last
    cell of the first row (or column) is the chain's end-to-end distance, which checks the sum.
    """
    import pandas as pd
    books = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    w = []

    def isnum(v):
        return isinstance(v, (int, float)) and not pd.isna(v)

    def istext(v):
        return isinstance(v, str) and v.strip() and not v.strip().replace(".", "").isdigit() \
            and v.strip() not in ("-",)

    for sheet, df in books.items():
        g = df.values
        R, C = g.shape
        at = lambda r, c: g[r, c] if 0 <= r < R and 0 <= c < C else None
        w.append("\n=== sheet %s (%dx%d)" % (sheet, R, C))
        for r in range(R):
            for c in range(C):
                if not istext(at(r, c)) or istext(at(r - 1, c - 1)):
                    continue
                chain = [(r, c)]
                while istext(at(chain[-1][0] + 1, chain[-1][1] + 1)):
                    chain.append((chain[-1][0] + 1, chain[-1][1] + 1))
                if len(chain) < 2:
                    continue
                if isnum(at(r, c + 1)):
                    style = "row"
                    seg = [at(a, b + 1) for a, b in chain[:-1]]
                    total = at(r, c + len(chain) - 1)
                elif isnum(at(r + 1, c)):
                    style = "col"
                    seg = [at(a, b - 1) for a, b in chain[1:]]
                    total = at(r + len(chain) - 1, c)
                else:
                    continue
                names = [str(at(a, b)).strip() for a, b in chain]
                segs = [round(float(s), 2) if isnum(s) else None for s in seg]
                ssum = sum(s for s in segs if s is not None)
                flag = "" if isnum(total) and abs(ssum - float(total)) < 0.05 else "  <-- sum %.1f vs %s" % (ssum, total)
                w.append("  [%s @%d,%d] %d stations  %s ~ %s  %.1f km%s" % (
                    style, r, c, len(names), names[0], names[-1],
                    float(total) if isnum(total) else float("nan"), flag))
                w.append("      " + " ".join("%s -%s-" % (n, s) for n, s in zip(names, segs + [""])))
    Path(out).write_text("\n".join(w), encoding="utf-8")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance-table", nargs="?", default=None,
                    const=str(RAW / "dgk_15137040_각 선구별 거리표.xlsx"),
                    help="parse Korail's per-line triangular distance tables")
    ap.add_argument("--report", nargs="?", const=str(RAW / "probe_report.txt"), default=None)
    ap.add_argument("--dump", nargs="*", type=int, default=None,
                    help="write every sheet of these ids as UTF-8 CSV to --out")
    ap.add_argument("--out", default=str(RAW / "csv"))
    ap.add_argument("--header", type=int, default=0)
    ap.add_argument("--fetch", nargs="*", type=int, default=None,
                    help="ids to download (default: every id in FILES)")
    ap.add_argument("--show", type=int, default=None)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()
    if args.fetch is not None:
        fetch(args.fetch or list(FILES))
    if args.show is not None:
        show(args.show, args.rows)
    if args.distance_table:
        distance_table(args.distance_table, RAW / "probe_distance_table.txt")
    if args.report:
        report(args.report)
    if args.dump is not None:
        for i in (args.dump or list(FILES)):
            dump(i, args.out, args.header)


if __name__ == "__main__":
    main()
