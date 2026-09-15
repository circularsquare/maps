"""Japan — the two checks Anita asked for on 2026-09-14. Neither draws anything.

    python sources/jp_checks.py --fetch    the chart GIF, the roll table, 2024 population
    python sources/jp_checks.py            both checks

1. CHART. 社会実情データ図録 図録7770 publishes NHK's 1996 全国県民意識調査 religion question as an
   815x655 stacked-bar GIF, one bar per prefecture plus the nation. Measured here segment by
   segment and written to data/raw/jp/nhk1996_7770_measured.csv.

   Geometry, read off the image: gridlines at rows 104, 169 ... 559 for 70% ... 0%, so 6.5 px
   per percentage point. Every segment is a rectangle with a black border and adjacent borders
   make a 1-3 px black run; a segment's value is the distance between the CENTRES of the runs
   that bound it, over 6.5. Printed labels sit inside large segments and can be wider than the
   bar, so a row of text can look like a border; each category appears once per bar, so two
   adjacent pieces of one colour are one segment cut by its label and are merged.

   PRECISION: every bar's measured total is within 0.08 points of the total printed above it
   (48 of 48, mean |d| 0.04), and segment values read against the printed labels on a 3x crop
   agree to 0.14 or better. One pixel is 0.154 points. Categories under about 0.3 points cannot
   be told from a zero-height segment, whose two borders merge.

2. ROLL. The Agency for Cultural Affairs' 宗教統計調査 table 2(2), 2024-12-31, Christian sheet,
   against 2024 population, the 1996 chart's Christian segment, and the Catholic Bishops'
   Conference's 2023 Catholics by diocese (カトリック教会現勢 2023 p.3, typed below).
"""
import csv
import os
import ssl
import sys
import urllib.request

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data", "raw", "jp")
GIF = os.path.join(RAW, "honkawa_7770.gif")
ROLL = os.path.join(RAW, "aca_r7_table2_2_prefecture.xls")
POP = os.path.join(RAW, "jinsui_2024_table2.xlsx")
OUT = os.path.join(RAW, "nhk1996_7770_measured.csv")
URLS = {GIF: "https://honkawa2.sakura.ne.jp/images2/7770.gif",
        ROLL: "https://www.e-stat.go.jp/stat-search/file-download?statInfId=000040387906&fileKind=0",
        POP: "https://www.stat.go.jp/data/jinsui/2024np/zuhyou/05k2024-2.xlsx"}

PX_PER_PT, AXIS_ROW, FRAME_TOP = 6.5, 559, 96
CATS = [("tendai_shingon", (255, 0, 0)), ("jodo_shinshu", (255, 153, 204)),
        ("zen", (204, 255, 204)), ("nichiren", (0, 255, 0)), ("soka_gakkai", (255, 255, 0)),
        ("rissho_koseikai", (255, 153, 0)), ("other_buddhist", (192, 192, 192)),
        ("shinto", (255, 255, 255)), ("christian", (0, 0, 255))]
PREFS = ["北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬", "埼玉",
         "千葉", "東京", "神奈川", "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡",
         "愛知", "三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山", "鳥取", "島根", "岡山",
         "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎", "熊本", "大分",
         "宮崎", "鹿児島", "沖縄"]
UNITS = ["全国"] + PREFS
# The total printed above each bar, read off the image; 7770a's ranking table agrees on all
# fifteen prefectures it lists.
PRINTED_TOTAL = [31.2, 33.6, 26.8, 26.5, 22.2, 24.6, 26.6, 21.4, 21.5, 19.6, 24.5, 21.1, 18.1,
                 27.0, 23.7, 26.2, 49.3, 43.7, 58.0, 28.3, 31.7, 39.8, 32.8, 36.6, 37.3, 39.9,
                 34.3, 35.5, 36.1, 32.1, 33.0, 34.3, 37.1, 36.7, 53.7, 36.1, 34.4, 40.5, 30.4,
                 23.7, 39.0, 38.6, 45.2, 38.7, 35.0, 35.9, 44.6, 7.8]
# CBCJ カトリック教会現勢 2023, p.3, total Catholics, and the prefectures each diocese covers.
DIOCESE = {"Sapporo": (14331, ["北海道"]), "Sendai": (9075, ["青森", "岩手", "宮城", "福島"]),
           "Niigata": (6677, ["新潟", "山形", "秋田"]),
           "Saitama": (19412, ["埼玉", "群馬", "栃木", "茨城"]), "Tokyo": (94855, ["東京", "千葉"]),
           "Yokohama": (53044, ["神奈川", "静岡", "山梨", "長野"]),
           "Nagoya": (26714, ["愛知", "岐阜", "石川", "富山", "福井"]),
           "Kyoto": (17512, ["京都", "滋賀", "奈良", "三重"]),
           "Osaka-Takamatsu": (50656, ["大阪", "兵庫", "和歌山", "徳島", "香川", "愛媛", "高知"]),
           "Hiroshima": (19389, ["広島", "岡山", "山口", "島根", "鳥取"]),
           "Fukuoka": (29080, ["福岡", "佐賀", "熊本"]), "Nagasaki": (57061, ["長崎"]),
           "Oita": (5833, ["大分", "宮崎"]), "Kagoshima": (8252, ["鹿児島"]),
           "Naha": (6210, ["沖縄"])}


def fetch():
    ctx = ssl.create_default_context()
    os.makedirs(RAW, exist_ok=True)
    for dest, url in URLS.items():
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  have {os.path.basename(dest)}")
            continue
        body = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                      timeout=120, context=ctx).read()
        with open(dest + ".part", "wb") as fh:
            fh.write(body)
        os.replace(dest + ".part", dest)
        print(f"  got {os.path.basename(dest)} ({len(body):,} bytes)")


def _runs(mask, lo, hi):
    out, y = [], lo
    while y <= hi:
        if mask[y]:
            s = y
            while y <= hi and mask[y]:
                y += 1
            out.append((s, y - 1))
        else:
            y += 1
    return out


def chart():
    from PIL import Image
    a = np.asarray(Image.open(GIF).convert("RGB")).astype(int)
    black = (a == 0).all(axis=2)
    colour = {n: (a == np.array(rgb)).all(axis=2) for n, rgb in CATS}
    nonwhite = np.zeros(black.shape, bool)
    for n, _ in CATS:
        if n != "shinto":
            nonwhite |= colour[n]
    has = nonwhite[FRAME_TOP + 1:AXIS_ROW, :].sum(axis=0) >= 3
    bars, x = [], 0
    while x < a.shape[1]:
        if has[x]:
            x0 = x
            while x < a.shape[1] and has[x]:
                x += 1
            bars.append((x0, x - 1))
        x += 1
    if len(bars) != len(UNITS):
        raise SystemExit(f"chart: {len(bars)} bars, expected {len(UNITS)}")
    names = [n for n, _ in CATS]
    out = []
    for (x0, x1), unit, printed in zip(bars, UNITS, PRINTED_TOTAL):
        xs = list(range(x0, x1 + 1))
        frac = black[:, xs].mean(axis=1)
        tops = []
        for xo in (x0 - 1, x1 + 1):   # the national bar's left outline IS the y-axis
            y = AXIS_ROW - 1
            while y > FRAME_TOP and black[y, xo]:
                y -= 1
            tops.append(y + 1)
        top = max(tops)
        sep = frac >= 0.85
        sep[top] = sep[AXIS_ROW] = True
        seps = _runs(sep, top, AXIS_ROW)
        segs = []
        for (s_hi, e_hi), (s_lo, _) in reversed(list(zip(seps[:-1], seps[1:]))):
            r0, r1 = e_hi + 1, s_lo - 1
            if r1 < r0:
                continue
            counts = {n: int(colour[n][r0:r1 + 1, xs].sum()) for n in names}
            best = max(counts, key=counts.get)
            lo, hi = (s_lo + _) / 2, (s_hi + e_hi) / 2
            if counts[best] < 0.3 * (r1 - r0 + 1) * len(xs):
                if segs:                  # all text or border: give it to the segment below
                    segs[-1][1] = hi
                continue
            if segs and segs[-1][2] == best:
                segs[-1][1] = hi
            else:
                segs.append([lo, hi, best])
        vals = {n: 0.0 for n in names}
        for lo, hi, c in segs:
            vals[c] += (lo - hi) / PX_PER_PT
        order = [names.index(c) for _, _, c in segs]
        total = (AXIS_ROW - top) / PX_PER_PT
        out.append({"unit": unit, "total_measured": round(total, 2), "total_printed": printed,
                    "legend_order_ok": order == sorted(order),
                    **{n: round(v, 2) for n, v in vals.items()}})
    d = np.array([r["total_measured"] - r["total_printed"] for r in out])
    if np.abs(d).max() > 0.15 or not all(r["legend_order_ok"] for r in out):
        raise SystemExit(f"chart: a bar disagrees with its printed total by {np.abs(d).max():.2f}")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"chart: 48 bars, total vs printed mean |d| {np.abs(d).mean():.3f}, max {np.abs(d).max():.3f}"
          f" -> {os.path.basename(OUT)}")
    return {r["unit"]: r for r in out}


def _spearman(x, y):
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def roll(nhk):
    import openpyxl
    import xlrd
    sh = xlrd.open_workbook(ROLL).sheet_by_name("キリスト教系宗教団体")
    chr_ = {}
    for r in range(sh.nrows):
        v = sh.row_values(r)
        if str(v[1]).strip() in PREFS + ["都道府県名／総数"]:
            chr_[str(v[1]).strip()] = (float(v[7]), float(v[20]))      # bodies, believers
    if len([p for p in chr_ if p in PREFS]) != 47:
        raise SystemExit("roll: the Christian sheet does not list 47 prefectures")
    nat = chr_["都道府県名／総数"][1]
    if nat != sum(chr_[p][1] for p in PREFS):
        raise SystemExit("roll: the national row is not the sum of the prefectures")
    pop = {}
    for row in openpyxl.load_workbook(POP, data_only=True).worksheets[0].iter_rows(values_only=True):
        c = [x for x in row if x is not None]
        if len(c) >= 3 and isinstance(c[1], str) and isinstance(c[2], (int, float)):
            raw = c[1].strip()               # some cells carry trailing spaces
            nm = {"東京都": "東京", "京都府": "京都", "大阪府": "大阪", "北海道": "北海道"}.get(
                raw, raw.rstrip("県"))
            if nm in PREFS:
                pop[nm] = c[2] * 1000
    if len(pop) != 47:
        raise SystemExit(f"roll: population parsed for {len(pop)} prefectures")
    bodies = np.array([chr_[p][0] for p in PREFS])
    bel = np.array([chr_[p][1] for p in PREFS])
    popv = np.array([pop[p] for p in PREFS])
    share = 100 * bel / popv
    nhk_sh = np.array([nhk[p]["christian"] for p in PREFS])
    t = PREFS.index("東京")
    rest = [i for i in range(47) if i != t]
    per_rest = bel[rest].sum() / bodies[rest].sum()
    print(f"roll: {nat:,.0f} Christian believers = {100 * nat / popv.sum():.2f}% of 2024 population; "
          f"NHK 1996 national {nhk['全国']['christian']:.2f}%")
    print(f"  Tokyo: {100 * bel[t] / nat:.1f}% of the roll on {100 * popv[t] / popv.sum():.1f}% of the "
          f"population; {share[t]:.2f}% roll against {nhk_sh[t]:.2f}% NHK 1996")
    print(f"  believers per Christian body: Tokyo {bel[t] / bodies[t]:.0f}, rest {per_rest:.0f}; Tokyo at "
          f"the rest's rate is {bodies[t] * per_rest:,.0f}, an excess of {bel[t] - bodies[t] * per_rest:,.0f}"
          f" ({100 * (bel[t] - bodies[t] * per_rest) / nat:.1f}% of the national roll)")
    ratio = share[rest] / np.where(nhk_sh[rest] > 0, nhk_sh[rest], np.nan)
    print(f"  outside Tokyo, roll / NHK 1996 per prefecture: median {np.nanmedian(ratio):.2f}; "
          f"Spearman roll vs NHK: all 47 {_spearman(share, nhk_sh):+.3f}, without Tokyo "
          f"{_spearman(share[rest], nhk_sh[rest]):+.3f}")
    for p in ("長崎", "神奈川", "沖縄"):
        i = PREFS.index(p)
        print(f"  {p}: roll {share[i]:.2f}%  NHK 1996 {nhk_sh[i]:.2f}%")
    cath, ro, nh = [], [], []
    for d, (n, ps) in DIOCESE.items():
        dp = sum(pop[p] for p in ps)
        cath.append(n / dp)
        ro.append(sum(chr_[p][1] for p in ps) / dp)
        nh.append(sum(nhk[p]["christian"] / 100 * pop[p] for p in ps) / dp)
    i = list(DIOCESE).index("Nagasaki")
    j = list(DIOCESE).index("Tokyo")
    print(f"  15 dioceses, Spearman: CBCJ Catholics vs roll Christians {_spearman(cath, ro):+.3f}; "
          f"Catholics vs NHK 1996 {_spearman(cath, nh):+.3f}; roll vs NHK 1996 {_spearman(ro, nh):+.3f}")
    print(f"  Catholics as a share of the roll's Christians: Nagasaki {100 * cath[i] / ro[i]:.1f}%, "
          f"Tokyo archdiocese {100 * cath[j] / ro[j]:.1f}%")


def main():
    if "--fetch" in sys.argv:
        fetch()
    nhk = chart()
    roll(nhk)


if __name__ == "__main__":
    main()
