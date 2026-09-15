"""Iran - the 1395 (November 2016) census, religion by province, from SCI's Statistical Yearbook.

Reads (or fetches) two files into data/raw/ir/ and writes data/normalized/ir.csv.
`sources/ir.md` is the write-up; `sources/ir_geo.py` builds the polygons and the grid.

## WHAT IS DRAWN

    Table 3-18  `جمعیت برحسب دین واستان: آبان ۱۳۹۵`   population by religion and province, COUNTS
                Statistical Yearbook of Iran 1395, chapter 3 (population), printed p159,
                PDF page index 36; source line: SCI, Office of Population, Labour Force and Census.

31 provinces, six columns: Muslim, Zoroastrian, Christian, Jewish (`کلیمی`), other (`سایر`) and
not stated (`اظهارنشده`). The chapter is SCI's own PDF, mirrored by Syracuse University's Iran
Data Portal (`irandataportal.syr.edu`), because `amar.org.ir` resets every connection from here and
its own copy of the yearbook (`n_Salname_95-V3.pdf`) survives in the Wayback Machine only truncated
(sources.md §11n). The table's text layer carries every province row in Persian digits; the bold
national row is drawn as a picture, so it is transcribed and checked against everything else.

## WHY 1395 AND NOT THE 1390 SHARES ASK 023 NAMED

Ask 023 and the scout (`sources.md §scout-2026-09-14-iran`) had only *Amar* no. 21's Table 3: 1390
shares to two decimals, with "other" merged into "not stated" and a Christian column split in two
under an "Assyrian or Chaldean" label that is flat across every province. Table 3-18 is the next
census, in counts, from the office's own yearbook, with other and not stated printed apart and one
Christian column. Anita's ruling was about grain (31 provinces) and about keeping other on its own
node; both hold here. The 1390 table stays in this module as a witness (check 7).

## THE CHECKS

1. Table 3-18 parsed off the page equals the transcription below, all 31 rows.
2. Each province's six cells sum to its printed total.
3. The 31 provinces sum to the transcribed national row, column by column.
4. The national row equals UNSD Demographic Yearbook table 28 for 2016, all six categories.
5. Table 3-17 on the page before (religion by sex for 1385, 1390 and 1395): its 1395 column is the
   national row, men plus women close, and its 1385 and 1390 columns equal UNSD's 2006 and 2011.
6. A second SCI release: the 1395 detailed-results table 3 (population by sex, province and
   citizenship, `3-jamiat-k.xls`, Wayback 2017) gives every province's total to the person.
7. 1390 witness: *Amar* no. 21 Table 3 rows sum to 100, its national row rounds from UNSD 2011,
   and the province that leads each minority (Yazd, Fars, Tehran) is the same in both censuses.

Usage:
    python sources/ir.py --fetch    the yearbook chapter (~2.2 MB) and the 1395 workbook (~30 KB)
    python sources/ir.py            normalise from data/raw/ir/
"""

import csv
import os
import re
import sys
import time
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ir")
OUT = os.path.join(ROOT, "data", "normalized", "ir.csv")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import FetchCheckError, check_body, digest, wayback_raw  # noqa: E402

SOURCE_ID = "ir_sci_census1395_yearbook_t3_18"
YEAR = 2016
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

PDF_URL = "https://irandataportal.syr.edu/wp-content/uploads/Population-3.pdf"
PDF = os.path.join(RAW, "sci_yearbook1395_ch3_population.pdf")
PDF_SIZE = 2_228_372
PDF_DIGEST = "WRANG4GEIGWXTSGTMKI7JDZSZP2EHXW4"     # SHA-1 base32, fetched 2026-09-14
PAGES = 56
PAGE_T317 = 35        # printed p158
PAGE_T318 = 36        # printed p159

XLS_URL = ("https://www.amar.org.ir/Portals/0/census/1395/results/tables/jamiat/tafsili/"
           "3-jamiat-k.xls")
XLS_TS = "20170801194630"
XLS_DIGEST = "72WO2ABYN6NWVXZWNTZ34J7ZPWPWQ6VW"      # the CDX digest; four captures 2017-2023 share it
XLS = os.path.join(RAW, "census1395_tafsili_3-jamiat-k.xls")

# The table's own column labels, in its order. They are the `source_category` strings.
CATS = ["مسلمان", "زرتشتی", "مسیحی", "کلیمی", "سایر", "اظهارنشده"]
EN = {"مسلمان": "Muslim", "زرتشتی": "Zoroastrian", "مسیحی": "Christian", "کلیمی": "Jewish",
      "سایر": "Other", "اظهارنشده": "Not stated"}

# Table 3-18, transcribed from the rendered page: province -> (COD-AB adm1_name, total, then CATS).
T318 = {
    "آذربایجان شرقی":      ("East Azerbaijan", 3909652, 3897236, 467, 4233, 35, 362, 7319),
    "آذربایجان غربی":      ("West Azerbaijan", 3265219, 3250233, 473, 7647, 46, 613, 6207),
    "اردبیل":              ("Ardabil", 1270420, 1266598, 181, 1455, 21, 51, 2114),
    "اصفهان":              ("Isfahan", 5120850, 5101133, 931, 8628, 1007, 6014, 3137),
    "البرز":               ("Alborz", 2712400, 2701984, 671, 3324, 60, 4027, 2334),
    "ایلام":               ("Ilam", 580158, 578654, 137, 859, 8, 17, 483),
    "بوشهر":               ("Bushehr", 1163400, 1158654, 140, 1833, 18, 300, 2455),
    "تهران":               ("Tehran", 13267637, 13179434, 8579, 43987, 5067, 9568, 21002),
    "چهارمحال و بختیاری":  ("Chaharmahal and Bakhtiari", 947763, 944990, 185, 1231, 8, 80, 1269),
    "خراسان جنوبی":        ("South Khorasan", 768898, 767455, 170, 1031, 10, 216, 16),
    "خراسان رضوی":         ("Razavi Khorasan", 6434501, 6409180, 961, 7159, 135, 1073, 15993),
    "خراسان شمالی":        ("North Khorasan", 863092, 861668, 124, 1036, 13, 64, 187),
    "خوزستان":             ("Khuzestan", 4710509, 4680333, 831, 6796, 68, 1211, 21270),
    "زنجان":               ("Zanjan", 1057461, 1055435, 170, 1074, 4, 147, 631),
    "سمنان":               ("Semnan", 702360, 700674, 99, 780, 16, 310, 481),
    "سیستان و بلوچستان":   ("Sistan and Baluchestan", 2775014, 2766139, 473, 4155, 40, 553, 3654),
    "فارس":                ("Fars", 4851274, 4835082, 839, 5880, 2816, 4649, 2008),
    "قزوین":               ("Qazvin", 1273761, 1271632, 127, 1253, 16, 672, 61),
    "قم":                  ("Qom", 1292283, 1284721, 151, 1112, 15, 63, 6221),
    "کردستان":             ("Kurdistan", 1603011, 1600537, 250, 1918, 39, 150, 117),
    "کرمان":               ("Kerman", 3164718, 3139968, 1280, 4460, 60, 3540, 15410),
    "کرمانشاه":            ("Kermanshah", 1952434, 1946809, 451, 2325, 105, 1431, 1313),
    "کهگیلویه و بویراحمد": ("Kohgiluyeh and Boyer-Ahmad", 713052, 711669, 104, 922, 9, 206, 142),
    "گلستان":              ("Golestan", 1868819, 1865881, 128, 1526, 15, 1007, 262),
    "گیلان":               ("Gilan", 2530696, 2527998, 160, 2210, 33, 210, 85),
    "لرستان":              ("Lorestan", 1760649, 1757509, 322, 2442, 33, 81, 262),
    "مازندران":            ("Mazandaran", 3283582, 3273724, 304, 3471, 32, 2100, 3951),
    "مرکزی":               ("Markazi", 1429475, 1425819, 248, 1674, 23, 384, 1327),
    "هرمزگان":             ("Hormozgan", 1776415, 1772720, 223, 2431, 12, 357, 672),
    "همدان":               ("Hamadan", 1738234, 1732458, 330, 2084, 31, 295, 3036),
    "یزد":                 ("Yazd", 1138533, 1131727, 3600, 1222, 31, 800, 1153),
}
# The bold `کل کشور` row, which is a picture on the page: total, then CATS.
NATIONAL = (79926270, 79598054, 23109, 130158, 9826, 40551, 124572)

# Table 3-17, both sexes / men / women for 1385, 1390 and 1395, per CATS label; and the `جمع` row.
T317 = {
    "مسلمان":    (70097741, 35663780, 34433961, 74682938, 37542060, 37140878,
                  79598054, 40325076, 39272978),
    "زرتشتی":    (19823, 10127, 9696, 25271, 13880, 11391, 23109, 12542, 10567),
    "مسیحی":     (109415, 54751, 54664, 117704, 63927, 53777, 130158, 69075, 61083),
    "کلیمی":     (9252, 4716, 4536, 8756, 4496, 4260, 9826, 5111, 4715),
    "سایر":      (54234, 27484, 26750, 49101, 24985, 24116, 40551, 21208, 19343),
    "اظهارنشده": (205317, 105504, 99813, 265899, 256321, 9578, 124572, 65430, 59142),
}
T317_TOTAL = (70495782, 35866362, 34629420, 75149669, 37905669, 37244000,
              79926270, 40498442, 39427828)

UNSD = {"Muslim": "مسلمان", "Zoroastrian": "زرتشتی", "Christian": "مسیحی", "Jewish": "کلیمی",
        "Other Religions": "سایر", "Not Specified": "اظهارنشده"}
UNSD_2016 = {"Muslim": 79598054, "Christian": 130158, "Not Specified": 124572,
             "Other Religions": 40551, "Zoroastrian": 23109, "Jewish": 9826}

# 1390 witness: Elham Fathi (SCI), *Amar* no. 21 (Azar-Dey 1395) pp. 23-26, Table 3, percentages to
# two decimals, transcribed from the rendered page (the text layer is unreadable). Linked from SCI
# news article 2918 as amar.org.ir/Portals/0/News/1396/1_srtc-amar-v4n5p23-fa.pdf (Wayback 2017).
# Columns: Muslim, `آشوری یا کلدانی`, `مسیحی`, Jewish, Zoroastrian, other and not stated.
T3_1390_NATIONAL = (99.38, 0.10, 0.06, 0.01, 0.03, 0.42)
T3_1390 = {
    "آذربایجان شرقی": (99.71, 0.06, 0.02, 0.00, 0.02, 0.19),
    "آذربایجان غربی": (99.46, 0.18, 0.04, 0.00, 0.02, 0.29),
    "اردبیل": (99.70, 0.07, 0.01, 0.00, 0.02, 0.20),
    "اصفهان": (99.33, 0.09, 0.11, 0.02, 0.02, 0.43),
    "البرز": (99.10, 0.09, 0.02, 0.00, 0.03, 0.76),
    "ایلام": (99.62, 0.12, 0.01, 0.00, 0.03, 0.22),
    "بوشهر": (96.93, 0.10, 0.01, 0.00, 0.02, 2.95),
    "تهران": (99.08, 0.12, 0.26, 0.04, 0.07, 0.43),
    "چهارمحال و بختیاری": (99.70, 0.09, 0.01, 0.00, 0.02, 0.17),
    "خراسان جنوبی": (99.04, 0.10, 0.01, 0.00, 0.03, 0.82),
    "خراسان رضوی": (99.43, 0.08, 0.01, 0.00, 0.02, 0.46),
    "خراسان شمالی": (99.61, 0.07, 0.01, 0.00, 0.02, 0.30),
    "خوزستان": (99.49, 0.13, 0.02, 0.00, 0.03, 0.35),
    "زنجان": (99.73, 0.06, 0.01, 0.00, 0.02, 0.18),
    "سمنان": (99.38, 0.08, 0.01, 0.00, 0.02, 0.51),
    "سیستان و بلوچستان": (99.49, 0.15, 0.01, 0.00, 0.03, 0.32),
    "فارس": (99.35, 0.08, 0.01, 0.06, 0.02, 0.48),
    "قزوین": (99.75, 0.07, 0.01, 0.00, 0.01, 0.15),
    "قم": (99.50, 0.06, 0.01, 0.00, 0.02, 0.41),
    "کردستان": (99.68, 0.09, 0.01, 0.00, 0.02, 0.20),
    "کرمان": (99.30, 0.12, 0.01, 0.00, 0.05, 0.52),
    "کرمانشاه": (99.67, 0.09, 0.01, 0.00, 0.02, 0.21),
    "کهگیلویه و بویراحمد": (99.63, 0.10, 0.01, 0.00, 0.02, 0.24),
    "گلستان": (99.45, 0.04, 0.01, 0.00, 0.01, 0.49),
    "گیلان": (99.79, 0.04, 0.01, 0.00, 0.01, 0.14),
    "لرستان": (99.56, 0.10, 0.01, 0.00, 0.02, 0.30),
    "مازندران": (99.51, 0.07, 0.01, 0.00, 0.01, 0.41),
    "مرکزی": (99.54, 0.07, 0.01, 0.00, 0.02, 0.36),
    "هرمزگان": (99.42, 0.12, 0.01, 0.00, 0.02, 0.43),
    "همدان": (99.54, 0.08, 0.01, 0.00, 0.02, 0.35),
    "یزد": (98.97, 0.07, 0.01, 0.01, 0.32, 0.62),
}
UNSD_2011 = {"Muslim": 74682938, "Not Specified": 265899, "Christian": 117704,
             "Other Religions": 49101, "Zoroastrian": 25271, "Jewish": 8756}

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}


def fold(s):
    """A Persian name as a join key: NFKC, Arabic yeh and kaf to Persian, no ZWNJ, no spaces."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    return re.sub(r"[\s‌‏‎]+", "", s)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def _get(url, where, kind, **pins):
    import urllib.request

    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
            check_body(body, kind, where=where, **pins)
            return body
        except FetchCheckError:
            raise
        except Exception as e:                    # noqa: BLE001 - the Wayback answers 5xx in bursts
            last = e
            print(f"  {where}: attempt {attempt + 1}: {e}")
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"{where}: no answer after 3 attempts ({last})")


def _write(path, body):
    with open(path + ".part", "wb") as fh:
        fh.write(body)
    os.replace(path + ".part", path)
    print(f"wrote {path} ({len(body):,} bytes, digest {digest(body)})")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == PDF_SIZE:
        print("already have", PDF)
    else:
        _write(PDF, _get(PDF_URL, "Iran Data Portal, yearbook 1395 ch. 3", "pdf",
                         pin_size=PDF_SIZE, pin_digest=PDF_DIGEST))
    if os.path.exists(XLS) and os.path.getsize(XLS) > 5_000:
        print("already have", XLS)
    else:
        _write(XLS, _get(wayback_raw(XLS_TS, XLS_URL), "Wayback, 3-jamiat-k.xls", "xls",
                         pin_digest=XLS_DIGEST))


def page_text(doc, pno):
    t = unicodedata.normalize("NFKC", doc.load_page(pno).get_text()).translate(DIGITS)
    return re.sub(r"\s+", " ", t)


ROW = re.compile(r"([^\d.:]+?)\s*\.{5,}[\s.]*" + r"\s".join([r"(\d+)"] * 7))


def read_t318(doc):
    """{folded province: (total, *CATS)} off page 159: a name, a dotted leader, seven numbers."""
    text = page_text(doc, PAGE_T318)
    if "برحسب دین واستان" not in text or "آبان 1395" not in text:
        raise SystemExit(f"page index {PAGE_T318} is not Table 3-18")
    return {fold(m.group(1)): tuple(int(g) for g in m.groups()[1:])
            for m in ROW.finditer(text)}


def read_t317(doc):
    """{label: nine numbers} off page 158: each label is followed by its dotted leader and nine."""
    text = page_text(doc, PAGE_T317)
    out = {}
    for label in CATS:
        m = re.search(re.escape(label) + r"\s*\.{3,}\s*" + r"\s".join([r"(\d+)"] * 9), text)
        out[label] = tuple(int(g) for g in m.groups()) if m else None
    return out


def read_xls():
    """{folded province: total} from the both-sexes block of 3-jamiat-k.xls, plus the national row."""
    import pandas as pd

    df = pd.read_excel(XLS, header=None)
    col0 = [fold(x) if isinstance(x, str) else "" for x in df[0]]
    start = col0.index(fold("مردوزن"))
    end = col0.index(fold("مرد"), start + 1)
    national = int(df.iloc[start, 1])
    rows = {}
    for i in range(start + 1, end):
        vals = [0 if pd.isna(v) else int(v) for v in df.iloc[i, 1:9]]
        rows[col0[i]] = (vals[0], sum(vals[1:]))       # total, and the citizenship columns summed
    return national, rows


def spearman(a, b):
    import numpy as np
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(a), np.asarray(b)).correlation)


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Iran - census 1395, Statistical Yearbook 1395 Table 3-18\n")
    say(doc.page_count == PAGES, f"the chapter is {doc.page_count} pages (expected {PAGES})")

    # 1. parsed = transcribed
    parsed = read_t318(doc)
    want = {fold(k): v[1:] for k, v in T318.items()}
    diff = sorted(k for k in want if parsed.get(k) != want[k])
    extra = sorted(set(parsed) - set(want))
    say(len(T318) == 31 and not diff and not extra and len(parsed) == 31,
        f"Table 3-18 parsed off the page: {len(parsed)} rows, every one identical to the "
        f"transcription (differ: {diff}, unexpected: {extra})")
    for k in diff:
        print(f"        {k}: page {parsed.get(k)} transcribed {want[k]}")

    # 2. rows close
    bad = [k for k, v in T318.items() if sum(v[2:]) != v[1]]
    say(not bad, f"every province's six cells sum to its total (failing: {bad})")

    # 3. provinces sum to the national row
    sums = tuple(sum(v[i] for v in T318.values()) for i in range(1, 8))
    say(sums == NATIONAL, f"the 31 provinces sum to the national row {NATIONAL} (got {sums})")
    say(sum(NATIONAL[1:]) == NATIONAL[0], "the national row's cells sum to its total")

    # 4. UNSD 2016
    try:
        import oracle
        u = oracle.oracle("Iran (Islamic Republic of)", 2016)
        tot = next(v for k, v in u.items() if "total" in k.lower())
        live = {c: v for c, v in tot.items() if c != oracle.TOTAL}
        say(live == UNSD_2016, "UNSD's cached Iran 2016 row equals the transcription")
        u11 = oracle.oracle("Iran (Islamic Republic of)", 2011)
        tot11 = next(v for k, v in u11.items() if "total" in k.lower())
        say({c: v for c, v in tot11.items() if c != oracle.TOTAL} == UNSD_2011,
            "UNSD's cached Iran 2011 row equals the transcription")
    except (SystemExit, ImportError, StopIteration):
        print("  -- oracle cache not present; using the transcribed UNSD rows")
    say(all(UNSD_2016[k] == NATIONAL[1 + CATS.index(c)] for k, c in UNSD.items()),
        "the national row equals UNSD table 28 Iran 2016 in all six categories")

    # 5. Table 3-17
    t317 = read_t317(doc)
    say(t317 == T317, "Table 3-17 parsed off p158 equals the transcription, all six religions")
    say(all(T317[c][6] == NATIONAL[1 + CATS.index(c)] for c in CATS),
        "its 1395 both-sexes column is Table 3-18's national row")
    say(all(v[3 * y] == v[3 * y + 1] + v[3 * y + 2] for v in list(T317.values()) + [T317_TOTAL]
            for y in range(3)), "men + women = both sexes, every row, all three censuses")
    say(all(sum(T317[c][i] for c in CATS) == T317_TOTAL[i] for i in range(9)),
        "the six religions sum to the printed totals in all nine columns")
    say(all(UNSD_2011[k] == T317[c][3] for k, c in UNSD.items()),
        "its 1390 column equals UNSD table 28 Iran 2011")

    # 6. the 1395 detailed results, a second release
    if os.path.exists(XLS):
        national, rows = read_xls()
        say(national == NATIONAL[0], f"3-jamiat-k.xls national total {national:,}")
        say(set(rows) == set(want), f"3-jamiat-k.xls names the same 31 provinces ({len(rows)})")
        mism = sorted(k for k, v in T318.items() if rows.get(fold(k), (None,))[0] != v[1])
        say(not mism, f"every province total equals 3-jamiat-k.xls to the person (differ: {mism})")
        say(all(t == s for t, s in rows.values()),
            "and in that workbook each province's citizenship columns sum to its total")
    else:
        say(False, f"{XLS} missing: run --fetch")

    # 7. the 1390 witness
    worst = max(abs(sum(v) - 100.0) for v in list(T3_1390.values()) + [T3_1390_NATIONAL])
    say(worst <= 0.03 + 1e-9, f"Amar Table 3 (1390) rows sum to 100 within {worst:.2f} "
        "(bound 0.03 = 6 cells x 0.005)")
    n11 = sum(UNSD_2011.values())
    rec = (round(100 * UNSD_2011["Muslim"] / n11, 2),
           round(100 * UNSD_2011["Christian"] / n11, 2),
           round(100 * UNSD_2011["Jewish"] / n11, 2),
           round(100 * UNSD_2011["Zoroastrian"] / n11, 2),
           round(100 * (UNSD_2011["Other Religions"] + UNSD_2011["Not Specified"]) / n11, 2))
    printed = (T3_1390_NATIONAL[0], round(T3_1390_NATIONAL[1] + T3_1390_NATIONAL[2], 2),
               T3_1390_NATIONAL[3], T3_1390_NATIONAL[4], T3_1390_NATIONAL[5])
    say(rec == printed, f"its national row rounds from UNSD 2011's counts: {rec} = {printed} "
        "(the two Christian columns together)")
    say(set(map(fold, T3_1390)) == set(want), "and it names the same 31 provinces")

    def share(k, c):
        return T318[k][2 + CATS.index(c)] / T318[k][1]

    names = list(T318)
    top = {}
    for c, col in (("زرتشتی", lambda v: v[4]), ("کلیمی", lambda v: v[3]),
                   ("مسیحی", lambda v: v[1] + v[2])):
        a = max(names, key=lambda k: share(k, c))
        b = max(names, key=lambda k: col(T3_1390[k]))
        top[c] = (a, b)
        say(a == b, f"{EN[c]}: the highest share is {T318[a][0]} in 1395 and "
            f"{T318[b][0]} in 1390")
    s95 = [share(k, "مسیحی") for k in names]
    print(f"      Spearman across provinces, 1395 Christian share against 1390's "
          f"`آشوری یا کلدانی` {spearman(s95, [T3_1390[k][1] for k in names]):+.2f}, its "
          f"`مسیحی` {spearman(s95, [T3_1390[k][2] for k in names]):+.2f}, the two summed "
          f"{spearman(s95, [T3_1390[k][1] + T3_1390[k][2] for k in names]):+.2f}")
    print(f"      Spearman, Zoroastrian {spearman([share(k, 'زرتشتی') for k in names], [T3_1390[k][4] for k in names]):+.2f}; "
          f"other + not stated {spearman([share(k, 'سایر') + share(k, 'اظهارنشده') for k in names], [T3_1390[k][5] for k in names]):+.2f}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    rows = []
    for fa, (en, total, *cells) in T318.items():
        for c, n in zip(CATS, cells):
            if n <= 0:
                continue
            rows.append({
                "geo_id": en, "geo_level": "province", "geo_name": nfc(fa),
                "source_category": nfc(c), "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": f"SCI Statistical Yearbook 1395 Table 3-18, counts; province total {total}",
            })
    return rows


def summary(rows):
    total = sum(r["count"] for r in rows)
    print(f"\n  31 provinces, {total:,} people, {total / 31:,.0f} each")
    by_cat, by_prov = {}, {}
    for r in rows:
        by_cat[r["source_category"]] = by_cat.get(r["source_category"], 0) + r["count"]
        by_prov.setdefault(r["geo_id"], {})[r["source_category"]] = r["count"]
    for c in CATS:
        n = by_cat[nfc(c)]
        lead = sorted(by_prov, key=lambda p: -by_prov[p].get(nfc(c), 0))[:3]
        where = ", ".join(f"{p} {by_prov[p][nfc(c)]:,} ({100 * by_prov[p][nfc(c)] / n:.1f}%, "
                          f"{100 * by_prov[p][nfc(c)] / sum(by_prov[p].values()):.3f}% of it)"
                          for p in lead)
        print(f"    {EN[c]:<12}{n:>11,}  {100 * n / total:7.3f}%   {where}")
    chr_share = sorted((100 * d[nfc("مسیحی")] / sum(d.values()), p) for p, d in by_prov.items())
    print("\n  Christian share by province, low to high: "
          + ", ".join(f"{p} {s:.3f}" for s, p in chr_share))


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing - run: python sources/ir.py --fetch")
    doc = fitz.open(PDF)
    check(doc)
    rows = emit()
    summary(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
