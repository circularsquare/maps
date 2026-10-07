"""Thailand: 2010 Population and Housing Census, language usually spoken in the household.

    python sources/th_census.py --fetch    download the 77 provincial reports and the
                                           whole-kingdom report, keep the language-table pages
                                           in data/raw/th/, then normalise
    python sources/th_census.py            normalise what is on disk

Source: NSO, "The 2010 Population and Housing Census", complete reports (รายงานผลสมบูรณ์), one
PDF per changwat plus the whole kingdom, listed at https://www.nso.go.th/nsoweb/main/summano/aE
(year 2553, type "สมบูรณ์"). Published 2012. Each provincial report's Table 6 (Table 7 in the
whole-kingdom report) is "Population by usual languages spoken at home, sex and area": total,
male, female, for the whole province, its municipal areas and its non-municipal areas.

The question (questionnaire item G) is asked once per HOUSEHOLD: "Language usually spoken in this
household: Thai only / Thai and other / Other language only; specify language". Every member of
the household is counted under its answer. The table prints the three-way split, then one row per
"other" language, the rows summing to "Thai and other" + "Other language only": each household
names one other language.

Writes data/normalized/th.csv: geo_id (religiondots' unit code, `TH` for the kingdom), geo_level
(province / national), geo_name, source_category (the row's English label as this script
canonicalises it), label_th (the row's Thai label), count, municipal, nonmunicipal.

Bueng Kan was created in March 2011, seven months after the census; NSO nevertheless published a
report for it. Nong Khai's report is the whole pre-2011 changwat (821,526, Bueng Kan included, the
figure religiondots has for unit 443), so Bueng Kan's report is a check and is not added.

The tables are weighted estimates (the 2010 census's detailed questions were asked of a sample of
households; the report's appendix gives the estimators), rounded to the person, so identities
hold to the rounding of their terms. The reports were typeset province by province and are
untidy: some leave out rows that are zero, print a figure twice or a line high, misprint a cell,
or set a number without its comma. The table is read by position on the page, not by text order,
and every quirk met is logged in the run's output.

CHECKS:
  * every row's total equals its male + female or its municipal + non-municipal (to 2); a cell
    contradicting its row elsewhere is logged and not used;
  * every table: Thai only + Thai and other + other only = total (to 3); the language rows never
    exceed Thai and other + other only (the gap is households naming no other language: 81,519
    in the kingdom, 81,307 summed over the provinces);
  * each province's total against the census total religiondots read for the same unit from
    NSO's provincial indicator sheets (a different document): 74 agree to 12 people, and Rayong
    and Ratchaburi's language tables are short by 50,163 and 44,987;
  * the 76 provinces against the whole-kingdom table, row by row: Total, Thai only and the two
    other splits differ by Rayong and Ratchaburi's shortfall (-95,155, of it -90,815 Thai only);
    every language row is within 2% (the largest gaps are Khmer, -1,602 of 180,533, and Burmese,
    -915 of 827,713).

REPAIRS, each found by the kingdom check and confirmed by it once made (see SHIFTED and
FROM_KINGDOM): Phitsanulok's and Chumphon's first block of language rows is printed one line
high; Narathiwat's English row repeats its "Other languages" subtotal (569,967) and is replaced
by the kingdom's English less the other provinces (1,076); Uttaradit's Vietnamese total is
printed as 7,204 beside eight zeros and is taken as 0.
"""
import csv
import difflib
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "th"
OUT = ROOT / "data" / "normalized" / "th.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
BASE = "https://www.nso.go.th/nsoweb/storage/title_presentation/2023/"

# religiondots unit -> (name as NSO's listing prints it, file). The listing's spellings are kept
# as printed (ประจวบครีขันธ์, ศรีษะเกษ, อยุธยา are its typos and short forms); the report's own
# text is checked for the official name in LOOKUP_TH below.
REPORTS = {
    "TH": ("ทั่วราชอาณาจักร", "20230512163226_54316.pdf"),
    "110": ("กรุงเทพมหานคร", "20230512163350_35405.pdf"),
    "362": ("กำแพงเพชร", "20230512152504_39961.pdf"),
    "357": ("เชียงราย", "20230512152504_14064.pdf"),
    "350": ("เชียงใหม่", "20230512152504_18005.pdf"),
    "363": ("ตาก", "20230512152504_28925.pdf"),
    "360": ("นครสวรรค์", "20230512152504_18026.pdf"),
    "355": ("น่าน", "20230512154547_46091.pdf"),
    "356": ("พะเยา", "20230512154547_74327.pdf"),
    "366": ("พิจิตร", "20230512154547_30901.pdf"),
    "365": ("พิษณุโลก", "20230512154547_28136.pdf"),
    "367": ("เพชรบูรณ์", "20230512154547_94086.pdf"),
    "354": ("แพร่", "20230512154844_73296.pdf"),
    "358": ("แม่ฮ่องสอน", "20230512154844_26358.pdf"),
    "352": ("ลำปาง", "20230512154844_44392.pdf"),
    "351": ("ลำพูน", "20230512154844_69639.pdf"),
    "364": ("สุโขทัย", "20230512154844_59962.pdf"),
    "353": ("อุตรดิตถ์", "20230512155647_49251.pdf"),
    "361": ("อุทัยธานี", "20230512155647_37257.pdf"),
    "271": ("กาญจนบุรี", "20230512150352_81574.pdf"),
    "222": ("จันทบุรี", "20230512150352_57217.pdf"),
    "224": ("ฉะเชิงเทรา", "20230512150352_75874.pdf"),
    "220": ("ชลบุรี", "20230512150715_51639.pdf"),
    "218": ("ชัยนาท", "20230512150715_99939.pdf"),
    "223": ("ตราด", "20230512150715_94865.pdf"),
    "226": ("นครนายก", "20230512150715_85220.pdf"),
    "273": ("นครปฐม", "20230512150715_71034.pdf"),
    "212": ("นนทบุรี", "20230512150715_26087.pdf"),
    "213": ("ปทุมธานี", "20230512150715_54262.pdf"),
    "277": ("ประจวบครีขันธ์", "20230512150715_90849.pdf"),
    "225": ("ปราจีนบุรี", "20230512151521_33769.pdf"),
    "214": ("อยุธยา", "20230512151521_30132.pdf"),
    "276": ("เพชรบุรี", "20230512151521_58966.pdf"),
    "221": ("ระยอง", "20230512151521_69146.pdf"),
    "270": ("ราชบุรี", "20230512151521_96705.pdf"),
    "216": ("ลพบุรี", "20230512151521_34907.pdf"),
    "211": ("สมุทรปราการ", "20230512151521_81163.pdf"),
    "275": ("สมุทรสงคราม", "20230512151521_36811.pdf"),
    "274": ("สมุทรสาคร", "20230512151521_13027.pdf"),
    "219": ("สระบุรี", "20230512151521_13153.pdf"),
    "227": ("สระแก้ว", "20230512151521_46003.pdf"),
    "217": ("สิงห์บุรี", "20230512151521_91859.pdf"),
    "272": ("สุพรรณบุรี", "20230512151521_98547.pdf"),
    "215": ("อ่างทอง", "20230512151521_86787.pdf"),
    "581": ("กระบี่", "20230512161654_15294.pdf"),
    "586": ("ชุมพร", "20230512161654_13537.pdf"),
    "592": ("ตรัง", "20230512161654_33949.pdf"),
    "580": ("นครศรีธรรมราช", "20230512161654_87433.pdf"),
    "596": ("นราธิวาส", "20230512161654_65434.pdf"),
    "594": ("ปัตตานี", "20230512162034_85664.pdf"),
    "582": ("พังงา", "20230512162034_22690.pdf"),
    "593": ("พัทลุง", "20230512162034_93568.pdf"),
    "583": ("ภูเก็ต", "20230512162034_81209.pdf"),
    "595": ("ยะลา", "20230512162034_33359.pdf"),
    "585": ("ระนอง", "20230512162850_21030.pdf"),
    "590": ("สงขลา", "20230512162850_44761.pdf"),
    "591": ("สตูล", "20230512162850_31609.pdf"),
    "584": ("สุราษฎร์ธานี", "20230512162850_95564.pdf"),
    "446": ("กาฬสินธุ์", "20230512160240_13482.pdf"),
    "440": ("ขอนแก่น", "20230512160240_79103.pdf"),
    "436": ("ชัยภูมิ", "20230512160240_84320.pdf"),
    "448": ("นครพนม", "20230512160240_23057.pdf"),
    "445": ("ร้อยเอ็ด", "20230512160927_47827.pdf"),
    "430": ("นครราชสีมา", "20230512160240_63039.pdf"),
    "431": ("บุรีรัมย์", "20230512160713_98380.pdf"),
    "433": ("ศรีษะเกษ", "20230512160927_11019.pdf"),
    "447": ("สกลนคร", "20230512160927_78983.pdf"),
    "443/bk": ("บึงกาฬ", "20230512160713_35860.pdf"),
    "442": ("เลย", "20230512160927_98053.pdf"),
    "443": ("หนองคาย", "20230512161301_54998.pdf"),
    "444": ("มหาสารคาม", "20230512160713_65743.pdf"),
    "439": ("หนองบัวลำภู", "20230512161301_48083.pdf"),
    "449": ("มุกดาหาร", "20230512160713_23017.pdf"),
    "435": ("ยโสธร", "20230512160713_45991.pdf"),
    "432": ("สุรินทร์", "20230512160927_22704.pdf"),
    "441": ("อุดรธานี", "20230512161301_18569.pdf"),
    "434": ("อุบลราชธานี", "20230512161301_41364.pdf"),
    "437": ("อำนาจเจริญ", "20230512161301_41746.pdf"),
}
OFFICIAL = {"433": "ศรีสะเกษ", "277": "ประจวบคีรีขันธ์", "214": "พระนครศรีอยุธยา"}

# The table's rows in printed order: (canonical label, English label regex). Rows 0-3 are the
# three-way split under the total; 4-43 are the "other" languages.
ROWS = [
    ("Total", r"^Total$"),
    ("Only Thai language", r"^Only Thai"),
    ("Thai and other languages", r"^Thai and other"),
    ("Only other languages", r"^Only other"),
    ("Karen", r"^Karen$"),
    ("Thaikueng", r"^Thai ?kh?ueng$"),
    ("Morn", r"^Morn?$"),
    ("Lao-krung", r"^Lao-? ?kr[ua]ng$"),
    ("Hmong/Mea", r"^Hmong"),
    ("Local languages", r"^Local languages?$"),
    ("Malay/yawi", r"^Malay ?/ ?yawi$"),
    ("Dialect and others in Thailand", r"^Dialect and others"),
    ("Chinese", r"^Chinese$"),
    ("Burmese", r"^Burm"),
    ("Vietnamese", r"^Vi[ae]tnam"),
    ("Lao", r"^Lao$"),
    ("Cambodia", r"^Cambodia"),
    ("Korean", r"^Korean$"),
    ("Japanese", r"^Japanese$"),
    ("Tagalog/Filipino", r"^Tagalog"),
    ("Bengali/Banca Lee/Bangladesh", r"^Bengali"),
    ("Malaysia", r"^Malaysia$"),
    ("Indonesia", r"^Indonesia$"),
    ("India/Hindi", r"^Ind[ai]+ ?/ ?Hindi$"),
    ("Arab", r"^Arub|^Arab"),
    ("Other languages in Asia", r"^Others? language in Asia$"),
    ("English", r"^English$"),
    ("German", r"^German$"),
    ("Greek", r"^Greek$"),
    ("Spanish", r"^Spanish$"),
    ("Polish", r"^Polish$"),
    ("Portugal", r"^Portug"),
    ("Russian", r"^Russian$"),
    ("Swedish", r"^Swedish$"),
    ("Finnish", r"^Finnish$"),
    ("French", r"^French$"),
    ("Danish", r"^Danish$"),
    ("Italian", r"^Italian$"),
    ("Hungarian", r"^Hungarian$"),
    ("Mexican", r"^Mexican$"),
    ("Cuban", r"^Cuban$"),
    ("Other languages in Europe, America, Australia", r"America, Australia$|^Others? language in Europe"),
    ("Africa", r"^Africa$"),
    ("Other languages in Africa", r"^in Africa$|^Others? language( in Africa)?$"),
]
# the kingdom table's Thai labels, for recognising a row whose English label is garbled
TH_LABELS = {
    "Karen": ["กระเหรี่ยง"], "Thaikueng": ["ไทยขึน/ไทยเลย/ลาวเลย"], "Morn": ["มอญ"],
    "Lao-krung": ["ลาวครั่ง/ลาวขี้ครั่ง"], "Hmong/Mea": ["ม้ง/แม้ว"], "Local languages": ["ภาษาถิ่น"],
    "Malay/yawi": ["มลายูถิ่น/นายู/ยาวี"], "Dialect and others in Thailand": ["ภาษาพื้นเมืองและขาวเขาอื่นๆ"],
    "Chinese": ["จีน"], "Burmese": ["พม่า"], "Vietnamese": ["เวียดนาม/ญวน/แกว"], "Lao": ["ลาว"],
    "Cambodia": ["เขมร"], "Korean": ["เกาหลี"], "Japanese": ["ญี่ปุ่น"],
    "Tagalog/Filipino": ["ตากาล็อค/ฟิลิปินโน"], "Bengali/Banca Lee/Bangladesh": ["เบงกาลี/บังกาลี"],
    "Malaysia": ["มาเลเซีย"], "Indonesia": ["อินโดนีเซีย"], "India/Hindi": ["อินเดีย/ฮินดี"],
    "Arab": ["อาหรับ"], "English": ["อังกฤษ"], "German": ["เยอรมัน"], "Greek": ["กรีก"],
    "Spanish": ["สเปน"], "Polish": ["โปล (โปแลนด์)"], "Portugal": ["ปอร์ตุเกส"],
    "Russian": ["รัสเซีย"], "Swedish": ["สวีดิช"], "Finnish": ["ฟินแลนด์"], "French": ["ฝรั่งเศส"],
    "Danish": ["เดนมาร์ก"], "Italian": ["อิตาเลี่ยน"], "Hungarian": ["ฮังกาเรียน"],
    "Mexican": ["เม็กซิกัน"], "Cuban": ["คิวบัน"], "Africa": ["แอฟริกา"],
}
SPLIT = ROWS[1:4]
LANGS = ROWS[4:]

VAL = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+|-)$")   # Phangnga prints some without commas
LATIN = re.compile(r"[A-Za-z]")
THAI = re.compile(r"[฀-๿]")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = RAW / "_download.pdf"
    for key, (name, fn) in REPORTS.items():
        out = RAW / f"{key.replace('/', '_')}.pdf"
        if out.exists():
            continue
        req = urllib.request.Request(BASE + fn, headers=UA)
        with urllib.request.urlopen(req, timeout=180) as r:
            tmp.write_bytes(r.read())
        doc = fitz.open(tmp)
        pages = table_pages(doc)
        if not pages:
            raise SystemExit(f"{key} {name}: no language table in {fn}")
        # keep the language pages plus the first text page that names the province
        keep = sorted(set(pages) | {name_page(doc)})
        small = fitz.open()
        for p in keep:
            small.insert_pdf(doc, from_page=p, to_page=p)
        small.set_metadata({"title": f"NSO 2010 census, {name}: pages {keep} of {fn}",
                            "subject": BASE + fn})
        small.save(out)
        doc.close()
        print(f"  {key:7s} {name:20s} {fn}  pages {keep}")
        time.sleep(0.5)
    tmp.unlink(missing_ok=True)


def table_pages(doc):
    """The pages of the language table: its title, and some of its rows (not the contents page).
    Phuket's first page has no Karen row and its English row is on the second page."""
    out = []
    for i in range(doc.page_count):
        t = doc[i].get_text()
        if ("usual languages spoken at home" in t or "ภาษาที่ใช้พูดในครัวเรือน" in t) and                 sum(k in t for k in ("Only Thai", "Morn", "Japanese", "English", "Karen",
                                     "Burm", "Africa")) >= 2:
            out.append(i)
    return out


def name_page(doc):
    """The first page whose text says จังหวัด (province) or ราชอาณาจักร (kingdom)."""
    for i in range(min(40, doc.page_count)):
        t = doc[i].get_text()
        if "จังหวัด" in t or "ราชอาณาจักร" in t:
            return i
    return 2


def _visual_words(page):
    """Words in the page's visual frame (the table pages are stored rotated 90 degrees).

    Surat Thani's pages carry two text layers, one with "3,063" and one with "3" and "063": a
    word lying inside another word's box is dropped."""
    m = page.rotation_matrix
    ws = []
    for w in page.get_text("words"):
        r = fitz.Rect(w[:4]) * m
        ws.append((r.x0, r.y0, r.x1, r.y1, w[4]))
    out = []
    for i, a in enumerate(ws):
        inside = any(j != i and b[0] - 0.5 <= a[0] and a[2] <= b[2] + 0.5 and b[1] - 0.5 <= a[1]
                     and a[3] <= b[3] + 0.5 and (b[2] - b[0] > a[2] - a[0] or (b[:4] == a[:4] and j < i))
                     for j, b in enumerate(ws))
        if not inside:
            out.append(a)
    return out


def _rows_of_page(page, prev=None):
    """-> [(y, thai, english, {col: value})] for every visual line of the table body.

    Read by position, not by text order: the PDFs' text order wanders (a value can come after
    the label it belongs to), and a few cells are missing from the text layer altogether."""
    words = _visual_words(page)
    # columns: the right edges of the figures (right-aligned), clustered; a cluster needs eight
    # figures, which leaves out the "6" of "Table 6" and the "5" of "5 years of age". Columns
    # are 50pt apart, but one column's figures can wander by 6pt (Mukdahan).
    edges = sorted(w[2] for w in words if VAL.match(w[4]) and w[4] != "-")
    groups = []
    for x in edges:
        if groups and x - groups[-1][-1] < 12:
            groups[-1].append(x)
        else:
            groups.append([x])
    spans = [(g[0], g[-1]) for g in groups if len(g) >= 8]
    if len(spans) not in (3, 9) and prev:
        spans = prev     # a continuation page of mostly dashes: the first page's columns
    cols = [b for a, b in spans]   # right-aligned: the common edge is the cluster's last
    if len(cols) not in (3, 9):
        raise SystemExit(f"page {page.number}: {len(cols)} columns of figures")
    hy = max(w[3] for w in words if w[4] in ("Male", "Female"))
    body = [w for w in words if w[1] > hy]
    lines = {}
    for w in sorted(body, key=lambda w: w[1]):
        cy = (w[1] + w[3]) / 2
        key = next((k for k in lines if abs(k - cy) < 4), cy)
        lines.setdefault(key, []).append(w)
    out = []
    left, right = cols[0] - 80, cols[-1] + 10
    for y in sorted(lines):
        ws = sorted(lines[y], key=lambda w: w[0])
        vals, where, th, en = {}, {}, [], []
        for w in ws:
            t = w[4]
            if VAL.match(t) and left < w[2] < right:
                c = min(range(len(cols)), key=lambda i: abs(cols[i] - w[2]))
                off = 0 if spans[c][0] - 2 <= w[2] <= spans[c][1] + 2 else abs(cols[c] - w[2])
                if off > (15 if t == "-" else 4):
                    # a hyphen inside a label ("Lao - krung"), or a stray "0" printed between
                    # columns (Lamphun's subtotal line); logged, and the identities catch harm
                    if t != "-":
                        FILLED.append((page.parent.name[-8:], "stray figure ignored", t))
                    continue
                v = 0 if t == "-" else int(t.replace(",", ""))
                if c in vals:
                    # two figures in one cell: Lamphun prints a stray "0" beside its subtotal's;
                    # keep the one nearer the column's edge
                    if abs(w[2] - cols[c]) < where[c]:
                        FILLED.append((page.parent.name[-8:], "stray figure ignored", vals[c]))
                        vals[c], where[c] = v, abs(w[2] - cols[c])
                    else:
                        FILLED.append((page.parent.name[-8:], "stray figure ignored", t))
                    continue
                vals[c], where[c] = v, abs(w[2] - cols[c])
            elif THAI.search(t):
                th.append(t)
            elif LATIN.search(t):
                en.append(t)
        out.append((y, " ".join(th), " ".join(en), vals, len(cols)))
    return out, spans


FILLED = []


def _fill(path, canon, vals, width):
    """Complete a row with a cell or two missing from the text layer, by total = male + female
    within each area block and whole = municipal + non-municipal. Never more than three."""
    v = [vals.get(i) for i in range(width)]
    missing = [i for i, x in enumerate(v) if x is None]
    for _ in range(4):
        for blk in range(0, width, 3):
            t, m, f = v[blk:blk + 3]
            if [t, m, f].count(None) == 1:
                if t is None:
                    v[blk] = m + f
                elif m is None:
                    v[blk + 1] = t - f
                else:
                    v[blk + 2] = t - m
        if width == 9:
            for j in range(3):
                a, b, c = v[j], v[j + 3], v[j + 6]
                if [a, b, c].count(None) == 1:
                    if a is None:
                        v[j] = b + c
                    elif b is None:
                        v[j + 3] = a - c
                    else:
                        v[j + 6] = a - b
    if None in v or len(missing) > 3:
        raise SystemExit(f"{path.name}: {canon}: cannot complete {vals}")
    if missing:
        FILLED.append((path.stem, canon, missing))
    return v


def parse(path):
    """-> [(canonical label, label_th, [9 ints])] in printed order, and the report's text."""
    doc = fitz.open(path)
    lines = []
    spans = None
    for p in table_pages(doc):
        got, spans = _rows_of_page(doc[p], spans)
        lines += got
    # a row whose figures straddle two visual lines (Nakhon Sawan prints one cell a line high)
    merged = []
    for ln in lines:
        if merged:
            y0, t0, e0, v0, w0 = merged[-1]
            if 0 < len(v0) < w0 and ln[3] and not set(v0) & set(ln[3]):
                merged[-1] = (ln[0], (t0 + " " + ln[1]).strip(), (e0 + " " + ln[2]).strip(),
                              {**v0, **ln[3]}, w0)
                continue
            if 0 < len(v0) < w0 and len(ln[3]) == w0 and all(ln[3][c] == x for c, x in v0.items()):
                # a cell printed twice, once a line high (Nakhon Sawan): keep the full line's
                FILLED.append((path.stem, "duplicate cell dropped", sorted(v0)))
                merged[-1] = (ln[0], (t0 + " " + ln[1]).strip(), (e0 + " " + ln[2]).strip(),
                              ln[3], w0)
                continue
        merged.append(ln)
    # A label wrapped over two lines carries its figures on the first line (Lamphun) or the
    # second (most). Nan prints them on both; Lamphun prints zeros on the second. When both lines
    # have figures, the second must repeat the first or be all zeros, and the first is kept.
    lines = []
    skip = False
    for n, ln in enumerate(merged):
        if skip:
            lines.append((ln[0], ln[1], ln[2], {}, ln[4]))
            skip = False
            continue
        nxt = merged[n + 1] if n + 1 < len(merged) else None
        wrap = ln[2].endswith(",") or ln[2] == "Others language"
        if wrap and ln[3] and nxt and nxt[3]:
            if not any(ln[3].values()):
                FILLED.append((path.stem, "zeros on the first line of a wrapped label dropped",
                               ln[2]))
                lines.append((ln[0], ln[1], ln[2], {}, ln[4]))
                continue
            if nxt[3] != ln[3] and any(nxt[3].values()):
                raise SystemExit(f"{path.name}: figures on both lines of {ln[2]!r} disagree")
            FILLED.append((path.stem, "second line of a wrapped label repeats or is zeros",
                           ln[2]))
            skip = True
        lines.append(ln)
    out, i, subtotal, th, en = [], 0, None, [], []
    for n, (y, t, e, vals, width) in enumerate(lines):
        if t:
            th.append(t)
        if e:
            en.append(e)
        if not vals:
            # a label line with no figures: the heading "Other languages", or the first line
            # of a wrapped label, which joins the next line's
            if e and not (e.endswith(",") or e.startswith("Others language")):
                th, en = [], []
            continue
        label_en = " ".join(en)
        if not e and n + 1 < len(lines) and not lines[n + 1][3]:
            label_en = (label_en + " " + lines[n + 1][2]).strip()   # English on the next line
        half = label_en.split()
        if len(half) % 2 == 0 and half[:len(half) // 2] == half[len(half) // 2:]:
            label_en = " ".join(half[:len(half) // 2])    # a label printed twice ("Morn Morn")
        if re.match(r"^Others? languages$", label_en):
            # some reports print the subtotal of the language rows on the heading; kept for
            # the record, not trusted (Chiang Mai's disagrees with its own rows)
            subtotal = vals.get(0)
            HEADINGS[path.stem] = [vals.get(c, 0) for c in range(width)] + ([0] * 6 if width == 3 else [])
            th, en = [], []
            continue
        th_label = " ".join(th)

        def is_row(j):
            canon, pat = ROWS[j]
            return bool(re.search(pat, label_en, re.I) or re.search(pat, e.strip(), re.I)
                        or th_label.replace(" ", "") in [k.replace(" ", "") for k in TH_LABELS.get(canon, [])])

        # rows are in a fixed order, but a report may leave one out (Uttaradit has no Polish)
        j = next((j for j in range(i, len(ROWS)) if is_row(j)), None)
        if j is None:
            if i >= len(ROWS):
                raise SystemExit(f"{path.name}: more rows than expected at {label_en!r}")
            # a garbled English label ("P Polish li h"): accept it at its place in the order
            # when it or the Thai label is close to the expected one
            canon = ROWS[i][0]
            r_en = difflib.SequenceMatcher(None, label_en.lower(), canon.lower()).ratio()
            r_th = max((difflib.SequenceMatcher(None, th_label, k).ratio()
                        for k in TH_LABELS.get(canon, [])), default=0)
            if max(r_en, r_th) < 0.6:
                raise SystemExit(f"{path.name}: row {i} expected {canon!r}, read {label_en!r} / {th_label!r}")
            FILLED.append((path.stem, "label garbled, matched by order", f"{label_en!r} -> {canon}"))
            j = i
        for k in range(i, j):
            FILLED.append((path.stem, "row not printed, taken as zero", ROWS[k][0]))
            out.append((ROWS[k][0], "", [0] * 9))
        canon = ROWS[j][0]
        v = _fill(path, canon, vals, width)
        if abs(v[0] - v[1] - v[2]) > 2 and width == 9 and                 abs(v[1] + v[2] - v[3] - v[6]) <= 2 and abs(v[3] - v[4] - v[5]) <= 2:
            # a misprinted total whose eight other cells agree with each other (Uttaradit's
            # Vietnamese: 7,204 beside eight zeros, and the language rows then exceed the
            # province's other-language households by exactly 7,204)
            FILLED.append((path.stem, f"misprinted total {v[0]} replaced by male + female", canon))
            v[0] = v[1] + v[2]
        if width == 3:
            # Bangkok's table has no municipal / non-municipal columns: the BMA is all
            # municipal area, so the whole is written as municipal
            v = v + v + [0, 0, 0]
        out.append((canon, th_label, v))
        th, en, i = [], [], j + 1
    for k in range(i, len(ROWS)):
        FILLED.append((path.stem, "row not printed, taken as zero", ROWS[k][0]))
        out.append((ROWS[k][0], "", [0] * 9))
    gone = [f for f in FILLED if f[0] == path.stem and f[1].startswith("row not printed")]
    if any(r in [c for c, _ in ROWS[:4]] for _, _, r in gone):
        raise SystemExit(f"{path.name}: the total or the three-way split is missing")
    if len(out) != len(ROWS):
        raise SystemExit(f"{path.name}: read {i} rows of {len(ROWS)}")
    if subtotal is not None:
        SUBTOTALS[path.stem] = (subtotal, sum(v[0] for _, _, v in out[4:]))
    text = " ".join(doc[p].get_text() for p in range(doc.page_count))
    text = text.replace("ํา", "ำ")   # sara am set as nikhahit + sara aa
    return out, text


SUBTOTALS = {}
RESID = {}
HEADINGS = {}
DIFF = {"sex": 0, "area": 0, "split": 0, "langs": 0, "kingdom": 0}


def close(kind, a, b, tol):
    """The tables are weighted estimates rounded to the person, so an identity can be off by
    the rounding of its terms; never by more than `tol` (one per term)."""
    DIFF[kind] = max(DIFF[kind], abs(a - b))
    return abs(a - b) <= tol


def check_table(key, rows):
    """Each row's total must equal male + female or municipal + non-municipal; a report can
    misprint one other cell (Chiang Rai's municipal Thaikueng, Saraburi's male Arab), which is
    logged, not used. The language rows may fall short of Thai and other + other only: the
    households that named no other language (81,519 in the kingdom), never exceed it."""
    d = {c: v for c, _, v in rows}
    for c, _, v in rows:
        sex = close("sex", v[0], v[1] + v[2], 2)
        area = close("area", v[0], v[3] + v[6], 2)
        assert sex or area, (key, c, v)
        if not (sex and area and all(close("sex", v[j], v[j + 1] + v[j + 2], 2) for j in (3, 6))):
            FILLED.append((key, "a misprinted sex or area cell, total kept", c))
    assert close("split", d["Total"][0], sum(d[c][0] for c, _ in SPLIT), 3), (key, "split")
    other = d["Thai and other languages"][0] + d["Only other languages"][0]
    s = sum(d[c][0] for c, _ in LANGS)
    assert key in [k for k, _ in FROM_KINGDOM] or s <= other + len(LANGS), (key, "language rows exceed the other-language households",
                                     other, s)
    return other - s


# Repairs, each found by the kingdom check below and each confirmed by it once made:
#   * Phitsanulok and Chumphon print the figures of the first block of language rows (Karen to
#     Bengali) one line high, the first on the "Other languages" heading: Chumphon's "Chinese"
#     row holds its 32,397 Burmese. Moved down a row, the kingdom's Hmong, Lao Khrang, Malay,
#     local-and-hill-tribe, Chinese, Burmese, Korean and Japanese rows come right.
#   * Narathiwat's English row repeats its "Other languages" subtotal (569,967). Replaced by the
#     kingdom's English less the other 76 provinces, column by column.
SHIFTED = ("365", "586")
FROM_KINGDOM = (("596", "English"),)


def shift_down(key, rows):
    first = [c for c, _ in LANGS].index("Karen")
    last = [c for c, _ in LANGS].index("Malaysia")
    vals = [HEADINGS[key]] + [v for _, _, v in rows[4 + first:4 + last]]
    for n, v in enumerate(vals):
        c, th, _ = rows[4 + first + n]
        rows[4 + first + n] = (c, th, list(v))
    FILLED.append((key, "language rows Karen-Malaysia moved down one line", ""))


def main():
    if "--fetch" in sys.argv:
        fetch()
    lookup = {}
    rd_lookup = Path(__file__).resolve().parents[2] / "religiondots" / "data" / "geo" / "th" / "th_lookup.csv"
    with open(rd_lookup, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lookup[r["unit"]] = r
    units = {k.split("/")[0] for k in REPORTS if k != "TH"}
    assert units == set(lookup), ("units", sorted(units ^ set(lookup)))

    tables, unspec = {}, {}
    for key, (name, _) in REPORTS.items():
        rows, text = parse(RAW / f"{key.replace('/', '_')}.pdf")
        if key in SHIFTED:
            shift_down(key, rows)
        unspec[key] = check_table(key, rows)
        if key in lookup:
            assert lookup[key]["name_th"] == OFFICIAL.get(key, name), (key, lookup[key]["name_th"])
        tables[key] = {c: (th, v) for c, th, v in rows}
    print(f"read {len(tables) - 1} provincial tables and the kingdom's; each row's total agrees "
          "with its sex or its area columns, and each table's three-way split adds to its total")

    # The file-to-province pairing: each table's total against the census total religiondots
    # read for the same changwat from NSO's provincial indicator sheets (a different document).
    # Rayong and Ratchaburi's language tables fall short of theirs, by exactly what the kingdom
    # check below finds missing.
    rd = Path(__file__).resolve().parents[2] / "religiondots" / "data" / "normalized" / "th.csv"
    sheet = {}
    with open(rd, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.search(r"census_total=(\d+)", r.get("note") or "")
            if r["geo_level"] == "province" and m:
                sheet[r["geo_id"]] = int(m.group(1))
    short = {}
    for key in lookup:
        t = tables[key]["Total"][1][0]
        if abs(t - sheet[key]) > 12:
            short[key] = sheet[key] - t
    assert set(short) == {"221", "270"}, short
    print(f"74 of 76 tables' totals match the indicator sheets' to 12 people; Rayong and "
          f"Ratchaburi, short by {short['221']:,} and {short['270']:,}")

    # Bueng Kan was still part of Nong Khai on census day, and Nong Khai's report includes it
    # (821,526 is the whole pre-2011 changwat, religiondots' figure too): it is a check, not added
    bk, nk = tables.pop("443/bk"), tables["443"]
    assert all(bk[c][1][0] < nk[c][1][0] for c, _ in ROWS[:4]), "Bueng Kan exceeds Nong Khai"
    over = [(c, bk[c][1][0], nk[c][1][0]) for c, _ in ROWS if bk[c][1][0] > nk[c][1][0]]
    print(f"Bueng Kan's own report ({bk['Total'][1][0]:,}) lies inside Nong Khai's "
          f"({nk['Total'][1][0]:,}); not added. Its language rows exceed Nong Khai's in {over}")

    kingdom = {c: v for c, (_, v) in tables["TH"].items()}
    provs = [k for k in tables if k != "TH"]
    for key, c in FROM_KINGDOM:
        rest = [sum(tables[k][c][1][j] for k in provs if k != key) for j in range(9)]
        new = [kingdom[c][j] - rest[j] for j in range(9)]
        FILLED.append((key, f"{c} {tables[key][c][1][0]:,} replaced from the kingdom", new[0]))
        tables[key][c] = (tables[key][c][0], new)
    for key in provs:
        d = {c: v for c, (_, v) in tables[key].items()}
        other = d["Thai and other languages"][0] + d["Only other languages"][0]
        unspec[key] = other - sum(d[c][0] for c, _ in LANGS)
        assert unspec[key] >= -len(LANGS), (key, unspec[key])

    # the 76 provinces against the whole-kingdom table, total column
    print("provinces less kingdom, by row (total column):")
    for c, _ in ROWS:
        s = sum(tables[k][c][1][0] for k in provs)
        r = s - kingdom[c][0]
        RESID[c] = r
        # Rayong and Ratchaburi's tables count 95,150 fewer people than the kingdom gives them,
        # nearly all Thai only; every language row must be within 2% (or 300 people)
        tol = 100_000 if c in [x for x, _ in ROWS[:4]] else max(300, 0.02 * kingdom[c][0])
        assert abs(r) <= tol, (c, s, kingdom[c][0])
        if r:
            print(f"  {c:48s} {s:>11,} {kingdom[c][0]:>11,} {r:>+8,}")
    print(f"households naming another language but no row: kingdom {unspec['TH']:,}, provinces "
          f"{sum(unspec[k] for k in provs):,}")
    print("largest rounding gap in each identity:", DIFF)
    print("repairs and oddities:", len(FILLED))
    for f in FILLED:
        print("  ", f)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    acc = {}
    for key, rows in tables.items():
        unit = key.split("/")[0]
        for c, (th, v) in rows.items():
            a = acc.setdefault((unit, c), [th, 0, 0, 0])
            a[1] += v[0]
            a[2] += v[3]
            a[3] += v[6]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "label_th", "count",
                    "municipal", "nonmunicipal"])
        for (unit, c), (th, n, m, nm) in acc.items():
            level = "national" if unit == "TH" else "province"
            name = "Thailand" if unit == "TH" else lookup[unit]["name_en"].replace(" Province", "")
            w.writerow([unit, level, name, c, th, n, m, nm])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(acc)} rows")


if __name__ == "__main__":
    main()
