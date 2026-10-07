"""Read PBS 2023 census Table 1 (one PDF per province) into a list of units.

Table 1: "Area, population by sex, sex ratio, population density, urban population, household
size and annual growth rate, Census-2023". Each PDF runs province, then each district, then
that district's tehsils/talukas/sub-divisions, each as a bold row followed by RURAL and URBAN
rows. Twelve columns:

   1 name  2 area km2  3 2023 all sexes  4 male  5 female  6 transgender  7 sex ratio
   8 density  9 urban proportion  10 household size  11 POPULATION 2017  12 2017-23 growth

Column 11 is the 2017 census count re-tabulated on the 2023 units, which is what makes this
table the ideal pair for helper1m: both years on one boundary basis, at tehsil.

Read by geometry with PyMuPDF, as religiondots/sources/pk_2023.py reads Table 9: numbers are
right-aligned against the table's vertical rules, so a number's column is decided by its RIGHT
edge against the rules the header draws. Every row's cells are checked (all = male + female +
trans; rural + urban = all, for 2023 and 2017), so a misread digit fails loudly.
"""

import os
import re

import fitz

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
BASE = "https://www.pbs.gov.pk/wp-content/uploads/census_tables/tables/"

# (file, province name as used here)
FILES = [
    ("table_1_kp_districts.pdf", "Khyber Pakhtunkhwa"),
    ("table_1_punjab_districts.pdf", "Punjab"),
    ("table_1_sindh_districts.pdf", "Sindh"),
    ("table_1_balochistan_districts.pdf", "Balochistan"),
    ("table_1_islamabad.pdf", "Islamabad"),
]

NCOL = 12
FOOTNOTES = set()
_NUM = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$|^-?\d+(\.\d+)?$|^-$")

# Block headers that are districts without saying DISTRICT (religiondots pk.md §9.2, trap 2).
DISTRICT_HEADERS = {"MALAKAND PROTECTED AREA"}


def fetch(raw_dir):
    import requests

    os.makedirs(raw_dir, exist_ok=True)
    for name, _ in FILES:
        dest = os.path.join(raw_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            continue
        r = requests.get(BASE + name, headers=UA, timeout=300)
        r.raise_for_status()
        if r.content[:5] != b"%PDF-" or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit(f"{name}: not a whole PDF")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  fetched {name} ({len(r.content):,} bytes)")


def _val(t):
    if t == "-":
        return 0
    t = t.replace(",", "")
    return float(t) if "." in t else int(t)


def _spans(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = " ".join(s["text"].split())
                if t:
                    x0, y0, x1, y1 = s["bbox"]
                    out.append({"t": t, "x0": x0, "x1": x1, "y": (y0 + y1) / 2,
                                "bold": "Bold" in s["font"]})
    return out


def _rules(page):
    """x of the table's vertical rules, drawn as thin rectangles in the header (13 of them)."""
    xs = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "re" and it[1].width < 2 and it[1].y0 < 90 and it[1].height > 15:
                xs += [it[1].x0, it[1].x1]
            elif it[0] == "l" and abs(it[1].x - it[2].x) < 0.5 and min(it[1].y, it[2].y) < 90:
                xs.append(it[1].x)
    xs.sort()
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] < 1.5:
            groups[-1].append(x)
        else:
            groups.append([x])
    rules = [sum(g) / len(g) for g in groups]
    return rules if len(rules) == NCOL + 1 else None


def read_pdf(path):
    """Returns rows: dicts with name, kind ('unit'|'RURAL'|'URBAN'), cells (list of 11), page."""
    doc = fitz.open(path)
    rows = []
    rules = None
    for pno in range(doc.page_count):
        page = doc[pno]
        r = _rules(page)
        if r is not None:
            rules = r
        if rules is None:
            raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: no column rules")
        sp = _spans(page)
        # the header's column-number row: the one y carrying all of 1..12 (a data row can
        # hold a bold '1' too, which is how the first version lost MERYAN TEHSIL)
        want = {str(i) for i in range(1, NCOL + 1)}
        hdr = [s["y"] for s in sp if s["t"] == "12" and s["y"] < 100
               and want <= {t["t"] for t in sp if abs(t["y"] - s["y"]) < 1}]
        if len(hdr) != 1:
            raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: column-number row {hdr}")
        top = hdr[0] + 2
        body = [s for s in sp if s["y"] > top]
        # the last page carries a footnote, starting at a '*' and running across the page;
        # everything from its first line down is kept out of the table
        foot = [s["y"] for s in body if s["x1"] - s["x0"] > 250 and not _NUM.match(s["t"])]
        if foot:
            cut = min(foot) - 1
            FOOTNOTES.update(s["t"] for s in body if s["y"] >= cut and s["t"] != "*")
            body = [s for s in body if s["y"] < cut]
        labels = [s for s in body if s["x1"] <= rules[1] + 1 and not _NUM.match(s["t"])]
        nums = [s for s in body if _NUM.match(s["t"]) and s["x0"] > rules[1] - 1]
        stray = [s for s in body if s not in labels and s not in nums]
        if stray:
            raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: unplaced text {stray[0]}")
        # A label sits on its numbers' line, except a long unit name that PBS squeezed onto
        # two lines (e.g. DARRA ADAM KHEL / SUB-DIVISION): those pieces carry no numbers of
        # their own and are joined, in reading order, to the nearest unit (non RURAL/URBAN)
        # label within 8 pt.
        # Number rows first (clustered on y), then labels onto them.
        used = set()
        nums_sorted = sorted(nums, key=lambda s: s["y"])
        clusters = []
        for s in nums_sorted:
            if clusters and s["y"] - clusters[-1][-1]["y"] < 1.5:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        slots = [{"y": sum(s["y"] for s in c) / len(c), "row": c, "parts": []} for c in clusters]
        loose = []
        for lab in sorted(labels, key=lambda s: (s["y"], s["x0"])):
            hit = [sl for sl in slots if abs(sl["y"] - lab["y"]) < 2.5]
            if hit:
                hit[0]["parts"].append(lab)
            else:
                loose.append(lab)
        for lab in loose:
            if not re.search(r"[A-Za-z]", lab["t"]):
                FOOTNOTES.add(lab["t"])          # the footnote's '*' marker
                continue
            # an unlabelled number row takes it first; otherwise it continues a unit name
            cands = [sl for sl in slots if not sl["parts"] and abs(sl["y"] - lab["y"]) < 8]
            if not cands:
                cands = [sl for sl in slots if sl["parts"] and abs(sl["y"] - lab["y"]) < 8
                         and sl["parts"][0]["t"] not in ("RURAL", "URBAN")]
            if not cands:
                raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: label {lab['t']!r} "
                                 f"has no number row near it")
            min(cands, key=lambda sl: abs(sl["y"] - lab["y"]))["parts"].append(lab)
        merged = []
        for sl in slots:
            if not sl["parts"]:
                raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: number row at "
                                 f"y={sl['y']:.1f} has no label")
            parts = sorted(sl["parts"], key=lambda s: (s["y"], s["x0"]))
            lab = dict(parts[0], t=" ".join(p["t"] for p in parts))
            merged.append([lab, sl["row"]])
        for lab, row in merged:
            for s in row:
                used.add(id(s))
            cells = [None] * (NCOL - 1)
            for s in row:
                col = sum((s["x1"] - 1.0) > x for x in rules)   # 1 = name ... 12 = growth
                k = col - 2
                if not 0 <= k < NCOL - 1 or cells[k] is not None:
                    raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: {lab['t']} cell "
                                     f"{s['t']!r} lands in column {col}")
                cells[k] = _val(s["t"])
            name = " ".join(lab["t"].split())
            kind = name if name in ("RURAL", "URBAN") else "unit"
            rows.append({"name": name, "kind": kind, "cells": cells, "page": pno + 1,
                         "bold": lab["bold"]})
        left = [s for s in nums if id(s) not in used]
        if left:
            raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: {len(left)} numbers on no "
                             f"labelled row, first {left[0]['t']!r} y={left[0]['y']:.1f}")
    return rows


def level_of(name, first):
    if first:
        return "province"
    if name.endswith(" DISTRICT") or name in DISTRICT_HEADERS:
        return "district"
    return "tehsil"


# cell indexes into `cells` (column number - 2)
AREA, POP, MALE, FEMALE, TRANS, SEXR, DENS, URBP, HH, POP17, GR = range(11)


def read_all(raw_dir):
    """Units: dicts province, level, name, district, area, pop23, pop17, rural/urban."""
    units = []
    errors = []
    for fname, prov in FILES:
        rows = read_pdf(os.path.join(raw_dir, fname))
        i = 0
        first = True
        district = None
        while i < len(rows):
            u = rows[i]
            if u["kind"] != "unit":
                errors.append(f"{fname} p{u['page']}: {u['kind']} row with no unit above")
                i += 1
                continue
            sub = {}
            j = i + 1
            while j < len(rows) and rows[j]["kind"] in ("RURAL", "URBAN"):
                sub[rows[j]["kind"]] = rows[j]["cells"]
                j += 1
            c = u["cells"]
            lv = level_of(u["name"], first)
            if prov == "Islamabad" and first:
                lv = "province"
            if lv == "district":
                district = u["name"]
            for k in (POP, MALE, FEMALE, TRANS, POP17):
                if c[k] is None:
                    c[k] = 0
            if c[POP] != c[MALE] + c[FEMALE] + c[TRANS]:
                errors.append(f"{fname} {u['name']}: {c[POP]:,} != m+f+t "
                              f"{c[MALE] + c[FEMALE] + c[TRANS]:,}")
            if set(sub) == {"RURAL", "URBAN"}:
                for k in (POP, POP17):
                    a, b = sub["RURAL"][k] or 0, sub["URBAN"][k] or 0
                    if c[k] != a + b:
                        errors.append(f"{fname} {u['name']} col{k + 2}: rural+urban "
                                      f"{a + b:,} != {c[k]:,}")
            else:
                errors.append(f"{fname} {u['name']}: rural/urban rows {sorted(sub)}")
            units.append({"province": prov, "level": lv, "name": u["name"],
                          "district": district if lv == "tehsil" else (u["name"] if lv == "district" else None),
                          "area": c[AREA], "pop23": c[POP], "pop17": c[POP17], "growth": c[GR],
                          "urban23": (sub.get("URBAN") or [0] * 11)[POP] or 0,
                          "file": fname, "page": u["page"]})
            first = False
            i = j
    return units, errors
