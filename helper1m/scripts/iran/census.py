"""Readers for SCI's settlement (abadi) workbooks of the 1390 and 1395 censuses.

Both files list, per province, every county, district (bakhsh), rural district (dehestan), city
and village with its population. Villages of 3 households or fewer print `*` (no count); their
people are inside every parent row. Non-settled (nomadic) households are counted only at county
level (1395 notes sheet: "اطلاعات خانوارهای غیرساکن تا سطح شهرستان لحاظ شده است"), so a county
row can exceed the sum of its districts.

read95(nn) / read90(nn) return a list of dicts with keys
    kind   'ostan' | 'county' | 'bakhsh' | 'dehestan' | 'city' | 'abadi'
    ost, cty, bkh, unit   codes (strings; unit = the 4-digit dehestan or city code)
    abadi  6-digit village code (abadi rows), name, pop (int, or None for '*')
"""
import re
import unicodedata
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "iran" / "raw"
REVISED_90 = {"02", "04", "07"}

_FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا",
                       "آ": "ا", "ؤ": "و", "ۀ": "ه", "‌": "", "‏": "", "‎": "",
                       "ـ": ""})


def fold(s):
    """Persian name key: Arabic letter forms folded, alef-madda to alef, no spaces/ZWNJ/hyphens."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).translate(_FOLD)
    return re.sub(r"[\s\-\.\(\)،,]+", "", s)


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    v = str(v).strip()
    if v in ("*", ""):
        return None
    return int(float(v))


def path95(nn):
    return RAW / "abadi95" / f"CN95_HouseholdPopulationVillage_{nn}_r.xlsx"


def path90(nn):
    return RAW / "abadi90" / (f"os{nn}-r.xls" if nn in REVISED_90 else f"os{nn}.xls")


KIND95 = {"1": "ostan", "2": "county", "3": "bakhsh", "4": "dehestan", "5": "city",
          "6": "abadi", "8": "abadi"}


def read95(nn):
    import openpyxl

    wb = openpyxl.load_workbook(path95(nn), read_only=True)
    ws = wb.worksheets[0]
    out = []
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 2:
            continue
        r = [x.strip() if isinstance(x, str) else x for x in r]
        if r[0] in (None, ""):
            continue
        rec = str(r[11]).strip()
        kind = KIND95[rec]
        if kind == "city" and r[12] not in (None, ""):
            continue  # a zone (mantaqe) of a zoned city; the city's own row carries the total
        out.append({"kind": kind, "ost": r[0], "cty": r[2] or "", "bkh": r[4] or "",
                    "unit": r[6] or "", "abadi": r[9] or "",
                    "name": {"ostan": r[1], "county": r[3], "bakhsh": r[5], "dehestan": r[7],
                             "city": r[7], "abadi": r[10]}[kind],
                    "ctyname": r[3], "bkhname": r[5], "unitname": r[7],
                    "rec": rec, "zone": r[12], "swdiv": r[13],
                    "pop": _num(r[15]), "hh": _num(r[14])})
    wb.close()
    return out


def read90(nn):
    import xlrd

    sh = xlrd.open_workbook(path90(nn)).sheet_by_index(0)
    out = []
    for i in range(sh.nrows):
        r = [str(x).strip() for x in sh.row_values(i)]
        code = r[0]
        if not re.fullmatch(r"\d+", code):
            continue
        n = len(code)
        city = r[8]
        if n == 2:
            kind, name = "ostan", r[5]
        elif n == 4:
            kind, name = "county", r[6]
        elif n == 6:
            # bakhsh 99 is the county's non-settled (nomadic) population
            kind, name = ("nomad", "") if code[4:6] == "99" else ("bakhsh", r[7])
        elif n == 10:
            kind, name = ("city", city) if city else ("dehestan", r[9])
        elif n == 19:
            kind, name = "abadi", r[10]
        else:
            raise ValueError(f"os{nn} row {i}: address {code!r} of length {n}")
        out.append({"kind": kind, "ost": code[:2], "cty": code[2:4] if n >= 4 else "",
                    "bkh": code[4:6] if n >= 6 else "", "unit": code[6:10] if n >= 10 else "",
                    "hoze": code[10:13] if n == 19 else "", "abadi": code[13:19] if n == 19 else "",
                    "name": name, "ctyname": r[6], "bkhname": r[7],
                    "unitname": city or r[9], "pop": _num(r[1]), "hh": _num(r[4])})
    return out
