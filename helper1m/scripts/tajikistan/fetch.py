"""Tajikistan population by city/district and region for helper1m.

Sources (all from the Agency on Statistics, www.stat.tj, CC BY 4.0; downloaded
into helper1m/data/tajikistan/raw/ by download.py):

  Annual bulletin "Шумораи аҳолии Ҷумҳурии Тоҷикистон то 1 январи соли 20XX /
  Численность населения Республики Таджикистан на 1 января 20XX года", the
  table "Численность постоянного населения поселков, городов, районов и
  областей" (thousands, one decimal). Each issue gives 1 January of the year
  before and of its own year:
      bulletin_2025.pdf            -> 2024, 2025
      bulletin_2024.pdf            -> 2023, 2024
      bulletin_2022_corrected.pdf  -> 2021, 2022
  Where two issues give the same year, the later issue wins (they agree for 2024).

  Census 2020, volume I, table 1 (Russian): permanent population by region, city
  and district for 2010 and 2020, exact counts, on 2020 boundaries.

Writes helper1m/data/tajikistan/population.csv (code, level, year, pop):
  level 2 = 65 cities and districts (codes from units.py)
  level 1 = 5 regions, the sum of their level-2 units
"""
import csv
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from units import CENSUS_OFF_BASIS, REGIONS, UNITS  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "tajikistan" / "raw"
OUT = HELPER / "data" / "tajikistan" / "population.csv"

BULLETINS = [  # oldest first; later issues overwrite shared years
    ("bulletin_2022_corrected.pdf", 2021, 2022),
    ("bulletin_2024.pdf", 2023, 2024),
    ("bulletin_2025.pdf", 2024, 2025),
]
CENSUS = "census2020_vol1_table1_ru.pdf"

# Published national totals (1 January, thousands) from the bulletins' own
# region series table, and the census national count; used as checks.
NATIONAL = {2021: 9716.8, 2022: 9886.8, 2023: 10078.4, 2024: 10288.3, 2025: 10508.5}
CENSUS_NATIONAL = {2010: 7564502, 2020: 9657005}

NUM = re.compile(r"^\d+(?:,\d+)?$")
URBAN = re.compile(r"мањалњои\s*шањрї")
RURAL = re.compile(r"дењот$")

# Hand fixes, (file, code, year) -> thousands, applied after the 2% rule.
#  Murghob 2025: unit row printed as "17" without a decimal; its sub-rows
#    (Murghob settlement 8.5, rural 8.1) give 16.6.
#  Isfara 2025: unit row is the city proper's 56.7, and the urban row repeats
#    2024's 65.1; the settlements (Isfara 56.7, Shurob 3.1, Nurafshon 1.5,
#    Neftobod 4.4) plus rural 227.6 give 293.3, which also brings Sughd's units
#    within rounding of the Sughd total.
OVERRIDES = {
    ("bulletin_2025.pdf", "TJ-GB-05", 2025): 16.6,
    ("bulletin_2025.pdf", "TJ-SU-02", 2025): 293.3,
}


# ---------------------------------------------------------------- bulletins
def page_rows(page, xnum=(120, 292)):
    """Rows of one table page: (tajik label, [numbers]). A row is anchored on a
    line of numbers; its label is the left-column text since the previous row."""
    spans = []
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            for sp in ln["spans"]:
                t = sp["text"].strip()
                if t:
                    spans.append((sp["bbox"][0], sp["bbox"][1], t))
    nums = [s for s in spans if xnum[0] <= s[0] < xnum[1] and NUM.match(s[2])]
    lines = []
    for y in sorted({round(s[1]) for s in nums}):
        if not lines or y - lines[-1] > 2:
            lines.append(y)
    rows, prev = [], None
    for y in lines:
        vals = sorted((s for s in nums if abs(s[1] - y) <= 2.5), key=lambda s: s[0])
        lo = prev if prev is not None else y - 12
        left = sorted((s for s in spans if s[0] < xnum[0] and lo + 3 < s[1] <= y + 3
                       and not NUM.match(s[2])), key=lambda s: (s[1], s[0]))
        label = re.sub(r"\s+", " ", " ".join(s[2] for s in left)).strip()
        rows.append((label, [float(v[2].replace(",", ".")) for v in vals]))
        prev = y
    return rows


def parse_bulletin(path, y1, y2):
    """{code: {y1: pop, y2: pop}} for units and regions, plus a list of notes."""
    doc = fitz.open(path)
    rows = []
    for page in doc:
        rows += page_rows(page)
    # the table starts at the national row with three values over 9,000
    start = next(i for i, (lab, v) in enumerate(rows)
                 if lab.startswith("Љумњурии Тољикистон") and len(v) >= 2 and v[0] > 9000)
    out, notes = {}, []
    names = {u[0]: u[1] for u in UNITS}
    cursor = start
    unit_idx = []  # (code, row index)
    region_hdr = {r[0]: re.compile(r[3]) for r in REGIONS}
    region_rows = {}
    for code, name, *_rest in UNITS:
        bul = re.compile(_rest[2], re.I)
        reg = code[:5]
        if reg not in region_rows:
            j = next(i for i in range(cursor, len(rows)) if region_hdr[reg].search(rows[i][0]))
            region_rows[reg] = j
            if reg != "TJ-DU":
                cursor = j + 1
        j = next((i for i in range(cursor, len(rows)) if bul.search(rows[i][0])), None)
        if j is None:
            sys.exit(f"{path.name}: unit {code} {name} not found")
        unit_idx.append((code, j))
        cursor = j + 1
    bounds = [j for _, j in unit_idx] + sorted(region_rows.values())
    for code, j in unit_idx:
        lab, v = rows[j]
        val = {y1: v[0], y2: v[1]}
        # sub-rows up to the next unit or region header
        nxt = min([b for b in bounds if b > j] + [len(rows)])
        sub = rows[j + 1:nxt]
        # A unit row that disagrees with its own urban + rural rows by more than
        # 2% is a typo in the unit row (Isfara 2025 prints the city proper's
        # 56.7 for the whole unit; Murghob 2025 prints 16.6 rounded to 17).
        # Smaller gaps are left alone: there the urban row is usually the one
        # that is off (Guliston's settlements add up to the unit row, not to
        # its urban row). Rows split by a page break lose their label, so only
        # units with both an urban and a rural row are checked.
        urb = [r for r in sub if URBAN.search(r[0])]
        rur = [r for r in sub if RURAL.search(r[0])]
        if urb and rur:
            for k, y in enumerate((y1, y2)):
                alt = urb[0][1][k] + rur[-1][1][k]
                if abs(alt - val[y]) > 0.02 * val[y]:
                    notes.append(f"{path.name} {code} {names[code]} {y}: unit row {val[y]} "
                                 f"but its urban + rural rows sum to {alt:.1f}; using the sum")
                    val[y] = round(alt, 1)
        for y in (y1, y2):
            fix = OVERRIDES.get((path.name, code, y))
            if fix is not None:
                notes.append(f"{path.name} {code} {names[code]} {y}: unit row {val[y]} "
                             f"replaced by hand with {fix}")
                val[y] = fix
        out[code] = val
    for reg, j in region_rows.items():
        out[reg] = {y1: rows[j][1][0], y2: rows[j][1][1]}
    return out, notes


# ---------------------------------------------------------------- census
def parse_census(path):
    """{code: {2010: n, 2020: n}} from census 2020 vol. I table 1."""
    doc = fitz.open(path)
    toks = [s.strip() for p in doc for s in p.get_text().split("\n") if s.strip()]
    skip = re.compile(r"^(ШУМОРА|Агент|\d+$|с\. 20|ҳар ду|ҷинс|оба пола|мардҳо|мужчины|занҳо|женщины)")
    runs, label, i = [], [], 0
    while i < len(toks):
        t = toks[i].replace(" ", "")
        if re.match(r"^(\d+|-)$", t):
            vals = []
            while i < len(toks) and re.match(r"^(\d+|-)$", toks[i].replace(" ", "")) and len(vals) < 6:
                vals.append(toks[i].replace(" ", ""))
                i += 1
            runs.append((" ".join(label[-4:]), vals))
            label = []
            continue
        if not skip.match(toks[i]):
            label.append(toks[i])
        i += 1
    out = {}
    targets = [(c, cen) for c, _n, _t, _o, _b, cen in UNITS] + [(r[0], r[4]) for r in REGIONS]
    for code, cen in targets:
        pat = re.compile(cen)
        hits = [v for lab, v in runs if pat.search(lab)
                and not re.search(r"Городское население|Сельское население", lab[-30:])]
        if code == "TJ-DU":
            hits = hits[:1]
        if len(hits) != 1:
            sys.exit(f"census: {code} /{cen}/ matched {len(hits)} rows")
        v = hits[0]
        out[code] = {2010: int(v[0]) if v[0] != "-" else None, 2020: int(v[3])}
    return out


def main():
    notes = []
    unit_pop = {u[0]: {} for u in UNITS}
    region_pub = {r[0]: {} for r in REGIONS}
    for fname, y1, y2 in BULLETINS:
        got, n = parse_bulletin(RAW / fname, y1, y2)
        notes += n
        for code, vals in got.items():
            target = unit_pop.get(code, region_pub.get(code))
            for y, v in vals.items():
                if y in target and abs(target[y] - round(v * 1000)) > 100:
                    notes.append(f"{fname} {code} {y}: {v} differs from the earlier issue's "
                                 f"{target[y] / 1000}; using the later issue")
                target[y] = round(v * 1000)

    census = parse_census(RAW / CENSUS)
    for code, vals in census.items():
        if code in unit_pop:
            if code in CENSUS_OFF_BASIS:
                continue
            for y, v in vals.items():
                if v is not None:
                    unit_pop[code][y] = v

    # ---------------- checks
    print("Region totals: sum of units vs the region row of the same table")
    years = sorted({y for v in unit_pop.values() for y in v})
    for y in [2021, 2022, 2023, 2024, 2025]:
        line = [f"  {y}:"]
        nat = 0
        for reg, *_ in REGIONS:
            s = sum(unit_pop[c][y] for c in unit_pop if c.startswith(reg))
            nat += s
            line.append(f"{reg} {s / 1000:,.1f}/{region_pub[reg][y] / 1000:,.1f}")
        line.append(f"| national {nat / 1000:,.1f} vs published {NATIONAL[y]:,.1f}")
        print(" ".join(line))
    print("Census: sum of units (all 65, Dushanbe and Rudaki on 2020 territory) vs published")
    for y in (2010, 2020):
        s = sum(census[c][y] or 0 for c in unit_pop)
        print(f"  {y}: units {s:,} vs national {CENSUS_NATIONAL[y]:,}")
        for reg, *_ in REGIONS:
            s = sum(census[c][y] or 0 for c in unit_pop if c.startswith(reg))
            print(f"     {reg}: units {s:,} vs region row {census[reg][y]:,}")
    print("Census 2020 (1 Oct) -> bulletin 1 Jan 2021, per unit; flags outside -2%..+4%:")
    for code, name, *_ in UNITS:
        c = census[code][2020]
        b = unit_pop[code][2021]
        ch = b / c - 1
        if not (-0.02 <= ch <= 0.04):
            print(f"  {code} {name}: census {c:,} -> 2021 {b:,} ({ch:+.1%})")
    print("Growth 2024 -> 2025 outside 0%..+4%:")
    for code, name, *_ in UNITS:
        a, b = unit_pop[code][2024], unit_pop[code][2025]
        if not (0 <= b / a - 1 <= 0.04):
            print(f"  {code} {name}: {a:,} -> {b:,} ({b / a - 1:+.1%})")
    for n in notes:
        print("NOTE", n)

    # ---------------- write
    rows = []
    for code, by in unit_pop.items():
        for y, v in sorted(by.items()):
            rows.append((code, 2, y, v))
    for reg, *_ in REGIONS:
        codes = [c for c in unit_pop if c.startswith(reg)]
        for y in years:
            if all(y in unit_pop[c] for c in codes):
                rows.append((reg, 1, y, sum(unit_pop[c][y] for c in codes)))
    rows.sort(key=lambda r: (r[1], r[0], r[2]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
