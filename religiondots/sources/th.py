"""Thailand — 2010 census religion, at province via spec §3.10 allocation.

Reads (or fetches) data/raw/th/ and writes data/normalized/th.csv.

**THE OFFICE MOVED HOSTS AND THE OLD ONES DIED, SO EVERY CENSUS FILE HERE COMES OUT OF THE
WAYBACK MACHINE.** `sources.md` recorded Thailand as *"data existed and the server is gone"*:
`statbbi.nso.go.th` and `web.nso.go.th` no longer resolve, `nsodw.nso.go.th` refuses
connections. What that missed is that **`www.nso.go.th` is alive** — it answers 418 to a bare
curl and 200 to a browser User-Agent — and that its old `/sites/2014/Documents/` tree, now
404, was archived wholesale in 2018-2022. That tree is the source below. The live site's new
CKAN (`catalog.nso.go.th`, keyless, real) carries only a 6-region 3-religion survey table and
is NOT used.

## What exists, and why this needs an allocation at all

**No published Thai census table crosses religion with province.** That was checked rather
than assumed, in both censuses:

  * 2010 — the national report's only religion table is Table 4, *"Population by religion,
    sex and area"*, where `area` is municipal/non-municipal. Each of the five regional volumes
    carries its own Table 4 at the same cut, and none of that volume's other 21 tables crosses
    religion with changwat.
  * 2000 — religion is Table 5, *"Population by religion, age group, sex and area"*, national
    only. (Table numbering is NOT stable between censuses: 2000's Table 4 is marital status.)

The province cut existed as one file per changwat on `statbbi`/`service.nso.go.th`
(`..._C-pop_2553_000_<PROV>_00400.xls`). Those hosts are gone and **the archive holds 2 of
77**, so that route is dead.

So the two halves are taken from different documents and reunited by spec §3.10:

| | geography | categories | file |
|---|---|---|---|
| fine geography | **77 provinces** | 3: Buddhist, Muslim, residual | `kpi_stat/<Province>_T.pdf` |
| fine categories | 5 regions | **9** | `2553/3/<region>/Table4.xls` |

`allocate.py --hierarchy parent --within 1` splits each province's residual by its own
region's composition of the other seven categories. That is the right cut rather than a
national one: Christianity is 3.05% of the North and 0.35% of the Northeast, and pooling would
scatter the hill churches across Isan.

## The three sources, and what each is trusted for

1. **`2553/3/<region>/Table1.xls`** — population by changwat, exact counts. This supplies the
   province DENOMINATOR and, for free, **the province→region assignment, read off which
   volume a province appears in rather than asserted here.**
2. **`2553/3/<region>/Table4.xls`** — the 9 religion categories per region, exact counts.
3. **`2553/kpi_stat/<Province>_T.pdf`** — a two-page provincial indicator sheet from the same
   census, carrying `Buddhists (%)` and `Muslims (%)` for 1990, 2000 and 2010. All 77 are
   archived and the Thai text is Unicode, not the substitution-cipher font of the 2000
   provincial reports.

**The percentages are the weak link and the rounding is stated rather than hidden.** The KPI
sheet gives one decimal place, so a share carries ±0.05% and a 600,000-person province ±300
people; the residual, being 100 − Buddhist − Muslim, carries twice that. The denominator does
NOT compound it — province totals come from Table 1 at full precision, not from the KPI
sheet's own population figure, which is rounded to hundreds.

**A missing line means a share too small to print, not a share of zero we measured.** Mae Hong
Son has no `Muslims (%)` row at all. Those are read as 0 and the people land in the residual,
which is where the allocation puts them back among the small categories anyway.

## The joins

Province identity travels as the **TIS 1099 code** carried by OCHA's COD attribute table
(`adm1_pcode` = `TH` + two digits), which also supplies each province's Thai and English name
in one row — so the Thai census names and the English KPI filenames meet on a published
standard rather than on each other. `geo_id` is then `<region digit><two-digit code>`, three
characters, which is what makes `allocate.py --within 1` work.

Usage:
    python sources/th.py --fetch    ~90 files from the Wayback Machine + one HDX xlsx, ~3 min
    python sources/th.py            rebuild from data/raw/th/
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "th")
OUT = os.path.join(ROOT, "data", "normalized", "th.csv")

SOURCE_ID = "th_census_2010"
YEAR = 2010
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = "religiondots/1.0 (map research; anitaxinchen@gmail.com)"
CDX = ("http://web.archive.org/cdx/search/cdx?url=www.nso.go.th/sites/2014/Documents/pop/"
       "&matchType=prefix&limit=60000&fl=original,timestamp,statuscode&collapse=urlkey")
COD_XLSX = ("https://data.humdata.org/dataset/d24bdc45-eb4c-4e3d-8b16-44db02667c27/"
            "resource/a925b917-01ae-48c4-9279-7efe680a6b11/download/"
            "tha_admin_boundaries.xlsx")

# The NSO's own regional scheme: four regions plus Bangkok, which is its own volume. The
# digit is this file's invention and exists only to prefix the province geo_id so that
# `allocate.py --within 1` can find a province's region; nothing downstream reads it as a
# published code.
REGIONS = {
    "bangkok": ("1", "Bangkok"),
    "central": ("2", "Central"),
    "north": ("3", "North"),
    "north_east": ("4", "Northeast"),
    "south": ("5", "South"),
}

# Table 4's nine rows, in file order under the total. Thai is the source's own label and is
# what taxonomy/th2010.py maps; the English is the source's own too, from the same sheet.
CATEGORIES = [
    ("พุทธ", "Buddhism"),
    ("อิสลาม", "Islam"),
    ("คริสต์", "Christianity"),
    ("ฮินดู", "Hinduism"),
    ("ขงจื้อ", "Confucianism"),
    ("ซิกข์", "Sikhism"),
    ("อื่น ๆ", "Other"),
    ("ไม่มีศาสนา", "No religion"),
    ("ไม่ทราบ", "Unknown"),
]
TOTAL_ROW = "ยอดรวม"
# Bangkok's volume labels its total `รวม` and enumerates districts rather than provinces —
# see read_region_table1.
BKK_TOTAL_ROW = "รวม"
BANGKOK_TH = "กรุงเทพมหานคร"

# The three categories the province sheets can support. The residual is deliberately NOT
# called `อื่น ๆ`, which is one of Table 4's own nine and would collide in allocate.py's
# parent map.
FINE_BUDDHIST = "พุทธ"
FINE_MUSLIM = "อิสลาม"
FINE_RESIDUAL = "อื่น ๆ ไม่มีศาสนา และไม่ทราบ"

# KPI-sheet filename -> COD English name, for the ones that are not simply a de-spaced
# version of each other. Each is a real difference in the NAME rather than in spelling: NSO
# shortens the long ceremonial ones. Kept as a table because a rule would be worse — there is
# no rule that turns `Ayutthaya` into `Phra Nakhon Si Ayutthaya`. Anything not in here and not
# matching on the fold is REPORTED rather than guessed (§12).
KPI_ALIASES = {
    "ayutthaya": "phranakhonsiayutthaya",
    "phachuap": "prachuapkhirikhan",
    "ubonatchathani": "ubonratchathani",
}

# Two filenames in the KPI directory are not provinces of the 2010 census.
KPI_SKIP = {
    # the national sheet, which the regional Table 4s already give
    "wholekingdom",
    # Bueng Kan was carved out of Nong Khai on 23 March 2011, SEVEN MONTHS AFTER the
    # census. NSO published a sheet for it anyway, back-computed; the census's own Table 1
    # has no such changwat and Nong Khai's 2010 population still contains it. Drawing both
    # would double-count, so Bueng Kan is dropped here and `th_geo.py` dissolves its
    # polygon into Nong Khai's — spec §8.1, boundaries at the vintage the data was
    # published on.
    "buengkan",
}

# Kanchanaburi's sheet was never archived: it is the one province of 76 with no captured
# `<Province>_T.pdf` anywhere in the Wayback Machine (checked against every capture of the
# kpi_stat directory, not just the collapsed listing). Its Buddhist and Muslim shares are
# therefore taken from its own region and the rows are marked so `countries.py` can put
# them on the `derived` tier. 848,000 people, 1.3% of the country.
NO_SHEET = "71"


def fold(s):
    """Lowercase, strip everything that is not a letter or digit."""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"[^0-9a-z฀-๿]+", "", s.lower())


# ---------------------------------------------------------------- fetch

def _get(url, dest, tries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 2048:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) < 2048:
                raise IOError("suspiciously short: %d bytes" % len(data))
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)
            return True
        except Exception as e:
            # The archive rate-limits and refuses connections under load rather than
            # returning 429, so back off rather than giving up (sources.md §5a).
            if attempt == tries - 1:
                print("    FAILED %s -- %s" % (os.path.basename(dest), e))
                return False
            time.sleep(3 * (attempt + 1))
    return False


def fetch():
    os.makedirs(RAW, exist_ok=True)

    cdx_path = os.path.join(RAW, "cdx_pop.txt")
    if not (os.path.exists(cdx_path) and os.path.getsize(cdx_path) > 1024):
        print("CDX: listing the archived /sites/2014/Documents/pop/ tree…")
        req = urllib.request.Request(CDX, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            open(cdx_path, "wb").write(r.read())
    rows = [l.split() for l in io.open(cdx_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if len(r) == 3 and r[2] == "200"]
    print("CDX: %d archived URLs with status 200" % len(rows))

    def wb(url, ts):
        return "https://web.archive.org/web/%sif_/%s" % (ts, url)

    # 1. the regional volumes: Table1 (province populations) and Table4 (religion)
    n = 0
    for url, ts, _ in rows:
        m = re.search(r"/pop/2553/3/([a-z_]+)/(table1|table4)\.xls$", url, re.I)
        if not m:
            continue
        region, table = m.group(1).lower(), m.group(2).lower()
        if region not in REGIONS:
            continue
        if _get(wb(url, ts), os.path.join(RAW, "region", "%s_%s.xls" % (region, table))):
            n += 1
    print("regional volumes: %d files" % n)

    # 2. the 77 provincial indicator sheets
    n = 0
    for url, ts, _ in rows:
        m = re.search(r"/pop/2553/kpi_stat/([A-Za-z]+)_T\.pdf$", url)
        if not m:
            continue
        if _get(wb(url, ts), os.path.join(RAW, "kpi", m.group(1) + ".pdf")):
            n += 1
        time.sleep(0.4)
    print("provincial KPI sheets: %d files" % n)

    # 3. OCHA COD attribute table, for the province name/code bridge
    _get(COD_XLSX, os.path.join(RAW, "tha_admin_boundaries.xlsx"))
    print("COD attribute table: ok")


# ---------------------------------------------------------------- parse

def read_cod():
    """Thai name -> (two-digit TIS code, English name), from OCHA's COD attribute table."""
    import pandas as pd
    df = pd.read_excel(os.path.join(RAW, "tha_admin_boundaries.xlsx"),
                       sheet_name="tha_admin1")
    out = {}
    for _, r in df.iterrows():
        code = str(r["adm1_pcode"]).strip()
        if not re.fullmatch(r"TH\d{2}", code):
            continue
        out[fold(r["adm1_name1"])] = (code[2:], str(r["adm1_name"]).strip())
    if len(out) != 77:
        raise SystemExit("COD admin1 gave %d provinces, expected 77" % len(out))
    return out


def read_region_table1(region):
    """Province Thai name -> total population, from that region's Table 1.

    The sheet repeats its header block every page and puts each province's municipal and
    non-municipal rows underneath it, indented. Take column 0 labels that are neither
    indented, nor the region total, nor part of a header.

    **BANGKOK'S VOLUME IS NOT LIKE THE OTHER FOUR AND IT IS NOT A BUG.** Bangkok is a
    single changwat, so its Table 1 enumerates its FIFTY DISTRICTS (เขต) instead of
    provinces, and labels the total `รวม` where the regional volumes say `ยอดรวม`. Reading
    it the same way would put fifty districts into a province table. So Bangkok returns
    exactly one unit — itself — and its districts are deliberately discarded.

    **The districts are a real level this build declines to use**, and the reason is that
    religion is published for Bangkok as a whole and nothing else: allocating the city's
    nine categories across fifty districts by population would draw a district geography
    that asserts nothing the city total does not already say, while looking like a
    measurement. spec §8.2's placement grid already spreads Bangkok's dots by where people
    live, which is the honest version of the same thing. The cost is stated in
    `sources/th.md`: Bangkok is 8.3M people in one unit, the coarsest on this map after
    Île-de-France.
    """
    import xlrd
    path = os.path.join(RAW, "region", "%s_table1.xls" % region)
    sh = xlrd.open_workbook(path).sheet_by_index(0)
    out, region_total = {}, None
    for r in range(sh.nrows):
        label = str(sh.cell_value(r, 0)).strip()
        if not label or label.startswith("ในเขต") or label.startswith("นอกเขต"):
            continue
        if label.startswith("ตาราง") or label.startswith("Table") or "จังหวัด" in label:
            continue
        val = None
        for c in range(1, min(sh.ncols, 6)):
            v = sh.cell_value(r, c)
            if isinstance(v, float) and v > 0:
                val = int(round(v))
                break
        if val is None:
            continue
        if label in (TOTAL_ROW, BKK_TOTAL_ROW):
            region_total = val
        elif region != "bangkok":
            out[label] = val
    if region == "bangkok":
        out = {BANGKOK_TH: region_total}
    return out, region_total


def read_region_table4(region):
    """Region -> {Thai category: count}, from that region's Table 4."""
    import xlrd
    path = os.path.join(RAW, "region", "%s_table4.xls" % region)
    sh = xlrd.open_workbook(path).sheet_by_index(0)
    wanted = {th for th, _ in CATEGORIES} | {TOTAL_ROW}
    out = {}
    for r in range(sh.nrows):
        label = str(sh.cell_value(r, 0)).strip()
        if label not in wanted:
            continue
        for c in range(1, min(sh.ncols, 8)):
            v = sh.cell_value(r, c)
            if isinstance(v, float) and v > 0:
                out[label] = int(round(v))
                break
    missing = wanted - set(out)
    if missing:
        raise SystemExit("%s Table 4 is missing %s" % (region, sorted(missing)))
    return out


NUM = re.compile(r"^-?[\d,]+\.?\d*$")
LATIN = re.compile(r"[A-Za-z]")


def _value_2010(lines, marker):
    """The 2010 figure for one indicator, or None if the sheet does not print one.

    **THE SHEET DOES NOT ALWAYS PRINT THREE NUMBERS AND THAT COST A WRONG ANSWER BEFORE IT
    WAS CAUGHT.** Each indicator is laid out as

        <Thai label>
        <1990>
        <2000>
        <2010>   <English label>

    but 1990 and 2000 are often the footnote marker `a` — NSO's *"less than half the last
    digit shown"* — or `na`. A reader that collects "the first three numeric lines" walks
    straight through those into the NEXT indicator's values: Lampang's Muslim share came
    out as its 90.2% household-registration rate, and Buddhist + Muslim then exceeded the
    population, which is the only reason it was noticed.

    So the anchor is the ENGLISH LABEL rather than a count of numbers. The 2010 value is
    always the first token of the line that also carries the English label, because that is
    how the two-column bilingual layout falls out of the PDF's text order. Scanning stops
    at that line, so no indicator can borrow the next one's numbers.
    """
    for i, ln in enumerate(lines):
        if marker not in ln:
            continue
        for ln2 in lines[i + 1:i + 9]:
            s = ln2.strip()
            if not s or not LATIN.search(s):
                continue
            first = s.split()[0]
            if NUM.match(first):
                return float(first.replace(",", ""))
            # `a` or `na` sitting where the value goes: printed, and not a number.
            return None
    return None


def read_kpi(path):
    """-> (Buddhist %, Muslim %, the sheet's own population in thousands) for 2010."""
    import fitz
    lines = fitz.open(path)[0].get_text().splitlines()
    tot = _value_2010(lines, "ประชากรรวม")
    bud = _value_2010(lines, "นับถือศาสนาพุทธ")
    isl = _value_2010(lines, "นับถือศาสนาอิสลาม")
    if tot is None or bud is None:
        raise SystemExit("%s: could not read the 2010 column" % path)
    # A share too small to print gets `a` or no row at all. Neither is a measured zero,
    # and both are read as one: the people land in the residual, where the allocation puts
    # them back among the small categories anyway.
    return bud, (isl or 0.0), tot


def _kpi_key(stem):
    """KPI filename -> the fold used to meet COD's English name."""
    f = fold(stem)
    return KPI_ALIASES.get(f, f)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()

    cod = read_cod()                                    # thai fold -> (code, english)
    by_english = {fold(en): (code, th) for th, (code, en) in cod.items()}
    print("COD: 77 provinces, Thai and English names on the TIS 1099 code")

    # ---- the coarse half: 5 regions x 9 categories
    region_cats, region_totals = {}, {}
    for region in REGIONS:
        region_cats[region] = read_region_table4(region)
        region_totals[region] = region_cats[region][TOTAL_ROW]
    kingdom = sum(region_totals.values())
    print("\nregions (Table 4):")
    for region in REGIONS:
        print("  %-11s %12s" % (region, "{:,}".format(region_totals[region])))
    print("  %-11s %12s  <- five regions summed" % ("kingdom", "{:,}".format(kingdom)))

    # ---- the fine half: provinces, their region, and their exact totals
    prov_total, prov_region, prov_thai = {}, {}, {}
    unresolved = []
    for region in REGIONS:
        t1, t1_total = read_region_table1(region)
        if t1_total is None:
            raise SystemExit("%s Table 1 has no ยอดรวม row" % region)
        # Table 1's own region total must be Table 4's, or the two halves are different
        # populations and the allocation would be splitting the wrong denominator.
        if t1_total != region_totals[region]:
            raise SystemExit("%s: Table 1 total %d != Table 4 total %d"
                             % (region, t1_total, region_totals[region]))
        for th_name, tot in t1.items():
            hit = cod.get(fold(th_name))
            if hit is None:
                unresolved.append((region, th_name, tot))
                continue
            code, _en = hit
            prov_total[code] = tot
            prov_region[code] = region
            prov_thai[code] = th_name
    print("\nprovinces from Table 1: %d resolved on the TIS code" % len(prov_total))
    if unresolved:
        print("  !! %d Thai province names not in COD:" % len(unresolved))
        for region, name, tot in unresolved:
            print("     %-11s %-24s %10s" % (region, name, "{:,}".format(tot)))

    # ---- the fine categories: Buddhist and Muslim shares per province
    kpi_dir = os.path.join(RAW, "kpi")
    shares, unmatched, skipped = {}, [], []
    for fn in sorted(os.listdir(kpi_dir)):
        if not fn.endswith(".pdf"):
            continue
        stem = fn[:-4]
        if fold(stem) in KPI_SKIP:
            skipped.append(stem)
            continue
        hit = by_english.get(_kpi_key(stem))
        if hit is None:
            unmatched.append(stem)
            continue
        code, _th = hit
        shares[code] = read_kpi(os.path.join(kpi_dir, fn))
    print("\nKPI sheets: %d matched, %d skipped (%s)"
          % (len(shares), len(skipped), ", ".join(sorted(skipped))))
    if unmatched:
        print("  !! unmatched KPI filenames: %s" % sorted(unmatched))

    # The one province with no sheet takes its region's shares, marked as such.
    estimated = []
    for code in sorted(set(prov_total) - set(shares)):
        region = prov_region[code]
        c = region_cats[region]
        shares[code] = (100.0 * c[FINE_BUDDHIST] / c[TOTAL_ROW],
                        100.0 * c[FINE_MUSLIM] / c[TOTAL_ROW],
                        prov_total[code] / 1000.0)
        estimated.append(code)
    if estimated:
        print("  !! %d province(s) have no KPI sheet and take their REGION's shares "
              "(tier `derived`): %s"
              % (len(estimated), [prov_thai[c] for c in estimated]))
        if sorted(estimated) != [NO_SHEET]:
            print("     ^^ that set has CHANGED — expected only %s. Re-read the archive "
                  "before trusting this build." % NO_SHEET)

    # ---- write
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    parents = {th: (th if th in (FINE_BUDDHIST, FINE_MUSLIM) else FINE_RESIDUAL)
               for th, _ in CATEGORIES}

    for region, (digit, en) in REGIONS.items():
        for th, en_cat in CATEGORIES:
            rows.append([digit, "region", en, th, region_cats[region][th], BASIS, YEAR,
                         SOURCE_ID,
                         "english=%s; parent=%s; table=Table 4 (2553, %s volume)"
                         % (en_cat, parents[th], region)])

    drawn = 0
    for code in sorted(prov_total):
        if code not in shares:
            continue
        region = prov_region[code]
        digit = REGIONS[region][0]
        total = prov_total[code]
        bud_pct, isl_pct, kpi_pop = shares[code]
        bud = int(round(total * bud_pct / 100.0))
        isl = int(round(total * isl_pct / 100.0))
        res = total - bud - isl
        if res < 0:
            raise SystemExit("%s: Buddhist+Muslim exceed the population" % code)
        gid = digit + code
        note = ("province=%s; region=%s; census_total=%d; kpi_pop_000=%.1f; "
                "buddhist_pct=%.1f; muslim_pct=%.1f%s"
                % (prov_thai[code], region, total, kpi_pop, bud_pct, isl_pct,
                   "; shares_from=region" if code in estimated else ""))
        for cat, n in ((FINE_BUDDHIST, bud), (FINE_MUSLIM, isl), (FINE_RESIDUAL, res)):
            if n <= 0 and cat != FINE_RESIDUAL:
                continue
            rows.append([gid, "province", prov_thai[code], cat, n, BASIS, YEAR,
                         SOURCE_ID, note])
        drawn += total

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)

    # ---- checks the reader should see
    print("\nwrote %s" % OUT)
    print("  %d rows: %d region, %d province"
          % (len(rows), len(REGIONS) * len(CATEGORIES), len(rows) - len(REGIONS) * len(CATEGORIES)))
    print("  province population written: %s of %s (%.2f%%)"
          % ("{:,}".format(drawn), "{:,}".format(kingdom),
             100.0 * drawn / kingdom))

    print("\nspec §3.10 reconciliation — each region's province residuals against its own "
          "Table 4 non-Buddhist non-Muslim total:")
    for region, (digit, en) in REGIONS.items():
        c = region_cats[region]
        t4_res = c[TOTAL_ROW] - c[FINE_BUDDHIST] - c[FINE_MUSLIM]
        p_res = 0
        for code in prov_total:
            if prov_region[code] != region or code not in shares:
                continue
            b, i, _ = shares[code]
            p_res += prov_total[code] * (100.0 - b - i) / 100.0
        rel = (p_res - t4_res) / t4_res * 100.0 if t4_res else float("nan")
        flag = "  <-- CHECK" if abs(rel) > 25 else ""
        print("  %-11s table4=%11s  provinces=%11s  %+7.1f%%%s"
              % (region, "{:,}".format(t4_res), "{:,.0f}".format(p_res), rel, flag))


if __name__ == "__main__":
    main()
