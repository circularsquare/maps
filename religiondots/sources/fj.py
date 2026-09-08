"""Fiji — FBoS, 2007 Census of Population and Housing, Table P01-3.

Reads (or fetches) data/raw/fj/ and writes data/normalized/fj.csv.

**THE MOST RELIGIOUSLY PLURAL COUNTRY IN THE PACIFIC, AND THE ONLY ONE THAT NEEDS MORE THAN A
CHRISTIAN PALETTE.** 837,271 people: **64.9% Christian, 27.7% Hindu, 6.3% Muslim**, plus 2,548
Sikhs. Nothing else on this map from the region carries a Hindu or Muslim plurality anywhere,
and Fiji has both — the Indo-Fijian population brought to the cane districts from 1879.

**THE SOURCE IS A PDF AND THE CHOICE OF TIER IS THE WHOLE DESIGN DECISION.** Fiji publishes
religion two ways and they trade geography against categories:

    THIS FILE   Table P01-3, statsfiji.gov.fj ....  15 provinces, **23 categories**
    NOT USED    SPC PopGIS, fiji.popgis.spc.int ..  86 tikina, **6 categories**

PopGIS is 5.7x finer and lumps all eighteen Christian bodies into one `Christians` column.
That would erase **Methodist**, which is 290,555 people — **34.7% of Fiji and the largest
single body in the country** — and would leave Fiji looking like every other Pacific census on
this map. The categories are the reason the country is worth drawing, so the categories win.
§9k's trade-off, decided the same way and stated out loud. `sources/fj.md` §3 has the argument
and what it costs.

**BUT POPGIS IS NOT WASTED: IT IS THE INDEPENDENT WITNESS**, and a genuinely good one. It is
SPC's database built from FBoS microdata, sharing no code path with a PDF typeset in 2008.
Against the printed table it reproduces **the national total, Hindu and Muslim exactly** and
its 15 provinces are the same 15. It differs on the Christian/no-religion boundary by 1,929
people, **0.23% of Fiji**, and that difference is itself internally consistent — see `check()`
and `sources/fj.md` §5. Both are asserted here.

**THE PRINTED TABLE DOES NOT PRINT ITS OWN RESIDUAL.** The six top-level rows sum to 836,376
against a stated total of 837,271, so **895 people are unaccounted on the page**. PopGIS says
what they are: its `other religion` is exactly the PDF's `Sikh + Other religion + 895`. The
residual is carried as `Not stated` rather than silently dropped or folded into a religion —
§3.5, and it is 0.11% of the country.

Usage:
    python sources/fj.py --fetch    one PDF from statsfiji.gov.fj + 7 PopGIS queries
    python sources/fj.py            normalise from data/raw/fj/
"""

import csv
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "fj")
OUT = os.path.join(ROOT, "data", "normalized", "fj.csv")

SOURCE_ID = "fj_cph_2007"
YEAR = 2007
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# WP File Download on statsfiji.gov.fj. The plugin's own AJAX route is open and is also how
# the whole census table library was enumerated (`sources/fj.md` §1); `wpfd_category_id=117`
# is `01_PROVINCE-OF-ENUMERATION` and file 686 is Table P01-3.
PDF_NAME = "fj_2007_religion_by_province.pdf"
PDF_URL = ("https://www.statsfiji.gov.fj/wp-admin/admin-ajax.php"
           "?juwpfisadmin=false&action=wpfd&task=file.download"
           "&wpfd_category_id=117&wpfd_file_id=686")

# SPC PopGIS 3 (GeoClip Observatory). `map3` is the province view, `map2` the tikina view.
POPGIS = "https://fiji.popgis.spc.int/GC_coldata.php"
POPGIS_NAME = "fj_popgis_province.json"
POPGIS_INDICS = ["total", "t_christians", "t_hindu", "t_muslim",
                 "t_other_religion", "t_no_religion", "t_not_stated"]

# The 15 provinces, in the PDF's own column order (alphabetical, Rotuma last). The order is
# ASSERTED against the parsed header rather than assumed -- see read().
PROVINCES = ["Ba", "Bua", "Cakaudrove", "Kadavu", "Lau", "Lomaiviti", "Macuata",
             "Nadroga/Navosa", "Naitasiri", "Namosi", "Ra", "Rewa", "Serua",
             "Tailevu", "Rotuma"]
# FBoS province id, which is what COD-AB's FBOS_PID and PopGIS's codgeo both carry.
FBOS_PID = {p: str(i + 1) for i, p in enumerate(PROVINCES)}

# The PDF truncates every label to 14 characters. Expanded here, with the truncation kept in
# the comment so the mapping back to the page is checkable by eye.
EXPAND = {
    "Anglican": "Anglican",
    "Apostolic": "Apostolic",
    "Assembly of Go": "Assembly of God",
    "All Nation Chr": "All Nation Christian",
    "Baptist": "Baptist",
    "Catholic": "Catholic",
    "Christ Mis Flw": "Christ Mission Fellowship",
    "Church of Chri": "Church of Christ",
    "Gospel": "Gospel",
    "Jehovah's Witn": "Jehovah's Witnesses",
    "Latter Day Sai": "Latter Day Saints",
    "Methodist": "Methodist",
    "Penticostal": "Penticostal",          # FBoS's own spelling; kept verbatim (§2.4)
    "Presbyterian": "Presbyterian",
    "Salvation Army": "Salvation Army",
    "Seventh Day Ad": "Seventh Day Adventist",
    "United Penteco": "United Pentecostal",
    "Other Christia": "Other Christian",
}
CHRISTIAN_CHILDREN = list(EXPAND.values())
TOP_LEVEL = ["Christian", "Hindu", "Sikh", "Moslem", "Other religion", "No religion"]
TOTAL_CAT = "Total"
# Not printed on the page; the residual the stated total implies. See the module docstring.
RESIDUAL_CAT = "Not stated"

NATIONAL = 837_271
EXPECTED_PROVINCES = 15

# Read off page 1 of the PDF, the `Total` column of Table P01-3.
PRINTED_2008 = {
    "Christian": 543_588, "Hindu": 232_103, "Sikh": 2_548, "Moslem": 52_594,
    "Other religion": 1_294, "No religion": 4_249,
    "Anglican": 6_328, "Apostolic": 5_089, "Assembly of God": 47_873,
    "All Nation Christian": 13_294, "Baptist": 1_772, "Catholic": 76_603,
    "Christ Mission Fellowship": 14_180, "Church of Christ": 1_356, "Gospel": 2_835,
    "Jehovah's Witnesses": 8_450, "Latter Day Saints": 5_126, "Methodist": 290_555,
    "Penticostal": 15_326, "Presbyterian": 2_907, "Salvation Army": 1_144,
    "Seventh Day Adventist": 32_370, "United Pentecostal": 1_361, "Other Christian": 17_019,
}


def fetch():
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 20_000:
        print("already have", dest)
    else:
        print("GET", PDF_URL)
        r = requests.get(PDF_URL, headers=ua, timeout=600, verify=False)
        r.raise_for_status()
        # §5a, and the PDF-truncated-at-source rule: a Content-Length can match a damaged
        # file, so check the trailer rather than the size.
        if not r.content.startswith(b"%PDF") or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit("statsfiji returned something that is not a whole PDF "
                             f"({len(r.content):,} bytes)")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(dest):,} bytes")

    dest = os.path.join(RAW, POPGIS_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        print("already have", dest)
        return
    out = {}
    s = requests.Session()
    s.headers.update(ua)
    for ind in POPGIS_INDICS:
        print("POPGIS", ind)
        r = s.get(POPGIS, params={"view": "map3", "dataset": "d9_religion",
                                  "indic": ind, "vars": ind}, timeout=300, verify=False)
        r.raise_for_status()
        out[ind] = r.json()["content"][ind]
    r = s.get("https://fiji.popgis.spc.int/GC_refdata.php",
              params={"nivgeo": "pid", "view": "map3", "extent": "cid", "lang": "en"},
              timeout=300, verify=False)
    r.raise_for_status()
    out["_territories"] = r.json()["content"]["territories"]
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(dest):,} bytes")


def _religion_lines():
    """The RELIGION block of Table P01-3, page 1, as (indent, label, [16 figures])."""
    import fitz

    path = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)
    if doc.page_count < 1:
        raise SystemExit(f"{path}: PyMuPDF reports {doc.page_count} pages")
    text = doc[0].get_text()
    doc.close()

    head = text.find("RELIGION")
    if head < 0:
        raise SystemExit(f"{path}: page 1 has no RELIGION block -- the table has changed")
    block = text[head:]

    rows = []
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("─"):
            continue
        # `  Assembly of Go 47,873  13,434 ...` — the label is fixed-width and its dot
        # leaders may run straight into the first figure, so split on the FIGURES.
        figs = re.findall(r"(?<![\d,])(?:[\d]{1,3}(?:,[\d]{3})*|-)(?![\d,])", line)
        if len(figs) < EXPECTED_PROVINCES + 1:
            continue
        figs = figs[-(EXPECTED_PROVINCES + 1):]
        label = line[:line.index(figs[0])] if figs[0] in line else line
        label = label.strip().strip(".").strip()
        label = re.sub(r"[\s.]+$", "", label)
        indent = len(line) - len(line.lstrip())
        vals = [0 if f == "-" else int(f.replace(",", "")) for f in figs]
        rows.append((indent, label, vals))
    return rows


def read():
    rows = _religion_lines()
    by_label = {}
    for indent, label, vals in rows:
        clean = EXPAND.get(label, label)
        if clean in by_label:
            raise SystemExit(f"category {clean!r} appears twice in the table")
        by_label[clean] = vals

    want = [TOTAL_CAT] + TOP_LEVEL + CHRISTIAN_CHILDREN
    missing = [c for c in want if c not in by_label]
    if missing:
        raise SystemExit(f"Table P01-3 is missing rows {missing} -- either the PDF changed "
                         f"or the fixed-width parse broke. Parsed: {sorted(by_label)}")

    # The residual the page does not print: total minus the six top-level rows, per province.
    tot = by_label[TOTAL_CAT]
    top = [sum(by_label[c][i] for c in TOP_LEVEL) for i in range(len(tot))]
    by_label[RESIDUAL_CAT] = [t - s for t, s in zip(tot, top)]

    out = []
    for i, prov in enumerate(PROVINCES):
        gid = FBOS_PID[prov]
        for cat in want[1:] + [RESIDUAL_CAT]:
            parent = ("Christian" if cat in CHRISTIAN_CHILDREN else None)
            note = "level=province"
            if parent:
                note += f"; nested inside {parent}"
            if cat == RESIDUAL_CAT:
                note += ("; residual: the table's stated total minus its six printed "
                         "top-level rows, not a printed row")
            out.append({"geo_id": gid, "geo_level": "province", "geo_name": prov,
                        "source_category": cat, "count": by_label[cat][i + 1],
                        "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID, "note": note})
        out.append({"geo_id": gid, "geo_level": "province", "geo_name": prov,
                    "source_category": TOTAL_CAT, "count": tot[i + 1], "basis": BASIS,
                    "year": YEAR, "source_id": SOURCE_ID,
                    "note": "level=province; universe total, not a religion category"})
    return out, by_label


def check(rows, by_label):
    ok = True
    per = {}
    for r in rows:
        per.setdefault(r["source_category"], {})[r["geo_id"]] = r["count"]

    good = len(PROVINCES) == EXPECTED_PROVINCES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} province   {len(PROVINCES):>3} units "
          f"(expected {EXPECTED_PROVINCES})")

    nat = {c: sum(v.values()) for c, v in per.items()}
    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat[TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    # ---- witness 1: every figure typeset on the page ----
    bad = [(c, nat[c], PRINTED_2008[c]) for c in PRINTED_2008 if nat[c] != PRINTED_2008[c]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all {len(PRINTED_2008)} national figures match "
          f"Table P01-3 as printed ({len(bad)} failures)")
    for c, g, w in bad[:6]:
        print(f"        {c}: parsed {g:,} vs printed {w:,}")

    # ---- witness 2: the 18 Christian bodies must rebuild `Christian`, on all 15 ----
    bad = [g for g in per[TOTAL_CAT]
           if sum(per[c][g] for c in CHRISTIAN_CHILDREN) != per["Christian"][g]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 18 Christian bodies sum to `Christian` on "
          f"all {len(PROVINCES)} provinces ({len(bad)} failures) {bad[:3]}")

    # ---- witness 3: the drawn partition must sum to the stated total ----
    drawn = TOP_LEVEL[1:] + CHRISTIAN_CHILDREN + [RESIDUAL_CAT]
    bad = [g for g in per[TOTAL_CAT] if sum(per[c][g] for c in drawn) != per[TOTAL_CAT][g]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the drawn categories partition the total on all "
          f"{len(PROVINCES)} provinces ({len(bad)} failures)")

    resid = nat[RESIDUAL_CAT]
    print(f"      the page prints six top-level rows summing to {NATIONAL - resid:,} against "
          f"a stated\n      total of {NATIONAL:,}. The {resid:,} difference "
          f"({100.0 * resid / NATIONAL:.2f}%) is carried as `{RESIDUAL_CAT}` (§3.5).")

    # ---- witness 4: SPC PopGIS, which shares no code path with the PDF ----
    path = os.path.join(RAW, POPGIS_NAME)
    if not os.path.exists(path):
        print("  !! no PopGIS file; run --fetch to enable the independent witness")
    else:
        pg = json.load(open(path, encoding="utf-8"))
        terr = pg["_territories"]
        code_at = {c: i for i, c in enumerate(terr["codgeo"])}
        name_at = dict(zip(terr["codgeo"], terr["libgeo"]))
        s = lambda k: sum(x or 0 for x in pg[k])
        print(f"\n  the INDEPENDENT WITNESS — SPC PopGIS ({len(terr['codgeo'])} provinces), "
              "an SPC database\n  built from FBoS microdata against a PDF typeset in 2008:")
        pairs = [("total", nat[TOTAL_CAT]), ("t_hindu", nat["Hindu"]),
                 ("t_muslim", nat["Moslem"])]
        for k, mine in pairs:
            g = s(k)
            ok &= g == mine
            print(f"    {'OK ' if g == mine else 'BAD'} {k:<18} popgis {g:>9,}  "
                  f"printed {mine:>9,}")
        # These three differ, and the difference is internally consistent -- see fj.md §5.
        print(f"    -- popgis christians {s('t_christians'):,} vs printed "
              f"{nat['Christian']:,}  (+{s('t_christians') - nat['Christian']:,})")
        print(f"    -- popgis no_religion {s('t_no_religion'):,} vs printed "
              f"{nat['No religion']:,}  ({s('t_no_religion') - nat['No religion']:,})")
        other_pdf = nat["Sikh"] + nat["Other religion"] + resid
        good = s("t_other_religion") == other_pdf
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} popgis other_religion {s('t_other_religion'):,}"
              f" == printed Sikh + Other religion + the {resid} residual = {other_pdf:,}")
        print("       ^ this is what identifies the unprinted residual as `not stated` "
              "rather than\n         a religion: PopGIS puts it exactly where a residual "
              "goes.")
        # And the geography: PopGIS's 15 provinces must be our 15, by name.
        got = {name_at[c].replace("_", "/") for c in terr["codgeo"]}
        bad = got ^ set(PROVINCES)
        ok &= not bad
        print(f"    {'OK ' if not bad else 'BAD'} PopGIS names the same 15 provinces "
              f"({len(bad)} differences) {sorted(bad)[:4]}")

    print(f"\n  {len(rows):,} rows. The drawn categories, national:")
    for cat in sorted(drawn, key=lambda c: -nat[c]):
        mark = "   (inside Christian)" if cat in CHRISTIAN_CHILDREN else ""
        print(f"    {nat[cat]:>9,}  {100.0 * nat[cat] / NATIONAL:6.2f}%  {cat}{mark}")

    # Methodist is why the province tier was chosen over PopGIS's finer one.
    meth = per["Methodist"]
    tot = per[TOTAL_CAT]
    top = sorted(((meth[g] / tot[g], g) for g in tot), reverse=True)
    name = {FBOS_PID[p]: p for p in PROVINCES}
    print("\n  `Methodist` is 34.7% of Fiji and the largest body in the country — and PopGIS's "
          "\n  finer tier cannot show it at all. Its five strongest provinces:")
    for share, g in top[:5]:
        print(f"    {name[g]:<16} {meth[g]:>7,} of {tot[g]:>7,}  {100 * share:6.2f}%")
    hin = per["Hindu"]
    top = sorted(((hin[g] / tot[g], g) for g in tot), reverse=True)
    print("  and `Hindu` is the other half of the country, on the cane coast:")
    for share, g in top[:4]:
        print(f"    {name[g]:<16} {hin[g]:>7,} of {tot[g]:>7,}  {100 * share:6.2f}%")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, by_label = read()
    check(rows, by_label)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
