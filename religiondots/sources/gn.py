"""Guinea — RGPH 2014 (RGPH-3), religion by région administrative.

Reads (or fetches) data/raw/gn/RGPH3_etat_structure.pdf and writes data/normalized/gn.csv.
`sources/gn.md` is the write-up; `sources/gn_geo.py` builds the polygons and the grid.

## §11w SAID "NO RELIGION VOLUME", AND THE TABLE WAS IN THE STRUCTURE VOLUME ALL ALONG

§11w swept the RGPH-3 thematic series and recorded *"full RGPH-3 thematic series, no
religion volume"*. Both halves are true and the conclusion drawn from them was wrong: there
is no volume ABOUT religion, and **the *État et structure de la population* volume carries
religion in its chapter 5**, as three tables:

    Tableau 5.09  religion x urban/rural x sex         percentages      p88 (PDF page 90)
    Tableau 5.10  région administrative x religion     percentages      p88 (PDF page 90)
    Tableau 5.11  religion x urban/rural, 1996 and 2014 percentages     p88 (PDF page 90)

Tableau 5.10 is what is drawn: eight régions, five answers, one decimal. The office is live,
the PDF is a plain GET on `stat-guinee.org`, and the Wayback Machine has the same file from
2020 as a fallback. **A volume's title is not its table list**; open the table list.

## NOTHING FINER EXISTS IN PRINT, CHECKED ACROSS BOTH CENSUSES

Every RGPH-3 (2014) report the Wayback CDX lists for `stat-guinee.org` was downloaded, all
eighteen, and every page mentioning religion was checked for prefecture names: none prints
religion below the région. The RGPH-2 (1996) volumes are scanned images, so text search says
nothing about them; the *État de la population* volume's own table list was read instead and
its only religion table is Tableau 5-3, religion by sex and residence, national. The INS
statistical yearbooks (2014, 2021, 2024) repeat the national line and nothing else. The INS
NADA (`microdata.insguinee.org`) holds no census. `rgph.insguinee.org` is an indicator portal
with no religion theme. So eight régions is the ceiling, in 2014 and in 1996 alike.

## THE TABLE IS PERCENTAGES, SO THE COUNTS ARE THIS MODULE'S ARITHMETIC (§3.4, ci.py's move)

    shares        Tableau 5.10, région x religion, one decimal           this PDF
    denominators  Tableau 2.07, resident population by région, counts    this PDF, p34
    magnitudes    UNSD Demographic Yearbook table 28, Guinea 2014        tools/oracle.py

Each région's share is applied to its population, then each religion is rescaled so the eight
régions sum to its national count. That takes the one-decimal rounding out of every national
figure and leaves it only in the distribution between régions, where it cannot be helped.

## THE RELIGION UNIVERSE IS ORDINARY HOUSEHOLDS, AND UNSD'S `Unknown` IS EXACTLY THE REST

UNSD's Guinea 2014 row is Muslim 9,358,718, Christian 711,349, No Religion 252,786, Animist
167,406, Other 12,873 and **Unknown 20,129**. Tableau 2.04 of this volume gives the population
of collective households (barracks, hotels, boarding schools, orphanages) as **20,129**, and
Tableau 2.05 splits it 5,750 urban and 14,379 rural, which are UNSD's urban and rural
`Unknown` to the person. So the five answers are the 10,503,132 people in ordinary
households, and **the report's percentages are shares of that population, not of all
residents**: 9,358,718 / 10,503,132 is 89.10%, the printed 89.1. `check()` asserts that all
fifteen of Tableau 5.09's national, urban and rural cells are reproduced from UNSD's counts
over the ordinary-household denominators, which is also what ties the two sources' labels
together.

The collective-household population is not published by région, so the shares are applied to
resident population and the per-religion rescale spreads the 0.19% deficit across the régions
in proportion. The largest error that can introduce is a fraction of one percent of a région.

## THE QUESTIONNAIRE AGREES WITH THE COLUMN ORDER

The Zambia lesson (§9db) is that a column header can be wrong. The RGPH 2014 household form
(IPUMS enumeration materials, `enum_form_gn2014a.pdf`, p3) asks P11 *Religion* with codes
**0 = Sans religion, 1 = Musulmane, 2 = Chrétienne, 3 = Animiste, 4 = Autre religion**, which
is Tableau 5.10's column order exactly, and five boxes, which is its column count. There is no
denominational split to lose: the form never asked one.

Usage:
    python sources/gn.py --fetch    one ~3.8 MB PDF from stat-guinee.org
    python sources/gn.py            normalise from data/raw/gn/
"""

import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gn")
OUT = os.path.join(ROOT, "data", "normalized", "gn.csv")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import FetchCheckError, check_body   # noqa: E402  shared, not copied

SOURCE_ID = "gn_rgph3_2014_etat_structure"
YEAR = 2014
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = ("https://www.stat-guinee.org/images/Documents/Publications/INS/rapports_enquetes/"
       "RGPH3/RGPH3_etat_structure.pdf")
WAYBACK = ("http://web.archive.org/web/20200921185302id_/http://www.stat-guinee.org/images/"
           "Documents/Publications/INS/rapports_enquetes/RGPH3/RGPH3_etat_structure.pdf")
PDF = os.path.join(RAW, "RGPH3_etat_structure.pdf")
PAGES = 122

# 0-based page indices.
PAGE_T207 = 35        # printed p34, Tableau 2.07, population by région
PAGE_T510 = 89        # printed p88, Tableaux 5.09, 5.10, 5.11
PAGE_ANNEX = 105      # printed p104, the annex's région table repeats the same totals

CATS = ["Sans religion", "Musulmane", "Chrétienne", "Animiste", "Autres religions"]

# Tableau 5.10, transcribed, and asserted equal to what is parsed off the page. Two copies so
# that a re-issued PDF and a transcription slip each fail loudly rather than either winning.
T510 = {
    "Boké":       (0.3, 96.8, 2.8, 0.1, 0.0),
    "Conakry":    (0.3, 94.8, 4.8, 0.0, 0.1),
    "Faranah":    (0.8, 89.1, 9.7, 0.1, 0.2),
    "Kankan":     (0.2, 98.7, 1.1, 0.0, 0.0),
    "Kindia":     (0.5, 97.2, 2.3, 0.0, 0.0),
    "Labé":       (0.2, 99.4, 0.4, 0.0, 0.0),
    "Mamou":      (0.1, 99.4, 0.5, 0.0, 0.0),
    "N'Zérékoré": (14.2, 46.7, 28.1, 10.4, 0.6),
}
T510_ENSEMBLE = (2.4, 89.1, 6.8, 1.6, 0.1)

# Tableau 5.09's Total columns (both sexes), same category order.
T509 = {
    "Urban": (0.6, 91.7, 7.4, 0.2, 0.1),
    "Rural": (3.4, 87.7, 6.4, 2.3, 0.1),
    "Total": (2.4, 89.1, 6.8, 1.6, 0.1),
}

# Tableau 2.07: (masculin, féminin, effectif). Résident population, census of April 2014.
T207 = {
    "Boké":       (526_919, 556_228, 1_083_147),
    "Conakry":    (833_755, 827_218, 1_660_973),
    "Faranah":    (450_823, 490_731, 941_554),
    "Kankan":     (980_449, 992_088, 1_972_537),
    "Kindia":     (749_192, 812_144, 1_561_336),
    "Labé":       (448_859, 545_599, 994_458),
    "Mamou":      (332_008, 399_180, 731_188),
    "N'Zérékoré": (762_301, 815_767, 1_578_068),
}
RESIDENT = 10_523_261            # Tableau 2.02 / 2.04, population résidente
MEN = 5_084_306

# Tableaux 2.04 and 2.05: the ordinary-household population is the religion universe.
ORDINARY = {"Total": 10_503_132, "Urban": 3_651_372, "Rural": 6_851_760}
COLLECTIVE = {"Total": 20_129, "Urban": 5_750, "Rural": 14_379}

# UNSD's labels -> Tableau 5.10's columns. `Unknown` is the collective households; see the
# docstring, and check() asserts it to the person.
UNSD = {"No Religion": "Sans religion", "Muslim": "Musulmane", "Christian": "Chrétienne",
        "Animist": "Animiste", "Other": "Autres religions"}
UNSD_NATIONAL = {                 # transcribed from `python tools/oracle.py Guinea`, and
    "Muslim": 9_358_718,          # asserted against the cache when it is present
    "Christian": 711_349,
    "No Religion": 252_786,
    "Animist": 167_406,
    "Other": 12_873,
    "Unknown": 20_129,
}

PCT = re.compile(r"^\d{1,3},\d$")
SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) > 1_000_000:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for url in (URL, WAYBACK):
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
        except Exception as e:                       # noqa: BLE001 - try the archive next
            print(f"  {url[:60]}...: {e}")
            continue
        # §5a, and [[reference_pdf_truncated_at_source]]: check the trailer, not the status.
        # Not pinned, so a PDF without %%EOF in its last 2 KB is refused, with where its
        # markers are (sources/fetch_checks.py::check_body).
        try:
            check_body(body, "pdf", where=f"{url[:60]}...")
        except FetchCheckError as e:
            print(f"  {e}")
            continue
        with open(PDF + ".part", "wb") as fh:
            fh.write(body)
        os.replace(PDF + ".part", PDF)
        print(f"wrote {PDF} ({len(body):,} bytes) from {url[:40]}")
        return
    raise SystemExit("neither stat-guinee.org nor the Wayback copy returned the PDF")


def _lines(doc, pno):
    return [ln for ln in (despace(x) for x in doc.load_page(pno).get_text().splitlines())
            if ln]


def read_t510(doc):
    """Tableau 5.10 off the page: a name line, then five percentages and `100,0`."""
    lines = _lines(doc, PAGE_T510)
    start = next((i for i, ln in enumerate(lines) if ln.startswith("Tableau 5.10")), None)
    if start is None:
        raise SystemExit(f"no `Tableau 5.10` on page index {PAGE_T510}")
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("5.3.3")),
               len(lines))
    out = {}
    i = start + 1
    while i < end:
        if PCT.match(lines[i]):
            i += 1
            continue
        j, nums = i + 1, []
        while j < end and PCT.match(lines[j]):
            nums.append(float(lines[j].replace(",", ".")))
            j += 1
        if len(nums) == 6:
            out[lines[i]] = tuple(nums)
            i = j
        else:
            i += 1
    return out


def unsd_counts():
    """UNSD table 28, Guinea 2014, from the oracle cache when present; else the transcription."""
    try:
        import oracle
        rows = oracle.oracle("Guinea", 2014)
    except SystemExit:
        rows = None
    if not rows:
        print("  -- oracle cache not present; using the transcribed UNSD row")
        return {"Total": dict(UNSD_NATIONAL)}, False
    out = {}
    for area, cats in rows.items():
        key = next((k for k in ("Total", "Urban", "Rural") if k.lower() in area.lower()), None)
        if key:
            out[key] = {c: v for c, v in cats.items() if c != oracle.TOTAL}
    return out, True


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Guinea — RGPH 2014, État et structure de la population, Tableau 5.10\n")
    say(doc.page_count == PAGES, f"the volume is {doc.page_count} pages (expected {PAGES})")

    # 1. the parsed table is the transcribed one
    parsed = read_t510(doc)
    by_key = {norm(k): v for k, v in parsed.items()}
    rows_ok = all(by_key.get(norm(k), (None,))[:5] == v for k, v in T510.items())
    say(rows_ok and len(parsed) == 9,
        f"Tableau 5.10 parsed off the page: {len(parsed)} rows (8 régions + Ensemble), "
        "every région identical to the transcription")
    if not rows_ok:
        for k, v in T510.items():
            print(f"        {k}: page {by_key.get(norm(k))} transcribed {v}")
    say(by_key.get("ensemble", (None,))[:5] == T510_ENSEMBLE,
        f"its Ensemble row is {by_key.get('ensemble')}")

    # 2. every row closes
    worst = max(abs(sum(v) - 100.0) for v in list(T510.values()) + [T510_ENSEMBLE])
    say(worst <= 0.25 + 1e-9, f"every row's five cells sum to 100 within {worst:.2f} pp "
        "(bound 0.25 = 5 cells x 0.05)")

    # 3. the région populations, twice printed, and their sums
    text207 = despace(" ".join(_lines(doc, PAGE_T207)))
    textann = despace(" ".join(_lines(doc, PAGE_ANNEX)))

    def spaced(n):
        return f"{n:,}".replace(",", " ")

    found = [k for k, (_m, _f, t) in T207.items() if spaced(t) in text207]
    say(len(found) == 8, f"all 8 région totals appear on Tableau 2.07's page ({len(found)})")
    found = [k for k, (_m, _f, t) in T207.items() if spaced(t) in textann]
    say(len(found) == 8, f"and again in the annex région table, printed p104 ({len(found)})")
    say(all(m + f == t for m, f, t in T207.values()),
        "every région's men + women = its total")
    say(sum(t for *_x, t in T207.values()) == RESIDENT,
        f"the 8 régions sum to the resident population {RESIDENT:,}")
    say(sum(m for m, *_x in T207.values()) == MEN, f"and the men to {MEN:,}")

    # 4. the région shares, population-weighted, reproduce the Ensemble row
    pops = {k: t for k, (_m, _f, t) in T207.items()}
    for i, c in enumerate(CATS):
        w = sum(T510[k][i] * pops[k] for k in T510) / RESIDENT
        good = abs(w - T510_ENSEMBLE[i]) <= 0.10
        say(good, f"{c:<17} population-weighted {w:6.2f} against the printed "
            f"{T510_ENSEMBLE[i]:.1f}")

    # 5. UNSD's counts: universe, and all fifteen of Tableau 5.09's cells
    unsd, live = unsd_counts()
    tot = unsd["Total"]
    if live:
        say(tot == UNSD_NATIONAL, "UNSD's cached Guinea 2014 row equals the transcription")
    say(sum(tot.values()) == RESIDENT, f"UNSD's six categories sum to {sum(tot.values()):,}")
    for area in ("Total", "Urban", "Rural"):
        if area not in unsd:
            continue
        u = unsd[area]
        say(u.get("Unknown") == COLLECTIVE[area],
            f"UNSD {area} `Unknown` {u.get('Unknown'):,} = Tableau 2.04/2.05 collective "
            f"households {COLLECTIVE[area]:,}")
        declared = sum(v for c, v in u.items() if c != "Unknown")
        say(declared == ORDINARY[area],
            f"UNSD {area} declared {declared:,} = ordinary households {ORDINARY[area]:,}")
        rec = [round(100.0 * u[k] / ORDINARY[area], 1) for k in UNSD]
        printed = [T509[area][CATS.index(UNSD[k])] for k in UNSD]
        say(rec == printed, f"Tableau 5.09 {area}: {printed} reproduced from UNSD's counts")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return {UNSD[k]: v for k, v in tot.items() if k in UNSD}, pops


def emit(national, pops):
    raw = {(r, c): T510[r][i] / 100.0 * pops[r] for r in T510 for i, c in enumerate(CATS)}
    scale = {}
    print("\n  per-religion rescale to UNSD's national count:")
    for c in CATS:
        s = sum(raw[(r, c)] for r in T510)
        scale[c] = national[c] / s
        print(f"    {c:<17} raw {s:>12,.0f}  x{scale[c]:.5f}  -> {national[c]:>10,}")
        # A category over 1% nationally is printed with enough digits that its rescale is a
        # rounding correction and nothing more; a large factor means a misread cell.
        if national[c] / ORDINARY["Total"] > 0.01 and abs(scale[c] - 1) > 0.02:
            raise SystemExit(f"{c} needs x{scale[c]:.4f}, too far from 1 for rounding")

    rows = []
    for r in T510:
        for i, c in enumerate(CATS):
            n = int(round(raw[(r, c)] * scale[c]))
            if n <= 0:
                continue
            rows.append({
                "geo_id": r, "geo_level": "region", "geo_name": r,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": (f"Tableau 5.10 pct={T510[r][i]:.1f}; Tableau 2.07 pop={pops[r]}; "
                         f"rescaled x{scale[c]:.5f} to UNSD table 28's national count"),
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/gn.py --fetch")
    doc = fitz.open(PDF)
    national, pops = check(doc)
    rows = emit(national, pops)

    total = sum(r["count"] for r in rows)
    print(f"\n  8 régions, {total:,} people ({total - ORDINARY['Total']:+,} against the "
          f"ordinary-household population), {total / 8:,.0f} each")
    by_cat, by_reg = {}, {}
    for r in rows:
        by_cat[r["source_category"]] = by_cat.get(r["source_category"], 0) + r["count"]
        by_reg.setdefault(r["geo_name"], {})[r["source_category"]] = r["count"]
    for c, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>11,}  {100.0 * n / total:6.2f}%  {c}")
    print(f"\n  {'région':<12}{'drawn':>11}{'resident':>11}{'ratio':>8}   shares as drawn")
    for reg, d in by_reg.items():
        s = sum(d.values())
        shares = "  ".join(f"{c[:4]} {100 * d.get(c, 0) / s:5.1f}" for c in CATS)
        print(f"  {reg:<12}{s:>11,}{pops[reg]:>11,}{s / pops[reg]:8.4f}   {shares}")
    nz = by_reg["N'Zérékoré"]
    print("\n  N'Zérékoré's share of each religion: "
          + ", ".join(f"{c} {100 * nz.get(c, 0) / by_cat[c]:.1f}%" for c in CATS))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
