"""Côte d'Ivoire — RGPH 2021, religion by district and région administrative.

Reads (or fetches) data/raw/ci/ and writes data/normalized/ci.csv.

**§11w recorded Côte d'Ivoire as the most valuable African lead still open, and the office
was never the way in.** `ins.ci` is a parked cPanel page that 404s every document it used to
serve; the office is now **ANStat** and `anstat.ci` answers 403 to every scripted request,
static PDFs included (`/assets/...` is behind the same Cloudflare rule as the app). Neither
wall had to be beaten. **The Wayback Machine has the file**, which is §11f's technique used
for a live host rather than a dead one.

AND THE ORACLE WAS TWO CENSUSES OUT OF DATE. UNSD table 28 lists Côte d'Ivoire at **2014**
with 11 categories. What is drawn here is the **RGPH 2021** — 29.4 million people against
2014's 22.7 million — published in ANStat's *Rapport thématique tome 1, État et structure de
la population*. §9k's Chile finding, repeated: the lead was stale in the useful direction.

THE ROUTE, AND THE TRAP IN IT. `web.archive.org/web/<ts>id_/<url>` returns the original
bytes. **The obvious capture is truncated and the right one is not.** The 2021-04 captures of
these files deliver **exactly 1,048,576 bytes (2^20) with no `%%EOF` trailer** — the archive
cut them at 1 MiB — while PyMuPDF still opens the head and reports a page count, so nothing
raises. Query the CDX for every capture, take the LARGEST `length`, and assert the trailer.
memory/reference_pdf_truncated_at_source in a new disguise: here the damage is the archive's
rather than the publisher's, and the file that is wrong is the one you would reach for first.

WHAT IS DRAWN. **33 units — 31 régions plus the autonomous districts of Abidjan and
Yamoussoukro** — 29,276,660 people in ordinary households, about 887,000 each. That is
Kenya's grain (§9o) and coarser than everything else African here except Zimbabwe. It is
drawn for the categories, exactly as Kenya was.

**`Harriste` IS THE REASON.** 140,482 people, 0.48%, and **no other source on this map counts
Harrism at all.** The Harrist Church follows William Wadé Harris, the Grebo preacher from
Liberia who walked the Ivorian and Ghanaian coast in 1913-15 and is usually credited with more
conversions than any missionary in African history. It is an African Initiated Church that
predates almost all the others, it is one of Côte d'Ivoire's four state-recognised
confessions, and it is counted here by name at régional level. See taxonomy/ci2021.py, which
adds the node.

THE TABLE IS PERCENTAGES AND THE MAGNITUDES COME FROM A SECOND TABLE — §3.4's move.
**Tableau 4.6** (p87-88) is *Répartition de la population des ménages ordinaires par district
/ région administrative selon la religion*, in percentages to **one decimal**. **Tableau 4.1**
(p81) is the same eight categories nationally **in counts**. **The annex** (p132-133) is
*Région administrative / Population totale / Superficie / Densité*, in counts, for exactly the
33 drawable units. So: shares from 4.6, magnitudes from 4.1, denominators from the annex, and
then each category is rescaled so the 33 units sum to its published national count. That
removes the one-decimal rounding from the national figures entirely and leaves it only in the
within-country distribution, which is where it cannot be helped.

**TWO NESTED TIERS IN ONE COLUMN — Serbia's §9p in a third country.** Tableau 4.6's first
column holds districts AND their régions interleaved, with nothing marking which is which:
`Bas-Sassandra` is a district and the `San-Pedro`, `Gboklè`, `Nawa` under it are its régions.
Summing the column double-counts the country. **The separator is not parsed out of the layout
— it is the annex**: a row is drawable if and only if its name appears in the population
table, which lists régions and the two autonomous districts and no other district. That is
self-checking in a way a hardcoded district list is not, and it lands on exactly 33.

**AND `Lacs` IS PRINTED TWICE.** Rows 17 and 22 of Tableau 4.6 are both labelled `Lacs`, with
different figures. The second one is **Lagunes** — its children in the table are
Agnéby-Tiassa, Grands-Ponts and La Mé, which are the Lagunes district's three régions, and
the real Lacs district's children (Bélier, Iffou, Moronou, N'Zi) sit under the first. Neither
is drawn, since both are districts, so the collision costs nothing here — but it would sink a
build that took the district tier, and §9p's rule is that a duplicated place name is a
finding rather than a typo. `check()` asserts it is still exactly two.

WHAT IS LOST, AND IT IS LARGE. `Autres religions chrétiennes` is **6,004,781 people, 20.5%**
— the second largest religious group in the country — and it is one undifferentiated cell.
The methodology text on p80 names the modalities the census actually collected: *Catholique,
Méthodiste/Protestante, **Evangélique**, **Céleste**, Harriste, Musulman, Animiste,
**Bouddhiste** et **Témoin de Jehova***. Four of those nine never get a cell anywhere in the
volume — not in Tableau 4.1 nationally, not in 4.6 by region, not in 4.7 by age. **The
Celestial Church of Christ is inside this cell**, and Benin (§9ai) counts it by name at
commune level, so this map can draw a Céleste geography in one country and not in its
neighbour. §3.9's trade, made by the office, and made worse than usual.

Usage:
    python sources/ci.py --fetch    one ~34 MB PDF from the Wayback Machine
    python sources/ci.py            normalise from data/raw/ci/
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
RAW = os.path.join(ROOT, "data", "raw", "ci")
OUT = os.path.join(ROOT, "data", "normalized", "ci.csv")

SOURCE_ID = "ci_rgph_2021_tome1"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# ANStat, Rapport thématique tome 1. Live at
# https://www.anstat.ci/assets/publications/files/rgpg_tom1.pdf, which is 403 to every
# scripted client; this is the archive's copy of that exact URL.
ORIGINAL = "https://www.anstat.ci/assets/publications/files/rgpg_tom1.pdf"
PDF = os.path.join(RAW, "rgpg_tom1.pdf")

# 0-based page indices. The volume is 151 pages.
PAGES_NATIONAL = (80,)          # Tableau 4.1, counts by religion
PAGES_REGIONAL = (86, 87)       # Tableau 4.6, percentages by district/région
PAGES_ANNEX = (131, 132)        # population totale by région

# Tableau 4.6's columns, in printed order. `Ensemble Chrétien` and `Total` are subtotals of
# the others and are read only so the row can be checked; they are never emitted.
T46_COLUMNS = [
    "Catholique", "Méthodiste/Protestant", "Harriste",
    "Autres religions chrétiennes", "Ensemble Chrétien", "Musulmane",
    "Animiste", "Autres religions", "Sans religion", "ND", "Total",
]
SUBTOTALS = {"Ensemble Chrétien", "Total"}
DRAWN_CATEGORIES = [c for c in T46_COLUMNS if c not in SUBTOTALS]

# Tableau 4.6 names its two autonomous districts differently from the annex.
ALIASES = {
    "yamoussoukro": "districtdeyamoussoukro",
    "districtdabidjan": "districtdabidjan",
}

UNITS = 33
CENSUS_HOUSEHOLD_POPULATION = 29_276_660   # ordinary households, the religion universe
LACS_DUPLICATES = 2

PCT = re.compile(r"^\d{1,3},\d$")
BIG = re.compile(r"^\d{1,3}(?: \d{3})+$")

# THE SAME DOCUMENT USES TWO DIFFERENT THOUSANDS SEPARATORS. Tableau 4.1 (p81) groups its
# digits with U+2009 THIN SPACE — `5 784 899` — while the annex table (p132) uses
# an ordinary U+0020. A regex written against one silently matches nothing on the other, and
# the failure looks like "that table is not on this page" rather than like an encoding
# problem. Every line is folded through this before any pattern is tried.
SPACES = dict.fromkeys(
    [0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F, 0x2000, 0x2001,
     0x2002, 0x2003, 0x2004, 0x2005, 0x2006], " ")


def despace(s):
    """Fold every Unicode space variant to U+0020, then squeeze runs."""
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def norm(s):
    """Fold a place name to a comparable key: accents, case and punctuation out."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    import json
    import urllib.parse
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) > 20_000_000:
        print("already have", PDF)
        return

    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0 Safari/537.36"}

    # Every capture, so the LARGEST can be taken. The first-listed one is truncated at
    # exactly 2^20 bytes and opens anyway.
    target = ORIGINAL.split("://", 1)[1]
    cdx = ("https://web.archive.org/cdx/search/cdx"
           f"?url={urllib.parse.quote(target, safe='')}"
           "&output=json&fl=timestamp,statuscode,length&limit=300")
    req = urllib.request.Request(cdx, headers=ua)
    with urllib.request.urlopen(req, timeout=300) as r:
        rows = json.load(r)
    caps = [(int(x[2]), x[0]) for x in rows[1:]
            if x[1] == "200" and str(x[2]).isdigit()]
    if not caps:
        raise SystemExit("no 200 capture of the tome in the Wayback CDX")
    length, ts = max(caps)
    print(f"{len(caps)} captures; taking {ts} (stored {length:,} bytes)")

    url = f"https://web.archive.org/web/{ts}id_/{ORIGINAL}"
    req = urllib.request.Request(url, headers=ua)
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = r.read()

    # §5a, and the archive's own truncation on top of it.
    if body[:4] != b"%PDF":
        raise SystemExit(f"not a PDF: starts {body[:16]!r}")
    if len(body) == 1 << 20:
        raise SystemExit("exactly 2^20 bytes — the archive truncated this capture; "
                         "pick a larger one")
    if not body.rstrip().endswith(b"%%EOF"):
        raise SystemExit(f"no %%EOF trailer on {len(body):,} bytes — truncated")
    with open(PDF, "wb") as fh:
        fh.write(body)
    print(f"wrote {PDF} ({len(body):,} bytes)")


def _lines(doc, pno):
    out = [despace(ln) for ln in doc.load_page(pno).get_text().splitlines()]
    return [ln for ln in out if ln]


def read_national(doc):
    """Tableau 4.1 — counts by religion, nationally. The magnitude anchor.

    Layout per row: name, M effectif, M poids, F effectif, F poids, TOTAL effectif,
    100,0, RM. One row prints the female effectif and its weight in a single cell, so
    the numbers are collected by scanning tokens rather than by fixed offsets.
    """
    out = {}
    for pno in PAGES_NATIONAL:
        lines = _lines(doc, pno)
        i = 0
        while i < len(lines):
            name = lines[i]
            if BIG.match(name) or PCT.match(name):
                i += 1
                continue
            j, nums = i + 1, []
            while j < len(lines):
                tok = lines[j]
                parts = tok.split()
                # A merged cell like "1 554 499     42,2" -> two tokens.
                if BIG.match(tok) or PCT.match(tok):
                    nums.append(tok)
                elif len(parts) > 1 and BIG.match(" ".join(parts[:-1])) \
                        and PCT.match(parts[-1]):
                    nums.append(" ".join(parts[:-1]))
                    nums.append(parts[-1])
                else:
                    break
                j += 1
            bigs = [int(n.replace(" ", "")) for n in nums if BIG.match(n)]
            if len(bigs) >= 3:
                out[name] = bigs[2]      # the Total effectif
                i = j
            else:
                i += 1
    return out


def read_regional(doc):
    """Tableau 4.6 — 11 percentages per row, districts and régions interleaved."""
    out = []
    for pno in PAGES_REGIONAL:
        lines = _lines(doc, pno)
        i = 0
        while i < len(lines):
            if PCT.match(lines[i]):
                i += 1
                continue
            j, nums = i + 1, []
            while j < len(lines) and PCT.match(lines[j]):
                nums.append(float(lines[j].replace(",", ".")))
                j += 1
            if len(nums) == len(T46_COLUMNS):
                out.append((lines[i], nums))
                i = j
            else:
                i += 1
    return out


def read_population(doc):
    """The annex — région, population totale. Returns {name: [values]}.

    TWO OTHER TABLES SHARE THESE PAGES and their rows have the same shape, so a
    name->value dict would silently keep whichever came last. An UPPERCASE table is a
    different universe (its own total is 11,225,273), and an urban/rural table
    contributes `Abidjan ville`, `Autres villes`, `Ensemble urbain` and `Rural` — each
    of which appears TWICE with DIFFERENT numbers. None of them is a région, so none
    reaches the drawn set; the values are kept as lists anyway so `check()` can assert
    that no DRAWN unit is ambiguous rather than trusting that.
    """
    out = {}
    for pno in PAGES_ANNEX:
        lines = _lines(doc, pno)
        i = 0
        while i < len(lines) - 1:
            name, nxt = lines[i], lines[i + 1]
            if (not BIG.match(name) and BIG.match(nxt)
                    and name != name.upper() and norm(name) != "total"):
                out.setdefault(name, []).append(int(nxt.replace(" ", "")))
                i += 2
            else:
                i += 1
    return out


def build():
    import fitz

    doc = fitz.open(PDF)
    national = read_national(doc)
    regional = read_regional(doc)
    population = read_population(doc)
    return doc, national, regional, population


def check(doc, national, regional, population):
    ok = True
    print("Côte d'Ivoire — RGPH 2021, religion by région administrative\n")

    good = doc.page_count == 151
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the tome is {doc.page_count} pages "
          f"(expected 151)")

    # 1. the duplicated place name, asserted rather than noticed (§9p)
    names = [n for n, _ in regional]
    lacs = sum(1 for n in names if norm(n) == "lacs")
    good = lacs == LACS_DUPLICATES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} `Lacs` appears {lacs}x in Tableau 4.6 "
          f"(expected {LACS_DUPLICATES}; the second is Lagunes) — both are "
          f"districts, so neither is drawn")

    # 2. the drawable set is the intersection with the annex, and it is 33
    pop_keys = {norm(k): k for k in population}
    drawn = [(n, v) for n, v in regional
             if ALIASES.get(norm(n), norm(n)) in pop_keys]
    good = len(drawn) == UNITS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(drawn)} of {len(regional)} "
          f"Tableau 4.6 rows appear in the population annex (expected {UNITS} "
          f"= 31 régions + 2 autonomous districts)")
    if not good:
        miss = [n for n, _ in regional
                if ALIASES.get(norm(n), norm(n)) not in pop_keys]
        print(f"        not in the annex: {miss}")

    # 2b. NO DRAWN UNIT MAY BE AMBIGUOUS. Two other tables share the annex pages and
    #     four of their rows repeat with different numbers; a dict would have kept the
    #     last one silently. None is a région — asserted, not assumed.
    ambiguous = []
    for n, _ in drawn:
        key = pop_keys[ALIASES.get(norm(n), norm(n))]
        if len(set(population[key])) > 1:
            ambiguous.append((key, sorted(set(population[key]))))
    good = not ambiguous
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} no drawn unit appears twice in the annex "
          f"with different values ({len(ambiguous)} ambiguous)")
    for key, vals in ambiguous:
        print(f"        {key}: {vals}")

    repeated = sorted(k for k, v in population.items() if len(set(v)) > 1)
    print(f"  -- {len(repeated)} annex names DO repeat with different values and "
          f"are all aggregates, not régions: {repeated}")

    # 3. every row's own categories sum to its printed Total
    i_tot = T46_COLUMNS.index("Total")
    i_chr = T46_COLUMNS.index("Ensemble Chrétien")
    worst = 0.0
    for n, v in regional:
        s = sum(v[k] for k, c in enumerate(T46_COLUMNS) if c not in SUBTOTALS)
        worst = max(worst, abs(s - v[i_tot]))
    # Nine cells at one decimal can each be off by 0.05, so the row bound is 0.45.
    good = worst <= 0.45
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} every row's 9 categories sum to its "
          f"printed Total within {worst:.2f} pp (bound 0.45 = 9 cells x 0.05)")

    # 4. the Christian subtotal is the sum of its four parts, on every row
    parts = ["Catholique", "Méthodiste/Protestant", "Harriste",
             "Autres religions chrétiennes"]
    worst = max(abs(sum(v[T46_COLUMNS.index(p)] for p in parts) - v[i_chr])
                for _, v in regional)
    good = worst <= 0.20
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} `Ensemble Chrétien` equals its four "
          f"parts on every row within {worst:.2f} pp — an INTERNAL check the "
          f"table volunteers")

    # 5. the population annex reproduces the published resident population
    drawn_pop = sum(population[pop_keys[ALIASES.get(norm(n), norm(n))]][0]
                    for n, _ in drawn)
    print(f"  -- the {len(drawn)} DRAWN units sum to {drawn_pop:,}, the RGPH 2021 "
          f"resident population (the annex also holds urban/rural aggregate rows, "
          f"excluded above)")

    # 6. the national table carries every drawn category, in counts
    missing = [c for c in DRAWN_CATEGORIES if c not in national]
    good = not missing
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} Tableau 4.1 gives a national COUNT for "
          f"all {len(DRAWN_CATEGORIES)} drawn categories "
          f"({len(missing)} missing{': ' + str(missing) if missing else ''})")
    if not good:
        print(f"        parsed national rows: {sorted(national)}")
        raise SystemExit("cannot anchor magnitudes without Tableau 4.1")

    nat_sum = sum(national[c] for c in DRAWN_CATEGORIES)
    print(f"  -- the 9 national counts sum to {nat_sum:,} against the published "
          f"ordinary-household population {CENSUS_HOUSEHOLD_POPULATION:,} "
          f"({nat_sum - CENSUS_HOUSEHOLD_POPULATION:+,})")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return drawn, pop_keys


def emit(drawn, pop_keys, national, population):
    """§3.4: shares from Tableau 4.6, denominators from the annex, magnitudes from
    Tableau 4.1. Each category is then rescaled so the 33 units sum to its published
    national count, which removes the 1-dp rounding from every national figure."""
    raw = {}
    for name, pcts in drawn:
        key = pop_keys[ALIASES.get(norm(name), norm(name))]
        pop = population[key][0]
        for c in DRAWN_CATEGORIES:
            v = pcts[T46_COLUMNS.index(c)] / 100.0 * pop
            raw[(name, c)] = v

    scale = {}
    for c in DRAWN_CATEGORIES:
        s = sum(raw[(n, c)] for n, _ in drawn)
        scale[c] = (national[c] / s) if s else 0.0
    print("\n  per-category rescale to the published national count:")
    for c in DRAWN_CATEGORIES:
        print(f"    {c:<32} x{scale[c]:.5f}   -> {national[c]:>11,}")

    rows = []
    for name, _ in drawn:
        key = pop_keys[ALIASES.get(norm(name), norm(name))]
        for c in DRAWN_CATEGORIES:
            n = int(round(raw[(name, c)] * scale[c]))
            if n <= 0:
                continue
            rows.append({
                "geo_id": key, "geo_level": "region", "geo_name": key,
                "source_category": c, "count": n, "basis": BASIS,
                "year": YEAR, "source_id": SOURCE_ID,
                "note": f"pct={raw[(name, c)] * 100 / max(population[key][0], 1):.1f}; "
                        f"pop={population[key][0]}; rescaled_to_national",
            })
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/ci.py --fetch")

    doc, national, regional, population = build()
    drawn, pop_keys = check(doc, national, regional, population)
    rows = emit(drawn, pop_keys, national, population)

    total = sum(r["count"] for r in rows)
    print(f"\n  {len(drawn)} units drawn, {total:,} people, "
          f"{total / len(drawn):,.0f} each. Categories, national:")
    for c in DRAWN_CATEGORIES:
        n = national[c]
        print(f"    {n:>11,}  {100.0 * n / total:6.2f}%  {c}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
