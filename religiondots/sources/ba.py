"""Bosnia and Herzegovina — BHAS, Census 2013, religion by municipality.

Reads (or fetches) data/raw/ba/ and writes data/normalized/ba.csv.

One PDF, **`RezultatiPopisa_BS.pdf`** — *Popis stanovništva, domaćinstava i stanova u
Bosni i Hercegovini, 2013. Rezultati popisa* (BHAS, June 2016), 12.5 MB, 268 pages, no
key, no login, no wall. Its **table 4.3**, *Stanovništvo prema izjašnjavanju o
vjeroispovijesti i spolu, nivo općine u BiH*, is religion by municipality: **142 units,
3,531,159 people, eight categories**, over twelve printed pages.

`sources.md` §11c recorded this country as *"reachable, not quick — popis.gov.ba is a React
SPA with no data endpoint found; the 2013 final results are 12-19 MB PDF books"*, and left
it there. Both halves are true and neither mattered. **The SPA has no data endpoint because
there is nothing behind it to serve: the books are static files under `/popis2013/doc/`,
and the directory listing 403s while the files themselves are open.** A 403 on a directory
is a statement about the directory. §12's rule to grep the bundle was run here and found
zero `/api` routes in all 125 KB of it, which is the *correct* negative result rather than
a failed search — and the file was one guessed path away the whole time.

**THE PARTITION IS EXACT IN BOTH DIRECTIONS AND THAT IS THE WHOLE RECONCILIATION.** The
eight categories sum to each municipality's own total exactly, and the 142 municipalities
sum to the published national row exactly, category by category, with a discrepancy of zero
everywhere. `check()` asserts both. Nothing is suppressed, rounded or prorated; BHAS
publishes single people.

**Two parse traps, and the second is the one that would have shipped quietly.**

1.  **A municipality name can wrap onto a second line**, and the wrap is marked only by a
    trailing space on the first fragment — `BOSANSKA ` / `KRUPA`. A parser that treats one
    line as one name gets `KRUPA`, which then fails to join to anything and looks like a
    boundary problem rather than a parse problem. Names are accumulated until the
    `Uk./Tot.` marker rather than read line by line.
2.  **Tables 4.1 and 4.2 are the same shape as 4.3 and sit immediately before it.** 4.1 is
    the entity level (BiH, FBiH, RS, Brčko) and 4.2 the cantons of the Federation, and both
    use identical headers and an identical `Uk./Tot.` / `M/M` / `Ž/F` row structure. Reading
    a generous page window picks all three up, sums to **5,583,946 against a country of
    3,531,159**, and *still passes a per-row partition check*, because each of those rows is
    internally consistent. The page window is therefore pinned and the national total is
    asserted — a check that only looks at rows cannot see this error.

**THE SEX ROWS ARE DROPPED AND ONLY `Uk./Tot.` IS READ.** Each unit publishes three rows;
summing M/M and Ž/F would double the country. They are skipped explicitly rather than by
falling off the end of a pattern.

**`geo_id` IS A FOLDED NAME, BECAUSE THE TABLE PUBLISHES NO CODE.** BHAS prints the
municipality name and nothing else. The fold is asserted unique across all 142, so a future
vintage that adds a colliding name fails here rather than silently merging two
municipalities.

**AND THE VINTAGE IS 2013, WHICH IS THE LAST CENSUS.** BiH has run none since. The counts
are twelve years old at the time of writing and the country's religious geography has not
been rearranged since 2013 — see `sources/ba.md` §5 for why that sentence is doing real
work here, and spec §14 and sources.md §11j for the test it has to pass.

Usage:
    python sources/ba.py --fetch    one GET, 12.5 MB
    python sources/ba.py            normalise from data/raw/ba/
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
RAW = os.path.join(ROOT, "data", "raw", "ba")
OUT = os.path.join(ROOT, "data", "normalized", "ba.csv")

SOURCE_ID = "ba_census_2013"
YEAR = 2013
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = "https://popis.gov.ba/popis2013/doc/RezultatiPopisa_BS.pdf"
PDF = os.path.join(RAW, "RezultatiPopisa_BS.pdf")

# Table 4.3 occupies printed pages 70-81, which are PDF pages 72-83 (1-based). PINNED, not
# searched: tables 4.1 (entity) and 4.2 (canton) are the same shape and immediately before,
# and a window that includes them passes every per-row check while tripling the country.
PAGE_FIRST = 72
PAGE_LAST = 83

# Column order of table 4.3, left to right. The first is the unit's own total and is not a
# category; taxonomy/ba2013.py excludes it.
CATEGORIES = [
    "Ukupno",             # Total
    "Islamska",           # Islamic
    "Katolička",          # Catholic
    "Pravoslavna",        # Orthodox
    "Agnostik",           # Agnostic
    "Ateist",             # Atheist
    "Nisu se izjasnili",  # Did not declare
    "Ostali",             # Other
    "Bez odgovora",       # No answer
]

# BHAS's own published national row, table 4.1. Asserted against the sum of the 142.
NATIONAL = {
    "Ukupno": 3_531_159,
    "Islamska": 1_790_454,
    "Katolička": 536_333,
    "Pravoslavna": 1_085_760,
    "Agnostik": 10_816,
    "Ateist": 27_853,
    "Nisu se izjasnili": 32_700,
    "Ostali": 40_655,
    "Bez odgovora": 6_588,
}
EXPECTED_UNITS = 142

# Header and furniture lines inside the table pages. Anything here resets the name buffer.
FURNITURE = {
    "Teritorija", "Spol", "Ukupno", "Vjeroispovijest", "Islamska", "Katolička",
    "Pravoslavna", "Agnostik", "Ateist", "Nisu se", "izjasnili", "Ostali", "Bez",
    "odgovora", "Area", "Sex", "Total", "Religion", "Islamic", "Catholic", "Orthodox",
    "Agnostic", "Atheist", "Did not declared", "Did not", "declared", "Other", "No answer",
    "DEMOGRAFIJA", "DEMOGRAPHY",
}

SEX_TOTAL = "Uk./Tot."
SEX_OTHER = ("M/M", "Ž/F")

NUM = re.compile(r"^\d{1,3}(?:\.\d{3})*$|^\d+$")


def fold(name):
    """Municipality name -> a stable ASCII key.

    Bosnian digraphs first: BHAS writes `BANjA LUKA` and `BIJELjINA` with a lowercase j
    inside an otherwise uppercase name, because Nj is one letter. Casefolding without
    handling that is harmless here, but `Đ`/`đ` does not decompose under NFKD the way the
    other diacritics do and has to be spelled out.
    """
    s = name.replace("Đ", "DJ").replace("đ", "dj")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]", "", s).upper()


def _fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) > 10_000_000:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    print("GET ", URL)
    # TLS verifies normally on popis.gov.ba; nothing is disabled here (§9h).
    r = requests.get(URL, headers=ua, timeout=600)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise SystemExit(f"{URL} did not return a PDF -- first bytes {r.content[:16]!r}. "
                         "§5a: a 200 is not a download.")
    with open(PDF, "wb") as fh:
        fh.write(r.content)
    print(f"wrote {PDF}  {len(r.content):,} bytes")


def _lines():
    import fitz

    if not os.path.exists(PDF):
        raise SystemExit(f"missing {PDF} -- run: python sources/ba.py --fetch")
    doc = fitz.open(PDF)
    if doc.page_count != 268:
        raise SystemExit(f"{PDF} has {doc.page_count} pages, expected 268. The book has "
                         "been re-issued and PAGE_FIRST/PAGE_LAST no longer locate table "
                         "4.3 -- re-find it before trusting anything below.")
    # Confirm the window really is 4.3 and not 4.1/4.2, by its own printed caption.
    head = doc[PAGE_FIRST - 1].get_text()
    if "nivo općine u BiH" not in head or "4.3." not in head:
        raise SystemExit(f"PDF page {PAGE_FIRST} is not the head of table 4.3 -- found:\n"
                         f"{head[:400]}")
    out = []
    for p in range(PAGE_FIRST - 1, PAGE_LAST):
        for ln in doc[p].get_text().splitlines():
            if ln.strip():
                out.append(ln)
    return out


def parse():
    """Table 4.3 -> {name: [9 ints]} in the column order of CATEGORIES."""
    lines = _lines()
    rows = {}
    buf = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == SEX_TOTAL:
            name = " ".join(x.strip() for x in buf).strip()
            buf = []
            nums = []
            j = i + 1
            while j < len(lines) and len(nums) < len(CATEGORIES) and NUM.match(lines[j].strip()):
                nums.append(int(lines[j].strip().replace(".", "")))
                j += 1
            if not name:
                raise SystemExit(f"a {SEX_TOTAL} row at line {i} has no name above it")
            if len(nums) != len(CATEGORIES):
                raise SystemExit(f"{name}: read {len(nums)} numbers, expected "
                                 f"{len(CATEGORIES)} -- {nums}")
            if name in rows:
                raise SystemExit(f"duplicate municipality name {name!r}")
            rows[name] = nums
            i = j
            continue
        if s in SEX_OTHER:
            # the per-sex rows: same nine columns again, deliberately dropped
            buf = []
            j = i + 1
            while j < len(lines) and NUM.match(lines[j].strip()):
                j += 1
            i = j
            continue
        if s in FURNITURE or NUM.match(s) or s.startswith("www.") or s.startswith("4."):
            buf = []
            i += 1
            continue
        buf.append(lines[i])
        i += 1
    return rows


def check(rows):
    """Assert what should be exact (spec §12). Nothing here is a tolerance."""
    if len(rows) != EXPECTED_UNITS:
        raise SystemExit(f"{len(rows)} municipalities, expected {EXPECTED_UNITS}")

    # 1. every unit's categories partition its own total, exactly
    bad = [(n, v[0], sum(v[1:])) for n, v in rows.items() if sum(v[1:]) != v[0]]
    if bad:
        raise SystemExit(f"{len(bad)} units whose categories do not sum to their total: "
                         f"{bad[:5]}")

    # 2. the 142 sum to BHAS's published national row, category by category, exactly
    for idx, cat in enumerate(CATEGORIES):
        got = sum(v[idx] for v in rows.values())
        want = NATIONAL[cat]
        if got != want:
            raise SystemExit(f"{cat}: units sum to {got:,}, national row says {want:,} "
                             f"(diff {got - want:+,}). If this is off by ~2x the country, "
                             f"the page window has picked up table 4.1 or 4.2.")

    # 3. the fold is the key, so it has to be injective
    seen = {}
    for n in rows:
        k = fold(n)
        if k in seen:
            raise SystemExit(f"fold collision: {n!r} and {seen[k]!r} both -> {k!r}")
        seen[k] = n

    print(f"  {len(rows)} municipalities, partition exact per unit and per category "
          f"against the national row, {len(seen)} distinct keys")


def normalise(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for name in sorted(rows):
            vals = rows[name]
            for cat, v in zip(CATEGORIES, vals):
                w.writerow([fold(name), "municipality", name, cat, v,
                            BASIS, YEAR, SOURCE_ID, ""])
                n += 1
    print(f"wrote {OUT}  {n:,} rows")


def main():
    if "--fetch" in sys.argv:
        _fetch()
    rows = parse()
    check(rows)
    normalise(rows)

    tot = sum(v[0] for v in rows.values())
    isl = sum(v[1] for v in rows.values())
    print(f"\n  {tot:,} people, {isl:,} Muslim ({isl / tot:.1%})")
    top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:5]
    for n, v in top:
        print(f"    {n:<22} {v[0]:>9,}  islam {v[1] / v[0]:>5.1%}  "
              f"cath {v[2] / v[0]:>5.1%}  orth {v[3] / v[0]:>5.1%}")


if __name__ == "__main__":
    main()
