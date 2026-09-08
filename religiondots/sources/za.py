"""South Africa — religion and Christian denomination, Community Survey 2016, by province.

Writes data/normalized/za.csv.

**THE RELEASE THIS BUILDS FROM IS NOT THE CENSUS, AND THAT IS THE WHOLE POINT.** Stats SA's
Census 2022 statistical release P0301.4 publishes religion for all nine provinces over
eleven substantive categories plus `Not Specified`, openly, and its `Christianity` is one
undivided cell holding 83.6% of the country. The **Community Survey 2016** provincial
profiles publish the same eleven *and* a second table splitting Christianity into fourteen
denominations. So: **24 usable categories here against the census's 11**, counting only the
substantive ones on both sides. On a country that is the historic centre of the African
Independent Church movement, that second table is the country.

The cost is six years and a survey rather than a census, and §3.1 forbids mixing the two.
The two releases are NOT interchangeable even where their category lists match verbatim:
CS 2016 puts `No religious affiliation/belief` at 10.9% of answers and Census 2022 at 2.9%,
and `Traditional African religion` moves 4.5% -> 7.8% the other way. Six years does not do
that; the question was administered differently. So nothing here is rescaled onto 2022
totals, and §3.1a is why.

**NINE PROVINCES IS THE OPEN CEILING FOR THIS VARIABLE AND IT WAS CHECKED, not assumed.**
See sources/za.md §2 for the full sweep. In short: Report 03-01-84 *Cultural dynamics in
South Africa*, named as the likely home of a finer cut, is province-only and **coarser**
(8 categories); Census 2011 asked no religion question at all, which kills Wazimap and every
2011 municipal product; and Stats SA's own keyless Census 2022 dissemination API serves 24
topics down to Main Place with religion among none of them. In the Census 2022 provincial
profiles essentially every other variable is tabulated by district and local municipality
and religion alone is not, which reads as a decision rather than an oversight. The finer
data does exist in the CS 2016 microdata (district and local municipality, same tables) and
that is account-walled; see ask/ and sources/za.md §2.

FIVE THINGS THIS PARSER HAD TO SURVIVE, all of them silent failures:

  * **The table NUMBER is not stable across the nine reports.** Free State, KwaZulu-Natal,
    Mpumalanga, North West and Northern Cape use 2.10a/2.10b; Eastern Cape and Gauteng use
    2.9a/2.9b; Limpopo uses 2.7/2.8 with no letter at all; Western Cape uses 2.11a/2.11b.
    Gauteng writes its own caption as `Table 2.9 a:` with a space inside the number. So the
    anchor is the caption TEXT (`religious affiliation`, `Christian denomination`) and the
    number is never matched on.
  * **The TOTAL row is usually labelled with the PROVINCE NAME, not "Total".** Eastern Cape,
    KwaZulu-Natal, Limpopo, Mpumalanga and North West all do this. Read as a data row it
    doubles the table and every share is halved, with no error anywhere.
  * **Northern Cape prints Bahaism as a bare dash** in both columns rather than as 0. Skip
    it and the province quietly has ten categories where the other eight have eleven.
  * **Gauteng spells it `Buddism`** and KwaZulu-Natal and Limpopo write `Seventh-Day
    Adventist` against everyone else's `Seventh Day Adventist`. Folded, and the fold is
    asserted to leave exactly 11 and 14 distinct categories.
  * **Both "Other" rows exist and mean different things** -- one is a non-Christian religion,
    one is a Christian denomination. Every category is therefore emitted prefixed,
    `Religion: ...` / `Christian: ...`, which is `sources/sz.py`'s convention. A bare join
    on "Other" would put 1.5M people of other faiths inside Christianity.

**NORTH WEST'S TABLE 2.10b DOES NOT ADD UP AND IT IS THE REPORT THAT IS WRONG, not the
parse.** Its fourteen printed rows sum to 3,072,039 against its own printed total of
3,408,521, and its printed percentages sum to 90.1 rather than 100.0; 336,482 people, 9.9%
of the province's Christians, are in no denomination row. Report 03-01-11's `Other` cell
reads 21 873, which is character for character the `Do not know` figure in that same table's
own footnote, so the likeliest story is that the wrong number was set in the Other row. That
is a guess and it is not acted on. Instead the residual joins the same
`Christian: Denomination not reported` row that all nine provinces already carry (see
below), where it is visible rather than invented.

THE DENOMINATION-NOT-REPORTED ROW IS NOT AN INVENTION, it is the reconciliation. Table
2.10b excludes its own `Do not know` and `Unspecified` and prints both, and the difference
between 2.10a's Christianity cell and 2.10b's total reproduces them. Checked against the
printed footnote in all nine reports: **exact in four** (Northern Cape 7,989, Free State
8,998, KwaZulu-Natal 18,874, Limpopo 8,346), **within one person in three** (Eastern Cape,
Gauteng, Mpumalanga), **unverifiable in one** because Report 03-01-07 prints no exclusion
note under Western Cape's table at all, and **broken in one**, North West, above. So Western
Cape's 46,055 is inferred rather than confirmed and is the largest residual of the eight
sound provinces. Those people are Christians whose denomination was not established, so
they are emitted rather than dropped, and they resolve to bare `christianity`.

Usage:
    python sources/za.py --fetch    nine PDFs, ~58 MB total, from cs2016.statssa.gov.za
    python sources/za.py            rebuild from data/raw/za/
"""

import csv
import os
import re
import ssl
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "za")
OUT = os.path.join(ROOT, "data", "normalized", "za.csv")

SOURCE_ID = "za_cs2016"
YEAR = 2016
BASIS = "self_id"

# cs2016.statssa.gov.za serves the nine provincial profiles as static WordPress uploads.
#
# ACCESS, AND IT IS NOT WHAT sources.md §11ag ASSUMED. The Stats SA hosts sit behind
# Imperva and their HTML pages are unreachable from a script -- `?page_id=` and `?p=`
# listings come back as a ~1 KB `_Incapsula_Resource` stub for curl AND for WebFetch, and a
# browser User-Agent does not help. But these PDF uploads are NOT walled. What blocks a
# plain `curl` on this host is the TLS chain, not the bot wall: it presents a self-signed
# intermediate and curl exits 60 before it ever sends the request, which reads exactly like
# a dead host. Downloading them needs the certificate check relaxed and nothing else.
#
# So the working method for anything on statssa.gov.za is: find the PDF URL (search, or a
# known pattern like this one), then fetch the PDF directly. Do not try to read the listing.
BASE = "https://cs2016.statssa.gov.za/wp-content/uploads/2018/07/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# geo_id is minted from the OFFICIAL Stats SA province code (1-9), which is the order every
# Stats SA release prints provinces in and is NOT the order the nine PDFs happen to sit in a
# directory listing. sources/za_geo.py checks it against the COD boundary file independently.
#
# THE REPORT NUMBERS ARE READ OUT OF THE PDFs AND ASSERTED, not trusted from here. They do
# not run in province-code order and they are not guessable: Western Cape is 03-01-07 and
# Mpumalanga is 03-01-13, which is the number an alphabetical or a code-order guess gives
# to Western Cape. Getting one wrong would mis-cite a report in the CSV's `note` column and
# nothing downstream would notice, so `read_province` checks each one against the running
# header of the file it just opened.
PROVINCES = [
    ("ZA01", "WesternCape",  "Western Cape",  "03-01-07"),
    ("ZA02", "EasternCape",  "Eastern Cape",  "03-01-08"),
    ("ZA03", "NorthernCape", "Northern Cape", "03-01-14"),
    ("ZA04", "FreeState",    "Free State",    "03-01-12"),
    ("ZA05", "KZN",          "KwaZulu-Natal", "03-01-10"),
    ("ZA06", "NorthWest",    "North West",    "03-01-11"),
    ("ZA07", "Gauteng",      "Gauteng",       "03-01-09"),
    ("ZA08", "Mpumalanga",   "Mpumalanga",    "03-01-13"),
    ("ZA09", "Limpopo",      "Limpopo",       "03-01-15"),
]

CAPTION = re.compile(r"^Table\s+\d+\.\d+\s*[ab]?\s*:\s*(.*)$")
COUNT = re.compile(r"^-?[\d][\d\s   ]*$")
PCT = re.compile(r"^\d{1,3},\d$")
DASH = {"-", "–", "—"}

# Header cells that would otherwise be glued onto the first row's label -- PyMuPDF emits a
# table's header as ordinary lines, so `Christian domination` / `Catholic` reads as one
# two-line label.
#
# THE SET HAS TO BE PER-TABLE AND THAT IS NOT FUSSiness. Northern Cape heads its
# denomination table with the bare word `Christianity`, which is ALSO a real row label in
# the religion table; a single shared set either loses Northern Cape's Catholic row or
# deletes every province's Christianity row, and each failure is silent. Western Cape's
# `Christian domination` is Report 03-01-07's own typo, kept verbatim so the match works.
_HEADER_COMMON = {"number", "n", "%", "percentage", "religious affiliation",
                  "religious affiliation/belief"}
HEADER = {
    "a": _HEADER_COMMON | {"religion"},
    "b": _HEADER_COMMON | {"christian denomination", "christian denominations",
                           "christian domination", "denomination", "christianity"},
}

# The canonical answer sets. Asserted after folding, per province, both tables.
RELIGIONS = [
    "Christianity", "Islam", "Traditional African religion", "Hinduism", "Buddhism",
    "Bahaism", "Judaism", "Atheism", "Agnosticism", "No religious affiliation/belief",
    "Other",
]
DENOMINATIONS = [
    "Catholic", "Anglican/Episcopalian", "Baptist", "Lutheran", "Methodist",
    "Presbyterian", "Pentecostal/Evangelistic",
    "African Independent Church/African Initiated Church", "Jehovah's Witness",
    "Seventh Day Adventist", "Mormon", "Reformed church",
    "Just a Christian/non-denominational", "Other",
]

# Spelling variants seen across the nine reports, folded to the canonical label above.
# `buddism` is Gauteng's typo and is in the CS 2016 microdata codebook too, so it is the
# survey's own spelling rather than a typesetting slip in one report.
VARIANTS = {
    "buddism": "Buddhism",
    "seventh-day adventist": "Seventh Day Adventist",
    "african independent church/african initiated church": (
        "African Independent Church/African Initiated Church"),
    "african independent church/african initiated church": (
        "African Independent Church/African Initiated Church"),
    "african independent church/african initiated church": (
        "African Independent Church/African Initiated Church"),
    "jehovahs witness": "Jehovah's Witness",
    "reformed church": "Reformed church",
    "just a christian/non-denominational": "Just a Christian/non-denominational",
}

NOT_REPORTED = "Denomination not reported"


def fold(label):
    """Strip the parenthesised exemplar list, the smart apostrophes and the case.

    The exemplar lists are long and differ between reports -- Free State prints
    `Pentecostal/Evangelistic (e.g. Assemblies of God; ...)` where North West prints the
    bare word -- so they cannot be part of the key. They are preserved verbatim in the
    per-province `note` column instead, because they are what tells a reader what the
    African Independent Church row actually contains.
    """
    s = str(label)
    s = s.replace("‘", "'").replace("’", "'").replace("�", "'")
    s = re.sub(r"\((?:e\.?g\.?|eg)[^)]*\)?", " ", s, flags=re.I)
    s = re.sub(r"\(.*?\)", " ", s)
    s = " ".join(s.split()).strip(" .:,")
    return s


def key(label):
    s = fold(label).lower().replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", s).strip()


CANON = {}
for _c in RELIGIONS + DENOMINATIONS:
    CANON[key(_c)] = _c
for _k, _v in VARIANTS.items():
    CANON[_k] = _v


def to_int(s):
    return int(re.sub(r"[\s   ]", "", s))


def fetch():
    os.makedirs(RAW, exist_ok=True)
    ctx = ssl.create_default_context()
    # See the note on BASE. The chain, not the content, is what a plain client trips on.
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for _gid, stem, name, _rep in PROVINCES:
        dest = os.path.join(RAW, stem + ".pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"have {stem}.pdf ({os.path.getsize(dest):,} bytes)")
            continue
        url = BASE + stem + ".pdf"
        print("GET", url)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        body = urllib.request.urlopen(req, context=ctx, timeout=600).read()
        # §5a: a 200 is not a download. The Imperva stub is a ~1 KB HTML body served with a
        # 200, so the magic and the trailer are both checked ([[reference_pdf_truncated_at_source]]).
        if not body.startswith(b"%PDF"):
            raise SystemExit(f"{url} did not return a PDF -- {len(body):,} bytes "
                             f"starting {body[:40]!r}")
        if b"%%EOF" not in body[-4096:]:
            raise SystemExit(f"{url} has no %%EOF trailer -- truncated at source, "
                             f"{len(body):,} bytes")
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, dest)
        print(f"  {name}: {len(body):,} bytes")


def find_table(doc, needle):
    """Return the lines following the caption whose TITLE contains `needle`.

    Skips the front matter, because the contents page carries the same caption text with a
    dot leader and a page number after it.
    """
    for i, page in enumerate(doc):
        if i < 10:
            continue
        lines = page.get_text().split("\n")
        for j, line in enumerate(lines):
            m = CAPTION.match(line.strip())
            if not m or needle not in m.group(1).lower() or "......" in line:
                continue
            return i, line.strip(), lines[j + 1:]
    return None, None, None


def parse(lines, province, tab):
    """Rows are label / count / percent, one per line, the label sometimes wrapping.

    Terminates on a row whose label is `Total` OR the province's own name -- see the module
    docstring; five of the nine reports use the latter.
    """
    ends = {"total", province.lower(), province.lower().replace("-", " ")}
    rows, label, total = [], [], None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line in DASH and label:
            rows.append({"label": " ".join(" ".join(label).split()), "n": 0,
                         "pct": 0.0, "dash": True})
            label = []
            if i + 1 < len(lines) and lines[i + 1].strip() in DASH:
                i += 1
        elif COUNT.match(line) and label:
            name = " ".join(" ".join(label).split())
            n = to_int(line)
            pct = None
            if i + 1 < len(lines) and PCT.match(lines[i + 1].strip()):
                pct = float(lines[i + 1].strip().replace(",", "."))
                i += 1
            if name.lower().strip(" .:") in ends:
                total = n
                i += 1
                break
            rows.append({"label": name, "n": n, "pct": pct, "dash": False})
            label = []
        elif not COUNT.match(line):
            if line.lower().strip(" .:") in HEADER[tab]:
                i += 1
                continue
            label.append(line)
        i += 1
    return rows, total


def read_province(stem, name, report):
    import fitz

    path = os.path.join(RAW, stem + ".pdf")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)
    # PyMuPDF reports page_count=0 on a file damaged at source rather than raising
    # ([[reference_pdf_truncated_at_source]]).
    if doc.page_count == 0:
        raise SystemExit(f"{path}: PyMuPDF reports zero pages -- damaged download")

    # Every page carries the running header `Provincial profile: <name> [Community Survey
    # 2016], Report <nnn>`, so the province AND its report number are both checkable
    # against the file actually opened. Eastern Cape's bibliography cites 03-01-63 as well,
    # so the test is that the expected number is the one in the running header -- i.e. the
    # most frequent -- rather than merely present somewhere.
    seen = {}
    for i in range(min(40, doc.page_count)):
        for m in re.findall(r"0\d-\d\d-\d\d", doc[i].get_text()):
            seen[m] = seen.get(m, 0) + 1
    if not seen:
        raise SystemExit(f"{path}: no report number anywhere in the first 40 pages")
    top = max(seen, key=lambda k: seen[k])
    if top != report:
        raise SystemExit(f"{path}: running header says report {top}, PROVINCES says "
                         f"{report} -- the wrong province's PDF is at this filename, or "
                         "Stats SA has renumbered")
    head = "".join(doc[i].get_text() for i in range(min(5, doc.page_count)))
    if name.lower() not in head.lower():
        raise SystemExit(f"{path}: {name!r} does not appear in the first five pages")

    out = {}
    for tab, needle, expect in (("a", "religious affiliation", RELIGIONS),
                                ("b", "christian denomination", DENOMINATIONS)):
        pg, cap, lines = find_table(doc, needle)
        if lines is None:
            raise SystemExit(f"{name}: no table captioned {needle!r} -- the report's table "
                             "numbering or wording has changed")
        rows, total = parse(lines, name, tab)
        if total is None:
            raise SystemExit(f"{name} 2.10{tab}: no total row found (looked for 'Total' "
                             f"and {name!r})")
        got = {}
        for r in rows:
            k = key(r["label"])
            canon = CANON.get(k)
            if canon is None:
                raise SystemExit(f"{name} 2.10{tab}: unrecognised category "
                                 f"{r['label']!r} (folded to {k!r})")
            if canon in got:
                raise SystemExit(f"{name} 2.10{tab}: {canon!r} appears twice")
            got[canon] = r
        missing = [c for c in expect if c not in got]
        extra = [c for c in got if c not in expect]
        if missing or extra:
            raise SystemExit(f"{name} 2.10{tab}: missing={missing} extra={extra}")
        out[tab] = {"page": pg, "caption": cap, "rows": got, "total": total}
    doc.close()
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()

    print("South Africa - Community Survey 2016 provincial profiles, Stats SA")
    print(f"  {len(PROVINCES)} reports from {RAW}\n")

    data, notes = {}, {}
    for gid, stem, name, rep in PROVINCES:
        rec = read_province(stem, name, rep)
        data[gid] = rec
        notes[gid] = rep
        a, b = rec["a"], rec["b"]
        sa = sum(r["n"] for r in a["rows"].values())
        sb = sum(r["n"] for r in b["rows"].values())
        chr_ = a["rows"]["Christianity"]["n"]
        resid = chr_ - sb
        print(f"  {gid} {name:15s} {rep}  2.10a p{a['page']:<3d} {sa:>10,}  "
              f"2.10b p{b['page']:<3d} {sb:>10,}  residual {resid:>9,} "
              f"({100.0 * resid / chr_:5.2f}% of Christians)")
        # The published totals are weighted and rounded; a row sum can be one or two people
        # off its own printed total and that is the report, not the parse.
        if abs(sa - a["total"]) > 2:
            raise SystemExit(f"{name} 2.10a: rows sum to {sa:,}, printed total "
                             f"{a['total']:,}")
        if resid < 0:
            raise SystemExit(f"{name}: table 2.10b holds MORE people ({sb:,}) than table "
                             f"2.10a's Christianity cell ({chr_:,}) -- the two tables are "
                             "not about the same universe and nothing here is safe")

    # ---- the one province whose own table does not add up (§ docstring) ----
    nw = data["ZA06"]
    nw_rows = sum(r["n"] for r in nw["b"]["rows"].values())
    if nw_rows >= nw["b"]["total"] - 2:
        raise SystemExit("North West's table 2.10b now reconciles. That is good news and "
                         "it means Stats SA has reissued Report 03-01-11, so re-read the "
                         "module docstring before trusting the residual treatment.")
    print(f"\n  KNOWN DEFECT, Report 03-01-11 (North West): table 2.10b's fourteen rows sum "
          f"to\n    {nw_rows:,} against its own printed total of {nw['b']['total']:,}; its "
          f"printed percentages sum to\n    "
          f"{sum(r['pct'] or 0 for r in nw['b']['rows'].values()):.1f}, not 100.0. The "
          f"{nw['b']['total'] - nw_rows:,} unaccounted people go to "
          f"'{NOT_REPORTED}'\n    with everybody else's, rather than being assigned to a "
          "denomination on a guess.")

    # ---- write ----
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows_out = []
    for gid, stem, name, rep in PROVINCES:
        rec = data[gid]
        for canon in RELIGIONS:
            if canon == "Christianity":
                continue                       # replaced by the fourteen denominations
            r = rec["a"]["rows"][canon]
            rows_out.append((gid, "province", name, f"Religion: {canon}", r["n"],
                             f"level=province; Report {rep} table 2.10a"
                             + ("; printed as a dash, read as zero" if r["dash"] else "")))
        for canon in DENOMINATIONS:
            r = rec["b"]["rows"][canon]
            src = fold(r["label"])
            exemplar = ""
            m = re.search(r"\((?:e\.?g\.?|eg)[^)]*\)?", r["label"], flags=re.I)
            if m:
                exemplar = "; " + " ".join(m.group(0).split())
            rows_out.append((gid, "province", name, f"Christian: {canon}", r["n"],
                             f"level=province; Report {rep} table 2.10b{exemplar}"))
        chr_ = rec["a"]["rows"]["Christianity"]["n"]
        short = rec["b"]["total"] - sum(r["n"] for r in rec["b"]["rows"].values())
        resid = chr_ - sum(r["n"] for r in rec["b"]["rows"].values())
        note = (f"level=province; Report {rep} table 2.10a Christianity less table "
                "2.10b's rows")
        # North West's residual is mostly a defect in its own report, and the REVIEW dict
        # is not on the map. Say so in the row itself, or this province reads as a finding.
        if short > 2:
            note += (f"; {short:,} of this is Report {rep}'s own shortfall, its 14 rows "
                     "summing to 90.1% of its printed total, see taxonomy/za2016.py")
        rows_out.append((gid, "province", name, f"Christian: {NOT_REPORTED}", resid, note))

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count",
                    "basis", "year", "source_id", "note"])
        for gid, lvl, name, cat, n, note in rows_out:
            w.writerow([gid, lvl, name, cat, n, BASIS, YEAR, SOURCE_ID, note])

    total = sum(r[4] for r in rows_out)
    cats = sorted({r[3] for r in rows_out})
    print(f"\nwrote {OUT}")
    print(f"  {len(rows_out)} rows, {len(PROVINCES)} provinces, {len(cats)} categories, "
          f"{total:,} people")
    if len(cats) != len(RELIGIONS) - 1 + len(DENOMINATIONS) + 1:
        raise SystemExit(f"expected {len(RELIGIONS) - 1 + len(DENOMINATIONS) + 1} "
                         f"categories, got {len(cats)}")

    # ---- what the country looks like, for the record ----
    nat = {}
    for _g, _l, _n, cat, n, _note in rows_out:
        nat[cat] = nat.get(cat, 0) + n
    print(f"\n  national, {total:,} people who gave an answer:")
    for cat in sorted(nat, key=lambda c: -nat[c]):
        print(f"    {cat:52s} {nat[cat]:>11,}  {100.0 * nat[cat] / total:5.2f}%")


if __name__ == "__main__":
    main()
